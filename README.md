# B-BIND

**Biophysical Bayesian Inference for Neurodegenerative Dynamics**

A lightweight, configurable implementation of B-BIND for inferring continuous
disease pseudotime from cross-sectional quantitative neuropathology data.

This repository contains the single-region B-BIND model as described in
Agrawal et al. (2026), adapted for general use with configurable observation
likelihoods and ordinal staging variables.

---

## Citation

If you use this code in published research, please cite:

> Agrawal, A., Rachleff, V. M., Travaglini, K. J., Mukherjee, S., Crane, P. K.,
> Hawrylycz, M., Keene, C. D., Lein, E., Mena, G. E., & Gabitto, M. I. (2026).
> B-BIND: Biophysical Bayesian inference for neurodegenerative dynamics.
> *The Annals of Applied Statistics*, 20(1), 285–306.
> https://doi.org/10.1214/25-AOAS2078

BibTeX:
```bibtex
@article{agrawal2026b,
  title={B-BIND: Biophysical Bayesian inference for neurodegenerative dynamics},
  author={Agrawal, Anamika and Rachleff, Victoria M and Travaglini, Kyle J and Mukherjee, Shubhabrata and Crane, Paul K and Hawrylycz, Michael and Keene, C Dirk and Lein, Ed and Mena, Gonzalo E and Gabitto, Mariano I},
  journal={The Annals of Applied Statistics},
  volume={20},
  number={1},
  pages={285--306},
  year={2026},
  publisher={Institute of Mathematical Statistics}
}
```

---

## Overview

B-BIND infers a continuous latent pseudotime `t ∈ [0, 1]` per donor from
quantitative neuropathology measurements. The generative model is:

```
t_d   ~ Beta(1, 1)                          donor pseudotime
beta0 ~ Normal(beta0_loc, beta0_scale)       per measurement × region
beta1 ~ HalfNormal(beta1_scale)             magnitude of change (always positive)
A     ~ HalfNormal(A_scale)                 NegBin dispersion

# Exponential-link measurements (counts, densities):
X_d ~ NegBin(logits = beta0 + direction * beta1 * t_d,  total_count = A)

# Sigmoid-link measurements (saturating):
X_d ~ NegBin(probs  = sigmoid(beta0 + direction * beta1 * t_d),  total_count = A)

# Ordinal staging variables:
stage_d ~ OrderedLogistic(eta = offset + scale * t0c_d,  cutpoints)
```

`beta1_direction` (+1 or -1) is fixed by the user based on domain knowledge —
it is not inferred. This ensures pseudotime is identifiable.

---

## Installation

For the conda environment:

```bash
conda env create -f environment.yml
conda activate bbind-env
```

---

## Usage

See `notebooks/run_bbind_seaad_mtg.py` for a worked example using SEA-AD
Middle Temporal Gyrus data. The notebook walks through:

1. Configuration (measurement types, per-measurement priors)
2. Data loading and preparation
3. Data-driven prior suggestions
4. MCMC inference with convergence filtering
5. Posterior inspection and validation plots

For the measurement config schema, see `MEASUREMENT_CONFIG.md`.

**SEA-AD data** is publicly available at:
https://portal.brain-map.org/explore/seattle-alzheimers-disease

---

## Repository structure

```
bbind/
    model.py          generative model (configurable obs likelihoods)
    inference.py      MCMC wrapper with convergence filtering
    utils.py          data preparation and prior suggestion utilities
notebooks/
    run_bbind_seaad_mtg.py    worked example on SEA-AD MTG data
MEASUREMENT_CONFIG.md         config schema and examples
environment.yml
LICENSE
CITATION.cff
```

---

## License

Apache License 2.0. See `LICENSE` for details.
