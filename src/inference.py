"""
bbind/inference.py  —  MCMC inference loop with convergence filtering.
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import numpyro as npp
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import effective_sample_size as ess, split_gelman_rubin as r_hat
import arviz as az


def run_bbind(
    model,
    model_args,
    model_kwargs=None,
    *,
    num_warmup=5000,
    num_samples=1000,
    num_chains=1,
    n_accepted=5,
    ess_threshold=50,
    rhat_threshold=1.05,
    rng_seed=0,
    save_path=None,
):
    """
    Run B-BIND MCMC with convergence filtering on pseudotime (times).

    Repeats NUTS sampling with incrementing random seeds until `n_accepted`
    chains pass ESS and r-hat thresholds on the inferred donor pseudotimes.

    Parameters
    ----------
    model : NumPyro model callable (from make_bbind_model)
    model_args : tuple of positional args passed to model
    model_kwargs : dict of keyword args passed to model, or None
    num_warmup : int
    num_samples : int
    num_chains : int
    n_accepted : int
        Number of accepted runs to collect before stopping.
    ess_threshold : float
        Minimum acceptable ESS per donor pseudotime.
    rhat_threshold : float
        Maximum acceptable r-hat per donor pseudotime.
    rng_seed : int
        Starting random seed; incremented on each attempt.
    save_path : str or None
        If given, pickle results to this path.

    Returns
    -------
    results : dict with keys
        "posterior_samples" : list of n_accepted sample dicts
        "az_objects"        : list of n_accepted ArviZ InferenceData objects
        "model_args"        : the model_args passed in (for reproducibility)
    """
    if model_kwargs is None:
        model_kwargs = {}

    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
        jit_model_args=True,
    )

    accepted_samples = []
    accepted_az      = []
    attempt = 0
    seed    = rng_seed

    while len(accepted_samples) < n_accepted:
        rng_key = jax.random.PRNGKey(seed)
        print(f"Attempt {attempt + 1}  (seed={seed})  —  "
              f"{len(accepted_samples)}/{n_accepted} accepted so far")

        mcmc.run(rng_key, *model_args, **model_kwargs)
        samples = mcmc.get_samples()

        # Convergence check on donor pseudotimes only
        times_chain = samples["times"][np.newaxis, :, :]   # (1, S, T) for diagnostics
        ess_vals  = ess(times_chain)
        rhat_vals = r_hat(times_chain)

        if np.any(np.isnan(ess_vals)):
            print("  Rejected: ESS contains NaN")
        elif np.any(ess_vals < ess_threshold):
            print(f"  Rejected: min ESS = {np.min(ess_vals):.1f} < {ess_threshold}")
        elif np.any(rhat_vals > rhat_threshold):
            print(f"  Rejected: max r-hat = {np.max(rhat_vals):.3f} > {rhat_threshold}")
        else:
            print(f"  Accepted  (min ESS={np.min(ess_vals):.1f}, "
                  f"max r-hat={np.max(rhat_vals):.3f})")
            # Build ArviZ object with posterior predictive
            pp = Predictive(model, samples)(
                jax.random.PRNGKey(seed + 1000), *model_args, **model_kwargs
            )
            az_obj = az.from_numpyro(mcmc, posterior_predictive=pp)
            accepted_samples.append(samples)
            accepted_az.append(az_obj)

        seed    += 1
        attempt += 1

    results = {
        "posterior_samples": accepted_samples,
        "az_objects":        accepted_az,
        "model_args":        model_args,
    }

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved to {save_path}")

    return results


def get_times(results):
    """
    Concatenate pseudotime samples across all accepted runs.

    Returns
    -------
    times_all : array (n_accepted * num_samples, T)
    times_mean : array (T,)
    times_std  : array (T,)
    """
    times_all = np.concatenate(
        [r["times"] for r in results["posterior_samples"]], axis=0
    )
    return times_all, times_all.mean(0), times_all.std(0)
