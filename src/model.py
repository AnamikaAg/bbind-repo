"""
bbind/model.py  —  Single-region B-BIND model.

See MEASUREMENT_CONFIG.md for config schema and examples.
"""

import jax.numpy as jnp
from jax import lax
from jax.nn import softplus
import numpyro as npp
import numpyro.distributions as dist


def make_bbind_model(measurement_config, A_scale=500.0):
    """
    Build a B-BIND NumPyro model from a measurement config list.

    Per-measurement priors are read directly from each config entry:
        beta0_loc, beta0_scale  : Normal prior on intercept
        beta1_scale             : HalfNormal scale on growth/decline magnitude
        beta1_direction         : +1 (increases with disease) or -1 (decreases)

    Use suggest_priors() to get data-driven starting values for these,
    then override per measurement as needed before calling this function.

    A_scale : float
        Shared HalfNormal scale for NegBin dispersion A, applied to all
        continuous measurements. Adjust if your scaled counts are much
        larger or smaller than the default (500).

    Data arrays passed to the returned model have shape (M, L, T):
        M = number of measurements of that type
        L = number of regions
        T = number of donors

    Returns
    -------
    model : callable
        model(data_exp, data_sigm, mask_exp, mask_sigm,
              staging_obs=None, staging_masks=None)
    """

    exp_cfg  = [c for c in measurement_config
                if c["type"] == "continuous" and c["obs_model"] == "negbin_exp"]
    sigm_cfg = [c for c in measurement_config
                if c["type"] == "continuous" and c["obs_model"] == "negbin_sigmoid"]
    ord_cfg  = [c for c in measurement_config if c["type"] == "ordinal"]

    M_exp  = len(exp_cfg)
    M_sigm = len(sigm_cfg)

    # Build (M,) arrays of per-measurement prior values
    # so they can be broadcast into the plate cleanly
    def _prior_array(cfg_list, key, default):
        import jax.numpy as jnp
        return jnp.array([float(c.get(key, default)) for c in cfg_list])

    exp_beta0_loc   = _prior_array(exp_cfg,  "beta0_loc",       0.0)
    exp_beta0_scale = _prior_array(exp_cfg,  "beta0_scale",     5.0)
    exp_beta1_scale = _prior_array(exp_cfg,  "beta1_scale",     5.0)
    exp_direction   = _prior_array(exp_cfg,  "beta1_direction", 1.0)  # (+1 or -1)

    sigm_beta0_loc   = _prior_array(sigm_cfg, "beta0_loc",       0.0)
    sigm_beta0_scale = _prior_array(sigm_cfg, "beta0_scale",     5.0)
    sigm_beta1_scale = _prior_array(sigm_cfg, "beta1_scale",     5.0)
    sigm_direction   = _prior_array(sigm_cfg, "beta1_direction", 1.0)

    def model(data_exp, data_sigm, mask_exp, mask_sigm,
              staging_obs=None, staging_masks=None):

        _, L, T = mask_exp.shape

        # ── 1. Donor pseudotime ────────────────────────────────────────────
        with npp.plate("donors", T):
            times = npp.sample("times", dist.Beta(jnp.ones(1), jnp.ones(1)))  # (T,)

        # ── 2. Exponential-link block  (M_exp, L, T) ──────────────────────
        # mean = A * exp(direction * beta1 * t + beta0)
        # NegativeBinomialLogits(logits = beta0 + direction*beta1*t, total_count=A)
        if M_exp > 0:
            with npp.plate("meas_exp", M_exp, dim=-2):
                with npp.plate("regions_exp", L, dim=-1):
                    beta0_exp = npp.sample(
                        "beta0_exp",
                        dist.Normal(
                            exp_beta0_loc[:, None] * jnp.ones((M_exp, L)),
                            exp_beta0_scale[:, None] * jnp.ones((M_exp, L)),
                        ),
                    )  # (M_exp, L)
                    beta1_exp = npp.sample(
                        "beta1_exp",
                        dist.HalfNormal(
                            exp_beta1_scale[:, None] * jnp.ones((M_exp, L)),
                        ),
                    )  # (M_exp, L)  — always positive; direction applied below
                    A_exp = npp.sample(
                        "A_exp",
                        dist.HalfNormal(A_scale),
                    )  # (M_exp, L)

            logits_exp = (
                beta0_exp[:, :, None]
                + exp_direction[:, None, None] * beta1_exp[:, :, None] * times[None, None, :]
            )  # (M_exp, L, T)

            with npp.handlers.mask(mask=mask_exp):
                with npp.plate("obs_T_exp", T, dim=-1):
                    with npp.plate("obs_L_exp", L, dim=-2):
                        with npp.plate("obs_M_exp", M_exp, dim=-3):
                            npp.sample(
                                "obs_exp",
                                dist.NegativeBinomialLogits(
                                    logits=logits_exp,
                                    total_count=A_exp[:, :, None],
                                ),
                                obs=data_exp,
                            )

        # ── 3. Sigmoid-link block  (M_sigm, L, T) ─────────────────────────
        # mean = A * sigmoid(beta0 + direction * beta1 * t)
        # NegativeBinomialProbs(probs=sigmoid(...), total_count=A)
        if M_sigm > 0:
            with npp.plate("meas_sigm", M_sigm, dim=-2):
                with npp.plate("regions_sigm", L, dim=-1):
                    beta0_sigm = npp.sample(
                        "beta0_sigm",
                        dist.Normal(
                            sigm_beta0_loc[:, None] * jnp.ones((M_sigm, L)),
                            sigm_beta0_scale[:, None] * jnp.ones((M_sigm, L)),
                        ),
                    )  # (M_sigm, L)
                    beta1_sigm = npp.sample(
                        "beta1_sigm",
                        dist.HalfNormal(
                            sigm_beta1_scale[:, None] * jnp.ones((M_sigm, L)),
                        ),
                    )  # (M_sigm, L)
                    A_sigm = npp.sample(
                        "A_sigm",
                        dist.HalfNormal(A_scale),
                    )  # (M_sigm, L)

            probs_sigm = jnp.clip(
                lax.logistic(
                    beta0_sigm[:, :, None]
                    + sigm_direction[:, None, None] * beta1_sigm[:, :, None] * times[None, None, :]
                ),
                1e-6, 1 - 1e-6,
            )  # (M_sigm, L, T)

            with npp.handlers.mask(mask=mask_sigm):
                with npp.plate("obs_T_sigm", T, dim=-1):
                    with npp.plate("obs_L_sigm", L, dim=-2):
                        with npp.plate("obs_M_sigm", M_sigm, dim=-3):
                            npp.sample(
                                "obs_sigm",
                                dist.NegativeBinomialProbs(
                                    probs=probs_sigm,
                                    total_count=A_sigm[:, :, None],
                                ),
                                obs=data_sigm,
                            )

        # ── 4. Ordinal staging likelihoods ─────────────────────────────────
        if ord_cfg:
            t0_norm = (times - jnp.min(times)) / (
                jnp.max(times) - jnp.min(times) + 1e-8
            )
            t0c = npp.deterministic("t0c", t0_norm - jnp.median(t0_norm))

            for cfg in ord_cfg:
                name = cfg["name"]
                obs  = staging_obs.get(name)   if staging_obs   else None
                mask = (staging_masks.get(name) if staging_masks
                        else jnp.ones(T, dtype=bool))
                _obs_ordered_logistic(
                    name                 = name,
                    t0c                  = t0c,
                    mask                 = mask,
                    obs                  = obs,
                    n_categories         = cfg["n_categories"],
                    prior_cutpoint_scale = cfg.get("prior_cutpoint_scale", 1.0),
                    prior_eta_scale      = cfg.get("prior_eta_scale", 3.0),
                )

    return model


# ---------------------------------------------------------------------------
# Ordinal likelihood
# ---------------------------------------------------------------------------

def _obs_ordered_logistic(name, t0c, mask, obs, n_categories,
                           prior_cutpoint_scale=1.0, prior_eta_scale=3.0):
    """
    Ordered logistic staging likelihood.

    Predictor:  eta = offset + scale * t0c
    Cutpoints:  K-1 ordered thresholds via positive-gap cumsum, centred near 0.
    obs must be 0-indexed integers in {0, ..., n_categories-1}.
    """
    T = t0c.shape[0]
    K = int(n_categories)

    offset = npp.sample(f"offset_{name}", dist.Normal(0.0, 0.5))
    scale  = npp.sample(f"scale_{name}",  dist.HalfNormal(float(prior_eta_scale)))
    eta    = npp.deterministic(f"eta_{name}", offset + scale * t0c)

    c0       = npp.sample(f"c0_{name}",
                          dist.Normal(0.0, float(prior_cutpoint_scale)))
    gaps_raw = npp.sample(f"gaps_raw_{name}",
                          dist.Normal(0.0, 0.6).expand([K - 1]).to_event(1))
    gaps     = softplus(gaps_raw)
    cumgaps  = jnp.cumsum(gaps)
    cutpoints = npp.deterministic(
        f"cutpoints_{name}",
        c0 + cumgaps - jnp.mean(cumgaps),
    )

    with npp.handlers.mask(mask=mask):
        with npp.plate(f"donors_{name}", T):
            npp.sample(
                f"obs_{name}",
                dist.OrderedLogistic(predictor=eta, cutpoints=cutpoints),
                obs=obs,
            )

