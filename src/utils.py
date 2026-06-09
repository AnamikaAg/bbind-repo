"""
bbind/utils.py  —  Data preparation for B-BIND.

Converts a merged neuropath + metadata DataFrame into the arrays
the model expects: scaled integer (M, L, T) data blocks, boolean masks,
and 0-indexed integer staging vectors.

Follows the original B-BIND data prep convention:
  1. Scale each feature to [0, 1] with zero_center=False  (sc.pp.scale equivalent)
  2. Multiply by 1000 and cast to int32  (inflated counts for NB likelihood)
  3. Order donors by PC1 of the full feature matrix  (warm-start heuristic)
"""

import numpy as np
import numpy.ma as ma
import pandas as pd


# ---------------------------------------------------------------------------
# PCA-based donor ordering (Mariano's warm-start heuristic)
# ---------------------------------------------------------------------------

def get_pca_donor_order(df_features):
    """
    Order donors by PC1 of the (zero-center=False) scaled feature matrix.
    Returns an index array `initial_cond` of length T.

    Parameters
    ----------
    df_features : DataFrame (T, total_features)
        All continuous neuropath features (no metadata columns).
    """
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.decomposition import PCA

    X = MaxAbsScaler().fit_transform(df_features.values.astype(float))
    pc1 = PCA(n_components=1).fit_transform(X)[:, 0]
    return np.argsort(pc1)


# ---------------------------------------------------------------------------
# Feature block preparation
# ---------------------------------------------------------------------------

def _scale_and_integerise(matrix, scale=1000):
    """
    Scale each column to [0,1] (max-abs, zero_center=False),
    multiply by `scale`, cast to int32.
    Matches sc.pp.scale(zero_center=False) + *1000 + int32 cast.
    """
    col_max = np.nanmax(np.abs(matrix), axis=0, keepdims=True)
    col_max = np.where(col_max == 0, 1.0, col_max)   # avoid div-by-zero
    scaled  = matrix / col_max
    return np.int32(np.round(scaled * scale))


def prepare_feature_block(df, feature_names, region_cols, initial_cond, scale=1000):
    """
    Build a scaled integer array of shape (M, L, T) for one feature block.

    Parameters
    ----------
    df : DataFrame (T, ...)
        Full merged DataFrame, already filtered to one region per row
        OR with region encoded as a suffix in column names.
    feature_names : list of str
        Base feature names (e.g. ["percent 6e10 positive area", ...]).
        These are matched as substrings in df.columns.
    region_cols : list of str
        Region identifiers as they appear in column names, in order.
        E.g. ["MTG"] for single-region, or ["HIP","MTG",...] for multi.
    initial_cond : array (T,)
        Donor ordering index from get_pca_donor_order.
    scale : int
        Integer inflation factor (default 1000).

    Returns
    -------
    data  : int32 array (M, L, T)
    mask  : bool array (M, L, T)   — True where data is not NaN
    """
    M = len(feature_names)
    L = len(region_cols)
    T = len(initial_cond)

    data = np.zeros((M, L, T), dtype=np.int32)
    mask = np.ones((M, L, T), dtype=bool)

    for m, feat in enumerate(feature_names):
        # find columns matching this feature across regions
        matching = [c for c in df.columns if feat in c]
        for l, region in enumerate(region_cols):
            # pick the column for this region
            region_cols_match = [c for c in matching if region in c]
            if len(region_cols_match) == 0:
                mask[m, l, :] = False
                continue
            col = region_cols_match[0]
            raw = df[col].values.astype(float)[initial_cond]   # reorder donors
            nan_mask = np.isnan(raw)
            mask[m, l, nan_mask] = False
            raw = np.where(nan_mask, 0.0, raw)
            # scale column independently
            col_max = np.max(np.abs(raw)) or 1.0
            data[m, l, :] = np.int32(np.round((raw / col_max) * scale))

    return data, ma.make_mask(mask)


# ---------------------------------------------------------------------------
# Staging data preparation
# ---------------------------------------------------------------------------

# String → integer maps for SEA-AD / standard staging variables
STAGING_MAPS = {
    "Braak": {
        "Braak 0": 0, "Braak I": 0, "Braak II": 1,
        "Braak III": 2, "Braak IV": 3, "Braak V": 4, "Braak VI": 5,
    },
    "Thal": {
        "Thal 0": 0, "Thal 1": 1, "Thal 2": 2,
        "Thal 3": 3, "Thal 4": 4, "Thal 5": 5,
    },
    "ADNC": {
        "Not AD": 0, "Low": 1, "Intermediate": 2, "High": 3,
    },
    "CERAD": {
        "None": 0, "Sparse": 1, "Moderate": 2, "Frequent": 3,
    },
}


def prepare_staging(df, staging_config, initial_cond):
    """
    Build integer staging arrays for all ordinal variables in the config.

    Parameters
    ----------
    df : DataFrame (T, ...)
        Must contain a column matching each staging variable's `col` key.
    staging_config : list of dict
        Subset of measurement_config where type == "ordinal".
        Each entry must have:
            "name"       : variable name (must match a key in STAGING_MAPS
                           OR provide a "col" and "value_map" override)
            "col"        : DataFrame column name (defaults to name if absent)
            "value_map"  : dict str->int (defaults to STAGING_MAPS[name])
    initial_cond : array (T,)
        Donor ordering index.

    Returns
    -------
    staging_obs   : dict {name: int array (T,)}   — 0-indexed, reordered
    staging_masks : dict {name: bool array (T,)}  — True where not NaN / unknown
    """
    staging_obs   = {}
    staging_masks = {}

    for cfg in staging_config:
        name      = cfg["name"]
        col       = cfg.get("col", name)
        value_map = cfg.get("value_map", STAGING_MAPS.get(name, {}))

        raw = df[col].values[initial_cond]
        integers = np.full(len(raw), -1, dtype=np.int32)
        valid    = np.zeros(len(raw), dtype=bool)

        for i, v in enumerate(raw):
            if pd.isna(v):
                continue
            key = str(v).strip()
            if key in value_map:
                integers[i] = value_map[key]
                valid[i]    = True

        staging_obs[name]   = integers
        staging_masks[name] = valid

    return staging_obs, staging_masks


# ---------------------------------------------------------------------------
# Data-driven prior suggestions
# ---------------------------------------------------------------------------

def suggest_priors(data, mask, feature_names, early_frac=0.25, late_frac=0.25):
    """
    Suggest per-measurement prior values from data, using the early and late
    donors (by current ordering, which follows PC1) as proxies for low and
    high pseudotime.

    Parameters
    ----------
    data : int array (M, L, T)
        Scaled integer data block (output of prepare_feature_block).
    mask : bool array (M, L, T)
    feature_names : list of str, length M
    early_frac : float
        Fraction of donors (from the left of the ordering) treated as early.
    late_frac : float
        Fraction of donors (from the right) treated as late.

    Returns
    -------
    suggestions : list of dict, one per measurement
        Each dict has keys: name, beta0_loc, beta0_scale, beta1_scale,
        beta1_direction.  Use as a starting point — override as needed.
    """
    M, L, T = data.shape
    n_early = max(1, int(T * early_frac))
    n_late  = max(1, int(T * late_frac))

    suggestions = []
    for m, name in enumerate(feature_names):
        # Average over valid observations across regions
        vals = data[m].astype(float)          # (L, T)
        valid = mask[m]                        # (L, T)
        vals_masked = np.where(valid, vals, np.nan)

        # Mean per donor across regions (ignore NaN)
        donor_mean = np.nanmean(vals_masked, axis=0)   # (T,)

        early_mean = np.nanmean(donor_mean[:n_early])
        late_mean  = np.nanmean(donor_mean[-n_late:])

        # beta0_loc: log of early-stage mean count (clamped away from 0)
        baseline    = max(early_mean, 1.0)
        beta0_loc   = float(np.log(baseline))
        beta0_scale = 3.0   # broad but not uninformative

        # beta1_scale: log-ratio of late to early mean — rough dynamic range
        late_clamped = max(late_mean, 1.0)
        log_ratio    = abs(np.log(late_clamped) - np.log(baseline))
        beta1_scale  = float(max(log_ratio, 1.0))   # at least 1.0

        # beta1_direction: sign of change
        beta1_direction = 1 if late_mean >= early_mean else -1

        suggestions.append({
            "name":            name,
            "beta0_loc":       round(beta0_loc,  2),
            "beta0_scale":     beta0_scale,
            "beta1_scale":     round(beta1_scale, 2),
            "beta1_direction": beta1_direction,
        })

    return suggestions


def print_prior_suggestions(suggestions):
    """Pretty-print suggest_priors() output for copy-paste into config."""
    print("Suggested per-measurement priors (edit before using):\n")
    for s in suggestions:
        direction_str = "+1 (increases)" if s["beta1_direction"] == 1 else "-1 (decreases)"
        print(f"  {s['name']}")
        print(f"    beta0_loc={s['beta0_loc']:.2f},  beta0_scale={s['beta0_scale']:.1f}")
        print(f"    beta1_scale={s['beta1_scale']:.2f},  beta1_direction={direction_str}")
        print()