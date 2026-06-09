# Measurement Configuration

The `measurement_config` list passed to `make_bbind_model()` describes every
measurement in your dataset. Each entry is a dict with a `type` field that
determines which fields are required.

---

## Continuous measurements

Set `"type": "continuous"` and choose an `obs_model`:

| `obs_model`       | Likelihood | Mean function | Suitable for |
|-------------------|------------|---------------|--------------|
| `"negbin_exp"`    | NegBin     | `A · exp(β₀ + dir · β₁ · t)` | counts and densities that grow/decline from near-zero |
| `"negbin_sigmoid"`| NegBin     | `A · σ(β₀ + dir · β₁ · t)`   | measurements that saturate at a ceiling |

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Must match a substring of the DataFrame column name |
| `type` | str | `"continuous"` |
| `obs_model` | str | `"negbin_exp"` or `"negbin_sigmoid"` |

### Prior fields (all optional — defaults shown)

| Field | Default | Description |
|-------|---------|-------------|
| `beta0_loc` | `0.0` | Normal prior mean on log-baseline (exp) or logit-baseline (sigmoid) at t=0 |
| `beta0_scale` | `5.0` | Normal prior scale on baseline |
| `beta1_scale` | `5.0` | HalfNormal scale on magnitude of change along pseudotime |
| `beta1_direction` | `1` | `+1` if marker increases with disease; `-1` if it decreases |

`beta1` is always sampled as a **positive** magnitude (HalfNormal). The
direction of change is fixed by `beta1_direction` and is not inferred.
This keeps pseudotime identifiable — t always runs 0 → 1, and the sign
of the relationship is specified by the user based on domain knowledge.

Use `suggest_priors()` (see `bbind/utils.py`) to get data-driven starting
values for `beta0_loc`, `beta1_scale`, and `beta1_direction`, then adjust
`beta1_direction` as needed before running inference.

### Example

```python
# Marker that increases with disease (tau pathology)
{"name":             "percent AT8 positive area",
 "type":             "continuous",
 "obs_model":        "negbin_exp",
 "beta0_loc":        -2.3,   # low baseline
 "beta0_scale":       3.0,
 "beta1_scale":       4.0,   # moderate dynamic range
 "beta1_direction":   1},    # increases

# Marker that decreases with disease (intact myelin)
{"name":             "LFB intact fraction",
 "type":             "continuous",
 "obs_model":        "negbin_sigmoid",
 "beta0_loc":         2.0,   # high at baseline
 "beta0_scale":       3.0,
 "beta1_scale":       3.0,
 "beta1_direction":  -1},    # decreases
```

---

## Ordinal (staging) measurements

Set `"type": "ordinal"` for discrete severity grades or staging variables.
These use an ordered logistic likelihood with pseudotime as the predictor.

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Variable name (used as prefix for all sampled parameters) |
| `type` | str | `"ordinal"` |
| `n_categories` | int | Number of ordered categories K (obs values must be 0 … K-1) |
| `col` | str | Exact DataFrame column name |
| `value_map` | dict | Maps raw string values in the DataFrame to 0-indexed integers |

### Optional prior fields

| Field | Default | Description |
|-------|---------|-------------|
| `prior_cutpoint_scale` | `1.0` | Normal scale on the first cutpoint `c0` |
| `prior_eta_scale` | `3.0` | HalfNormal scale on the pseudotime effect magnitude |

### Example

```python
# Braak staging (6 categories: 0–5)
{"name":         "Braak",
 "type":         "ordinal",
 "n_categories":  6,
 "col":          "braak",          # exact column name in your DataFrame
 "value_map": {
     "Braak 0": 0, "Braak I": 0,   # Braak I mapped to 0 (rare in SEA-AD)
     "Braak II": 1, "Braak III": 2,
     "Braak IV": 3, "Braak V": 4, "Braak VI": 5,
 }},

# ADNC (4 categories)
{"name":         "ADNC",
 "type":         "ordinal",
 "n_categories":  4,
 "col":          "adneurochange",
 "value_map":    {"Not AD": 0, "Low": 1, "Intermediate": 2, "High": 3}},

# Fazekas white matter hyperintensity grade (4 categories: 0–3)
{"name":         "Fazekas",
 "type":         "ordinal",
 "n_categories":  4,
 "col":          "fazekas_grade",
 "value_map":    {0: 0, 1: 1, 2: 2, 3: 3}},   # already numeric
```

---

## Shared dispersion prior

`A_scale` is passed directly to `make_bbind_model()` (not per-measurement)
and sets the HalfNormal scale on the NegBin dispersion parameter `A` for
all continuous measurements. Adjust it to match the order of magnitude of
your scaled data:

```python
model = make_bbind_model(measurement_config, A_scale=500.0)
```

After `prepare_feature_block()` scales and multiplies by 1000, typical
values for neuropathology data are in the range 10–1000, so `A_scale=500`
is a reasonable broad prior. If your measurements are sparser or denser,
scale accordingly.
