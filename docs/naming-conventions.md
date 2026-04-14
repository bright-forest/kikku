# Naming Conventions for Stage Operators and Value Objects

Code-level naming conventions for `kikku`-based models, grounded in the bellman-ddsl symbol reference. The retirement choice model is the running example.

## Guiding principles

1. **Perch suffixes are mandatory on arrays**: `_arvl`, `_dcsn`, `_cntn` tell you *where* in the stage timeline the quantity lives.
2. **Refinement suffixes distinguish EGM pipeline stages**: `_hat` (raw EGM correspondence), `_ref` (after upper-envelope cleaning).
3. **Operator/function names mirror the YAML equation blocks**: `cntn_to_dcsn_mover`, `dcsn_to_arvl_mover`, etc.
4. **Stage names are domain nouns** (`work_cons`, `retire_cons`, `labour_mkt_decision`), not implementation labels.
5. **Sub-equation callables carry `fn_` prefix** to distinguish them from array data.

### Alternative short-form perch suffixes (Matsya / StageClass)

The bellman-ddsl codebase also uses a shorter suffix convention in some contexts:

| Long form | Short form | Math | Notes |
|-----------|------------|------|-------|
| `_arvl` | `_a` | $\prec$ | `x_a` = arrival state |
| `_dcsn` | `_v` | $\circ$ | `x_v` = decision state ("v" for value perch) |
| `_cntn` | `_e` | $\succ$ | `x_e` = continuation state ("e" for end) |

**Recommendation**: use the long form (`_arvl`, `_dcsn`, `_cntn`) in `kikku`/FUES code for clarity. The short form is acceptable in performance-critical inner loops or when following existing bellman-dev conventions.

## Perch index alphabet

From the bellman-ddsl symbol reference:

| Perch | Math | YAML tag | Code suffix | Description |
|-------|------|----------|-------------|-------------|
| Arrival | $\prec$ | `[<]` | `_arvl` | Known before decisions; prestate variables |
| Decision | $\circ$ | (unmarked) | `_dcsn` | After shock observation; full info for controls |
| Continuation | $\succ$ | `[>]` | `_cntn` | After all decisions and post-decision shocks |

## Value and state arrays

### Core convention

```
{quantity}_{perch}
{quantity}_{perch}_hat     # raw EGM (before upper envelope)
{quantity}_{perch}_ref     # refined (after upper envelope)
```

### Quantities

| Code name | Math | Description |
|-----------|------|-------------|
| `v` | $\mathrm{v}$ | Value function |
| `dv` | $\partial\mathrm{v}$ | Marginal value (envelope condition) |
| `ddv` | $\partial^2\mathrm{v}$ | Second derivative of value |
| `c` | $c$ / $\kappa$ | Consumption (primary control) |
| `x` | $x$ | State variable (e.g. cash-on-hand) |
| `da` | $\delta_a$ | Concavity diagnostic / policy gradient |

### Full naming examples (retirement model)

| Current code | Proposed | Perch | Refinement | Notes |
|-------------|----------|-------|------------|-------|
| `q_hat` | `v_dcsn_hat` | dcsn | raw EGM | Value correspondence from EGM inversion |
| `c_hat` | `c_dcsn_hat` | dcsn | raw EGM | Consumption from EGM inversion |
| `egrid` | `x_dcsn_hat` | dcsn | raw EGM | Endogenous grid from EGM inversion |
| `da_pre_ue` | `da_dcsn_hat` | dcsn | raw EGM | Concavity factor before envelope |
| `v` (work_cons) | `v_arvl` | arvl | refined | Value at arrival (on `asset_grid_A`) |
| `c` (work_cons) | `c_arvl` | arvl | refined | Consumption at arrival |
| `da` (work_cons) | `da_arvl` | arvl | refined | Concavity at arrival |
| `v` (labour_mkt) | `v_arvl` | arvl | refined | Branching-stage arrival value |
| `dv` (labour_mkt) | `dv_arvl` | arvl | refined | Marginal value at arrival |
| `ddv` (labour_mkt) | `ddv_arvl` | arvl | refined | Second derivative at arrival |

### Continuation inputs (from previous period)

| Current code | Proposed | Notes |
|-------------|----------|-------|
| `dv_cntn` | `dv_cntn` | Marginal continuation value (from $t+1$ arrival) |
| `ddv_cntn` | `ddv_cntn` | Second derivative of continuation value |
| `v_cntn` | `v_cntn` | Continuation value level |

### FUES interface arrays

| Current code | Proposed | FUES param position | Description |
|-------------|----------|---------------------|-------------|
| `e_grid` / `x_dcsn_hat` | `x_dcsn_hat` | 1st | Raw endogenous grid |
| `vlu` / `v_hat` | `v_dcsn_hat` | 2nd | Raw value correspondence |
| `policy_1` / `kappa_hat` | `c_dcsn_hat` | 3rd | Raw primary control |
| `policy_2` / `x_cntn_hat` | `x_cntn_hat` | 4th | Continuation grid (jump detection) |
| `del_a` | `da_dcsn_hat` | 5th | Concavity / endogenous threshold |

Output arrays follow the same scheme with `_ref` suffix: `x_dcsn_ref`, `v_dcsn_ref`, `c_dcsn_ref`, `x_cntn_ref`.

## Operator and mover names

### From YAML equation blocks to code functions

The bellman-ddsl spec defines four equation blocks per stage. Each maps to a code-level operator:

| YAML block | Math | Code function name | Direction |
|-----------|------|--------------------|-----------|
| `arvl_to_dcsn_transition` | $\mathrm{g}_{\prec\circ}$ | `g_arvl_to_dcsn` | forward |
| `dcsn_to_cntn_transition` | $\mathrm{g}_{\circ\succ}$ | `g_dcsn_to_cntn` | forward |
| `cntn_to_dcsn_mover` | $\mathbb{B}$ | `mover_cntn_to_dcsn` | backward |
| `dcsn_to_arvl_mover` | $\mathbb{I}$ | `mover_dcsn_to_arvl` | backward |

### Sub-equation callables (EGM recipe)

Within `cntn_to_dcsn_mover`, the EGM method decomposes $\mathbb{B}$ into sub-equations. These are the `fn_*` callables passed to `kikku.asva.make_egm_1d` (scalar) or `kikku.asva.make_egm` (K-dimensional):

| Sub-equation | YAML | `fn_*` callable | Signature |
|-------------|------|-----------------|-----------|
| Inverse Euler | `InvEuler` | `fn_inv_euler` | `(dv_cntn_i, fixed_state, params) -> c_i` |
| Bellman RHS | `Bellman` | `fn_bellman_rhs` | `(c_i, v_cntn_i, fixed_state, params) -> v_i` |
| Reverse transition | `cntn_to_dcsn_transition` | `fn_cntn_to_dcsn` | `(c_i, x_cntn_i, fixed_state, params) -> x_dcsn_i` |
| Concavity | (diagnostic) | `fn_concavity` | `(c_i, ddv_cntn_i, fixed_state, params) -> da_i` |
| Marginal Bellman | `MarginalBellman` | `fn_marginal_bellman` | `(c_i, params) -> dv_i` |

Within `dcsn_to_arvl_mover`, the compose-and-interpolate step uses:

| Sub-equation | `fn_*` callable | Signature |
|-------------|-----------------|-----------|
| Arrival transition | `fn_arvl_transition` | `(x_cntn_grid, fixed_state, params) -> x_arvl_grid` |
| Constrained fallback | `fn_constrained` | `(x_arvl_pt, v_cntn_floor, fixed_state, params) -> (c, v, da)` |
| Marginal Bellman | `fn_marginal_bellman` | `(c_i, da_i, fixed_state, params) -> (dv, ddv)` |
| Interpolation | `interp_fn` | `(x_src, y_src, x_tgt) -> y_tgt` |

### Operator factory functions

Factories that *build* the operators at model-setup time follow the pattern `operator_factory_{mover}` (from Matsya / bellman-dev convention):

```python
# Build the backward mover (EGM step)
T_cntn_to_dcsn_work = operator_factory_cntn_to_dcsn(
    mover_spec, stage_params)

# Build the arrival mover (compose + interp)
T_dcsn_to_arvl_work = operator_factory_dcsn_to_arvl(
    mover_spec, stage_params)
```

The returned callable is the operator itself, named `T_{mover}_{stage}` for disambiguation.

### Composed stage operators

The full stage operator is $\mathbb{T} = \mathbb{I} \circ \mathbb{B}$. In code, the composed operator for a stage is named by its stage name:

| Stage | Code operator | Composition |
|-------|--------------|-------------|
| `work_cons` | `op_work_cons` | `mover_dcsn_to_arvl_work` $\circ$ `mover_cntn_to_dcsn_work` |
| `retire_cons` | `op_retire_cons` | `mover_dcsn_to_arvl_ret` $\circ$ `mover_cntn_to_dcsn_ret` |
| `labour_mkt_decision` | `op_labour_mkt` | `mover_dcsn_to_arvl_lmkt` (pure `max`, no $\mathbb{B}$) |

When the stage operator needs to be stage-qualified (e.g. in a dict), use the stage name as key:

```python
stage_ops = {
    'work_cons': op_work_cons,
    'retire_cons': op_retire_cons,
    'labour_mkt_decision': op_labour_mkt,
}
```

## Solution dict keys

The nest stores solutions per period per stage. Keys should use the perch-suffixed names:

```python
nest["solutions"][t] = {
    "retire_cons": {
        "c_arvl": ...,       # consumption on arrival grid
        "v_arvl": ...,       # value on arrival grid
        "da_arvl": ...,      # concavity on arrival grid
        "ddv_arvl": ...,     # second derivative on arrival grid
    },
    "work_cons": {
        "c_arvl": ...,       # consumption on arrival grid (after interp)
        "v_arvl": ...,       # value on arrival grid
        "da_arvl": ...,      # concavity on arrival grid
        # raw EGM (for diagnostics / plotting)
        "c_dcsn_hat": ...,   # raw consumption from EGM
        "v_dcsn_hat": ...,   # raw value correspondence
        "x_dcsn_hat": ...,   # raw endogenous grid
        "da_dcsn_hat": ...,  # raw concavity
    },
    "labour_mkt_decision": {
        "v_arvl": ...,       # max(v_work - delta, v_retire)
        "c_arvl": ...,       # consumption from chosen branch
        "dv_arvl": ...,      # marginal value
        "ddv_arvl": ...,     # second derivative
    },
    "ue_time": ...,
    "solve_time": ...,
}
```

## Grid naming

| Current | Proposed | Description |
|---------|----------|-------------|
| `asset_grid_A` | `x_arvl_grid` | Exogenous arrival grid (assets) |
| `n_w` / `w_min` / `w_max` | `n_dcsn` / `x_dcsn_min` / `x_dcsn_max` | Decision grid specification |
| `grid_size` | `n_arvl` | Number of arrival grid points |

For stage-specific grids, prefix with stage abbreviation if ambiguity arises:

```
x_arvl_grid          # shared arrival grid (assets)
x_dcsn_grid_work     # worker decision grid (cash-on-hand)
x_dcsn_grid_ret      # retiree decision grid (cash-on-hand)
```

## Branching stage conventions

For stages with `kind: branching`, the continuation values are branch-keyed:

| Current | Proposed | Description |
|---------|----------|-------------|
| `v_work` | `v_cntn_work` | Continuation value from work branch |
| `v_ret` | `v_cntn_ret` | Continuation value from retire branch |
| `V_cntn[>][work]` | `v_cntn_work` | Same, in YAML notation |

The branching operator receives all branch values and returns the envelope:

```python
v_arvl = max(v_cntn_work - delta, v_cntn_ret)
```

## Inter-period connectors

Connectors rename poststates to prestates across period boundaries:

```yaml
# nest.yaml
inter_connectors:
  - {b: a, b_ret: a_ret}
```

In code, this is a rename dict applied after each period solve:

```python
connector = {"b": "a", "b_ret": "a_ret"}
# maps x_cntn poststate names -> x_arvl prestate names
```

## Summary: the full pipeline for one stage

```
                    FORWARD                          BACKWARD
                    ───────                          ────────

  x_arvl ──g_arvl_to_dcsn──> x_dcsn ──g_dcsn_to_cntn──> x_cntn
                                                           │
                                                     v_cntn, dv_cntn
                                                           │
                              x_dcsn_hat <──fn_cntn_to_dcsn─┘
                              v_dcsn_hat <──fn_bellman_rhs───┘
                              c_dcsn_hat <──fn_inv_euler─────┘
                                   │
                                 [FUES]
                                   │
                              x_dcsn_ref, v_dcsn_ref, c_dcsn_ref
                                   │
                        ┌──mover_dcsn_to_arvl──┐
                        │  (compose + interp)   │
                        v                       v
                   v_arvl, c_arvl, dv_arvl, ddv_arvl
```
