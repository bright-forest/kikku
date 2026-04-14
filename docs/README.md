# kikku documentation

## Runner infrastructure (`kikku.run`)

| Document | What it covers |
|----------|---------------|
| [Package overview](run.md) | `RunSpec`, parameter tiers, modes, override system |
| [CLI reference](cli.md) | `parse_run` signature, all `RunSpec` fields, `make_run_dir` |
| [Parameter sweeps](sweep.md) | `param_grid`, `sweep`, timing tables, MPI |
| [Persistence](persistence.md) | `save_solution` (numpy) vs `save_nest` (pickle) |

## Estimation (`kikku.run.estimate` + `kikku.run.moments`)

| Document | What it covers |
|----------|---------------|
| [Estimation guide](estimation_guide.md) | Trial function, moment YAML, `make_criterion`, `estimate` |
| [Cross-entropy method](cross_entropy_method.md) | CE algorithm internals, MPI topology, tuning |
| [Gadi HPC guide](smm_gadi_guide.md) | PBS scripts, queue selection, checkpointing on NCI |

## Conventions

| Document | What it covers |
|----------|---------------|
| [Naming conventions](naming-conventions.md) | Perch suffixes, operator names, solution dict keys |
