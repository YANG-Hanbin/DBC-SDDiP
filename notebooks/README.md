# Notebook Quick Runs

This folder provides one runnable notebook per experiment project:

1. `scuc_quick_run.ipynb`
2. `sslp_quick_run.ipynb`
3. `generation_expansion_quick_run.ipynb`

Each notebook is a minimal `DBC/iDBC` runner:

1. Locate repo root and activate project.
2. Edit `cut_mode` (`:DBC` or `:iDBC`) and core parameters.
3. Generate a temporary config under `notebooks/tmp_*_notebook_config.jl`.
4. Run a dry-run command.
5. Run the real experiment command.

Notes:

- Keep `cut_mode = :iDBC` together with `enable_copy_branching = true` to activate copy-variable split behavior.
- In SSLP, prefer `mdc_iter >= 10` for iDBC because `(iDBC, 5)` is currently skipped by run-grid filtering.

## Paper Suite Notebooks

Use these when you want to run the paper presets directly from a notebook:

1. `paper_suite_sslp.ipynb`
2. `paper_suite_generation_expansion.ipynb`
3. `paper_suite_scuc.ipynb`

Each paper-suite notebook includes:

1. Config listing for `experiments/*/configs/paper_suite/`
2. One-config `dry-run`
3. One-config real run
4. Optional loop to run all paper-suite configs
