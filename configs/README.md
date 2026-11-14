# Dataset specifications

A *spec* is a small JSON file that maps a raw data file onto HumanChoice's schema, records provenance and fixes the
exclusion rules **before** any model is fitted (see `../docs/analysis_plan.md`). Real data files are not bundled.

Workflow:

```bash
# 1. download the data yourself from the source named in the spec's provenance; check licence and terms
# 2. copy the template, replace every VERIFY / MAP_ME with verified values, set "path" (and "sha256" once you have the file)
cp configs/bahrami_navajas_2020.template.json configs/my_dataset.json
# 3. validate - fails loudly on schema problems and prints a data-quality summary
humanchoice validate-data --spec configs/my_dataset.json
# 4. run the full pre-specified analysis (about 1 CPU-minute per 10 participants; use --jobs for parallelism)
humanchoice fit --spec configs/my_dataset.json --out results/my_dataset --jobs 4
```

`humanchoice fit` refuses to run while a provenance field still contains `VERIFY`.

**Paths.** `path` in a spec is resolved **relative to the JSON file itself**, not to the working directory: a spec in `configs/` pointing at `raw/data.csv` at the project root needs `"path": "../raw/data.csv"`.

**Provenance.** For a non-synthetic dataset `source_url`, `citation` and `licence` are mandatory. A missing or empty field blocks analysis exactly like a `VERIFY` placeholder; if a field truly does not exist
write `UNAVAILABLE: <documented reason>`.
