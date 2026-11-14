# Data

## `synthetic/`
Simulated behaviour with **known generating parameters**, produced by `examples/00_make_synthetic_dataset.py` (`humanchoice.simulation.save_dataset`):

| file | content |
|---|---|
| `<name>.csv` | trial-level behaviour: `participant_id, trial, choice, reward` (a missed trial has choice `<missed>`) |
| `<name>_truth.csv` | the generating parameters of every simulated participant |
| `<name>_config.json` | model, seed, trials, options, environment, package version, and `"synthetic": true` |

Synthetic data must never be presented as empirical human data.

## `processed/`
Reserved for validated versions of **real** datasets. Nothing is stored here: real data files are not bundled and are not redistributed.

## Adding a public dataset
1. Download it yourself from the source and **check the licence / terms**; record provenance (URL, DOI, version, access date, citation) in a spec ([`../configs/`](../configs)).
2. Map the raw columns to `participant_id, trial, choice, reward` (optionally `block, condition, reaction_time, available_actions`) in the spec; set `choice_categories`, the task's *stated* `reward_range` and `rescale_rewards`.
3. Fix the exclusion rules in the spec **before** fitting any model (they are applied by the loader and every excluded participant is listed with its reason).
4. `humanchoice validate-data --spec ...` fails loudly on schema problems and prints a data-quality summary; the file's SHA-256 is recorded with all results.
5. `humanchoice fit --spec ... --out results/<name>` runs the pre-specified analysis; it refuses to run while any provenance field still says `VERIFY`.

Choices may be any labels (`A..D`, `1..4`); timeouts are empty/NaN choices (use `missing: "mark"`, the default of a spec); rewards may be continuous.
