"""Command-line interface::

    humanchoice validate-data --spec configs/my_dataset.json
    humanchoice fit           --spec configs/my_dataset.json --out results/my_dataset --jobs 4
    humanchoice sensitivity   --spec configs/my_dataset.json --out results/my_dataset_sensitivity --jobs 4
    humanchoice reproduce empirical-bandit --quick        # synthetic rehearsal, no data needed
    humanchoice list-models

``fit`` refuses to run while any provenance field in the spec still says ``VERIFY`` (unless ``--allow-unverified`` is
given): licence, source and citation must be confirmed before a dataset is analysed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

import humanchoice
from humanchoice.data import DatasetSpec, DataValidationError, load_dataset
from humanchoice.envs import RestlessBandit
from humanchoice.pipelines import (
    default_ladder,
    run_empirical_analysis,
    run_empirical_by_condition,
    run_sensitivity_analyses,
    simulate_restless_rehearsal,
)

REHEARSAL_BANNER = "SYNTHETIC REHEARSAL - simulated participants with known generating models. NOT human data."


def _load(spec_path: str):
    return load_dataset(DatasetSpec.from_json(spec_path))


def _load_env_factory(spec: str | None):
    """Resolve ``package.module:callable`` to an environment factory (a zero-argument callable returning an environment)."""
    if not spec:
        return None
    module, _, name = spec.partition(":")
    if not name:
        raise DataValidationError("--env must look like 'package.module:callable'")
    try:
        factory = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise DataValidationError(f"cannot import environment factory '{spec}': {exc}") from exc
    return factory


def cmd_validate(args: argparse.Namespace) -> int:
    loaded = _load(args.spec)
    q = loaded.dataset.quality_report()
    print(f"dataset: {loaded.spec.name}   sha256: {loaded.sha256}")
    print(f"participants kept: {q['n_participants']}   excluded by pre-specified rules: {len(loaded.excluded)}")
    print(f"trials: {q['n_trials_total']} ({q['trials_per_participant_min']}-{q['trials_per_participant_max']} per participant)")
    print(f"missed trials: {q['n_missed_total']} ({q['missed_fraction_overall']:.1%})")
    print(f"rewards: mean {q['reward_mean']:.3f}, SD {q['reward_sd']:.3f}, range [{q['reward_min']:.3f}, {q['reward_max']:.3f}]")
    print("choice shares:", {k: round(v, 3) for k, v in q["choice_proportions"].items()})
    print("preprocessing:\n" + str(loaded.dataset.report))
    if loaded.spec.unverified_fields:
        print(f"\nWARNING: provenance fields still marked VERIFY: {loaded.spec.unverified_fields}")
    if args.json:
        Path(args.json).write_text(json.dumps(q, indent=2))
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    spec = DatasetSpec.from_json(args.spec)
    if spec.unverified_fields and not args.allow_unverified:
        print(
            f"Refusing to analyse '{spec.name}': provenance fields {spec.unverified_fields} are still marked VERIFY.\n"
            "Confirm the licence, source and citation against the data provider, edit the spec, or pass --allow-unverified "
            "for a private exploratory run.",
            file=sys.stderr,
        )
        return 3
    loaded = load_dataset(spec)
    banner = "" if not spec.synthetic else REHEARSAL_BANNER
    if spec.unverified_fields:
        banner = (banner + " " if banner else "") + "PROVENANCE NOT VERIFIED - exploratory run only."
    gate = pd.read_csv(args.recovery_gate) if getattr(args, "recovery_gate", None) else None
    kw = {
        "win_threshold": args.win_threshold, "n_starts": args.starts, "n_jobs": args.jobs, "max_participants": args.max_participants,
        "seed": args.seed, "banner": banner, "excluded": loaded.excluded, "spec_json": spec.to_json(),
        "env_factory": _load_env_factory(getattr(args, "env", None)), "recovery_gate": gate,
        "analysis_plan_path": getattr(args, "analysis_plan", None),
    }
    df = loaded.dataset.df
    by_schedule = "condition" in df.columns and df["condition"].nunique() >= 2 and not getattr(args, "pooled_only", False)
    if by_schedule:  # the plan requires pooled AND per-schedule results
        run_empirical_by_condition(loaded.dataset, args.out, min_participants=getattr(args, "min_schedule_size", 30), **kw)
    else:
        run_empirical_analysis(loaded.dataset, args.out, title=f"Empirical analysis: {spec.name}", **kw)
    print(f"report written to {Path(args.out) / 'report.md'}")
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    spec = DatasetSpec.from_json(args.spec)
    if spec.unverified_fields and not args.allow_unverified:
        print(f"Refusing to analyse '{spec.name}': provenance fields {spec.unverified_fields} are still marked VERIFY.", file=sys.stderr)
        return 3
    loaded = load_dataset(spec)
    out = run_sensitivity_analyses(loaded.dataset, args.out, n_starts=args.starts, n_jobs=args.jobs, seed=args.seed, win_threshold=args.win_threshold)
    print(f"{int(out['decision_changed'].sum())} variant verdicts differ from the baseline; see {Path(args.out) / 'sensitivity_summary.md'}")
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    if args.experiment != "empirical-bandit":
        print("only 'empirical-bandit' is available here; run `sh experiments/run_all.sh` for the other experiments", file=sys.stderr)
        return 2
    if args.spec:
        return cmd_fit(argparse.Namespace(spec=args.spec, out=args.out, jobs=args.jobs, max_participants=args.max_participants,
                                          starts=args.starts, seed=args.seed, win_threshold=args.win_threshold,
                                          allow_unverified=args.allow_unverified, env=args.env))
    n, T = (8, 60) if args.quick else (60, 150)
    ds = simulate_restless_rehearsal(n_participants=n, n_trials=T, seed=args.seed + 2026)
    kw: dict = {"n_starts": args.starts}
    if args.quick:
        kw |= {"n_starts": 2, "fractions": (0.5, 0.7), "ppc_participants": 8, "ppc_sims": 60, "profile_participants": 2}
    run_empirical_analysis(ds, args.out, n_jobs=args.jobs, seed=args.seed, title="Empirical-analysis pipeline: synthetic rehearsal",
                           banner=REHEARSAL_BANNER, env_factory=lambda: RestlessBandit(4), **kw)
    print(f"report written to {Path(args.out) / 'report.md'}")
    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    for binary in (False, True):
        K = 2 if binary else 4
        print(f"\nladder for K={K} options, {'binary' if binary else 'continuous'} rewards:")
        for key, m in default_ladder(K, binary).items():
            print(f"  {key:9s} {type(m).__name__:22s} params: {', '.join(m.param_names) or '(none)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="humanchoice", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"humanchoice {humanchoice.__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-data", help="validate a dataset spec and print a data-quality summary")
    v.add_argument("--spec", required=True)
    v.add_argument("--json", help="also write the data-quality summary to this JSON file")
    v.set_defaults(func=cmd_validate)

    def common(p):
        p.add_argument("--out", required=True, help="output directory")
        p.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
        p.add_argument("--max-participants", type=int, default=None, help="analyse a seeded random subset of N participants (independent of file order)")
        p.add_argument("--starts", type=int, default=4, help="optimiser starts per fit")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--win-threshold", type=float, default=0.5, help="reward counted as a 'win' by WSLS / stay statistics")
        p.add_argument("--allow-unverified", action="store_true", help="run even if provenance fields are marked VERIFY")
        p.add_argument("--recovery-gate", default=None, help="CSV (columns model,param,status) from experiments/identifiability: which parameters may be interpreted")
        p.add_argument("--analysis-plan", default=None, help="analysis-plan file to hash and copy (default: the plan bundled with the package)")
        p.add_argument("--pooled-only", action="store_true", help="do not additionally analyse each reward-drift schedule ('condition' column)")
        p.add_argument("--min-schedule-size", type=int, default=30, help="schedules with fewer participants are listed as skipped")
        p.add_argument("--env", default=None, help="'package.module:callable' returning an environment that reproduces the task's reward "
                       "process; enables the full-session outcome checks (otherwise they are reported as not run)")

    f = sub.add_parser("fit", help="run the full pre-specified analysis on a dataset spec")
    f.add_argument("--spec", required=True)
    common(f)
    f.set_defaults(func=cmd_fit)

    se = sub.add_parser("sensitivity", help="re-run the primary contrasts under every declared sensitivity analysis")
    se.add_argument("--spec", required=True)
    common(se)
    se.set_defaults(func=cmd_sensitivity)

    r = sub.add_parser("reproduce", help="reproduce an analysis (synthetic rehearsal unless --spec is given)")
    r.add_argument("experiment", choices=["empirical-bandit"])
    r.add_argument("--spec")
    r.add_argument("--quick", action="store_true", help="tiny sizes for a smoke test")
    common(r)
    r.set_defaults(func=cmd_reproduce)

    sub.add_parser("list-models", help="show the model ladders").set_defaults(func=cmd_list_models)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DataValidationError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
