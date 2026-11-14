#!/bin/sh
# Re-run every experiment with the reported settings (see the README for the expected runtime).
# Usage:  sh experiments/run_all.sh            # full run (numbers reported in the README); resumable
#         sh experiments/run_all.sh --quick    # smoke test only; writes to _quick_output/, never to results/
#         FORCE=1 sh experiments/run_all.sh    # ignore progress markers and re-run everything
# Progress markers live in _run_state/ (git-ignored); an interrupted run resumes at the first unfinished experiment.
set -e
cd "$(dirname "$0")/.."
mkdir -p _run_state
for exp in learning_rate choice_consistency individual_differences parameter_recovery model_recovery model_comparison interactive_models empirical_bandit identifiability model_recovery_ladder; do
    if [ -z "$FORCE" ] && [ "$1" != "--quick" ] && [ -f "_run_state/$exp.done" ]; then
        echo "=== $exp (already done, skipping) ==="
        continue
    fi
    echo "=== $exp ==="
    python "experiments/$exp/run.py" "$@"
    [ "$1" = "--quick" ] || touch "_run_state/$exp.done"
done
