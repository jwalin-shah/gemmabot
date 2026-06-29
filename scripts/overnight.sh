#! /usr/bin/env bash
# 8-hour overnight runner. Crash-proof. Restarts Claude Code when it exits.
# Uses CHECKPOINT.json so nothing is ever lost.
#
# Run this and walk away:
#   nohup bash scripts/overnight.sh &
#
# Or run with tmux and detach:
#   tmux new-session -s overnight
#   bash scripts/overnight.sh
#   Ctrl+B then D to detach
#   tmux attach -t overnight to check
#
# To check progress remotely:
#   tail -f overnight_results/stdout.log

set -e

cd "$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p overnight_results

# Init checkpoint if not exists
if [ ! -f overnight_results/CHECKPOINT.json ]; then
    echo '{"completed": {}, "uncapped": true, "total_calls_used": 0, "round": 0}' > overnight_results/CHECKPOINT.json
fi

# 8 hours = 480 minutes. Each loop iteration ~30-45 min.
# 16 iterations covers 8 hours with buffer.
echo "=========================================="
echo "  OVERNIGHT LOOP"
echo "  Started: $(date)"
echo "  Max iterations: 16 (covers ~8 hours)"
echo "  Checkpoint: overnight_results/CHECKPOINT.json"
echo "  Log: overnight_results/stdout.log"
echo "=========================================="

for i in $(seq 1 16); do
    echo ""
    echo "--- Iteration $i of 16 at $(date) ---"
    echo ""

    claude code --print \
        /loop "Run the self-improving research loop from OVERNIGHT_PLAN.md. No budget cap. Each iteration: spawn 1-2 experiment subagents + 1 adversarial review subagent IN PARALLEL. After they return, synthesize results, generate new experiments if review finds gaps, update overnight_results/CHECKPOINT.json, append to overnight_results/results_all.csv. Wake up in 30 minutes if experiments are done, or immediately if there's more to run." \
        2>> overnight_results/loop_errors.log \
        | tee -a overnight_results/stdout.log

    EXIT_CODE=$?
    echo "  Claude Code exited with code $EXIT_CODE at $(date)"

    # Small delay before restarting
    sleep 15
done

echo ""
echo "=========================================="
echo "  OVERNIGHT COMPLETE at $(date)"
echo "  Check overnight_results/ for results."
echo "=========================================="
