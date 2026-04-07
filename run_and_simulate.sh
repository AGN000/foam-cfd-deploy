#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_and_simulate.sh
#
# Full Phase 2 pipeline:
#   1. Clear GPU   2. Start server   3. Generate mesh + run OpenFOAM   4. PNG
#
# Usage:
#   # Single prompt
#   ./run_and_simulate.sh -p "Circular pipe, radius 0.05m, length 1m"
#
#   # Multiple prompts (each gets its own result image)
#   ./run_and_simulate.sh \
#       -p "Circular pipe, radius 0.05m, length 1m" \
#       -p "Lid-driven cavity 0.1m x 0.1m x 0.1m" \
#       -p "NACA 0012 airfoil, chord 0.3m, AoA 5 degrees"
#
#   # Custom output directory
#   ./run_and_simulate.sh -p "..." --outdir ~/results/
#
#   # Longer simulation timeout (default 600 s per case)
#   ./run_and_simulate.sh -p "..." --timeout 1200
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/checkpoints/unified/merged"
SERVER_LOG="/tmp/meshgen_server.log"
OUTDIR="$SCRIPT_DIR/outputs"
SIM_TIMEOUT=600

# ── Parse arguments ───────────────────────────────────────────────────────────
PROMPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prompt) PROMPTS+=("$2"); shift 2 ;;
        --outdir)    OUTDIR="$2";     shift 2 ;;
        --timeout)   SIM_TIMEOUT="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,20p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ${#PROMPTS[@]} -eq 0 ]]; then
    PROMPTS=(
        "Circular pipe, radius 0.05m, length 1m"
        "Lid-driven cavity 0.1m x 0.1m x 0.1m"
        "NACA 0012 airfoil, chord 0.3m, AoA 5 degrees"
    )
fi

mkdir -p "$OUTDIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     CFD Mesh + Simulate  ·  Qwen2.5-Coder-7B + OF11        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Prompts to simulate (${#PROMPTS[@]}):"
for i in "${!PROMPTS[@]}"; do
    printf "  [%d] %s\n" "$((i+1))" "${PROMPTS[$i]}"
done
echo ""

# ── 1. Clear GPU memory ───────────────────────────────────────────────────────
echo "==> [1/4] Clearing GPU memory..."
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
if [[ -n "$GPU_PIDS" ]]; then
    for pid in $GPU_PIDS; do
        kill -9 "$pid" 2>/dev/null && echo "    Killed PID $pid" || true
    done
    sleep 3
    echo "    GPU cleared."
else
    echo "    GPU already free."
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
echo "    Free: ${FREE} MiB / ${TOTAL} MiB"

# ── 2. Start the inference server ─────────────────────────────────────────────
echo ""
echo "==> [2/4] Starting inference server..."
cd "$SCRIPT_DIR"
MESH_MODEL_PATH="$MODEL_PATH" nohup python3 -m uvicorn inference.server:app \
    --host 0.0.0.0 --port 8000 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "    PID: $SERVER_PID  |  log: $SERVER_LOG"

echo ""
echo "==> Waiting for model to load (typically 60-90 s)..."
TIMEOUT=240; ELAPSED=0
while true; do
    STATUS=$(curl -sf http://localhost:8000/health 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded'))" 2>/dev/null \
        || echo "False")
    [[ "$STATUS" == "True" ]] && { echo "    Ready in ${ELAPSED}s"; break; }
    sleep 5; ELAPSED=$((ELAPSED+5))
    printf "    %3ds  %s\r" "$ELAPSED" "$(tail -1 $SERVER_LOG 2>/dev/null | cut -c1-60)"
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "ERROR: timeout. Check $SERVER_LOG"; exit 1
    fi
done

# ── 3. Run simulations ─────────────────────────────────────────────────────────
echo ""
echo "==> [3/4] Running mesh generation + simulations..."

RESULT_IMAGES=()
RESULT_LABELS=()
RESULT_STATUS=()

for prompt in "${PROMPTS[@]}"; do
    printf "\n  ► %s\n" "$(echo "$prompt" | cut -c1-72)"

    escaped=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$prompt")
    slug=$(echo "$prompt" | python3 -c "import sys,hashlib; print(hashlib.md5(sys.stdin.read().encode()).hexdigest()[:8])")
    out_png="$OUTDIR/sim_${slug}.png"

    printf "    Submitting to /simulate ... "
    resp=$(curl -s -X POST http://localhost:8000/simulate \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": $escaped, \"output_png\": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$out_png"), \"max_retries\": 5, \"sim_timeout\": $SIM_TIMEOUT}" \
        --max-time $((SIM_TIMEOUT + 120)))

    ok=$(echo "$resp" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(d.get('ok','false'))" 2>/dev/null || echo "false")
    iters=$(echo "$resp" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(d.get('iterations','0'))" 2>/dev/null || echo "0")
    result_img=$(echo "$resp" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(d.get('result_image') or '')" 2>/dev/null || echo "")
    err=$(echo "$resp" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print((d.get('error') or d.get('detail','')  )[:80])" 2>/dev/null || echo "parse error")
    patches=$(echo "$resp" | python3 -c \
        "import sys,json; d=json.load(sys.stdin); print(','.join(d.get('patches',[])))" 2>/dev/null || echo "")

    if [[ "$ok" == "True" ]]; then
        echo "✓  ${iters} iterations  |  patches: ${patches}"
        echo "    Image: $result_img"
        RESULT_IMAGES+=("$result_img")
        RESULT_STATUS+=("ok")
    else
        echo "✗  $err"
        RESULT_IMAGES+=("")
        RESULT_STATUS+=("failed")
    fi
    RESULT_LABELS+=("$prompt")
done

# ── 4. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Simulation Summary                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
for i in "${!RESULT_LABELS[@]}"; do
    status="${RESULT_STATUS[$i]}"
    img="${RESULT_IMAGES[$i]}"
    label=$(echo "${RESULT_LABELS[$i]}" | cut -c1-50)
    if [[ "$status" == "ok" ]]; then
        printf "  ✓  %-52s  %s\n" "$label" "$(basename "$img")"
    else
        printf "  ✗  %-52s  failed\n" "$label"
    fi
done
echo ""
printf "  Output dir : %s\n" "$OUTDIR"
printf "  Server PID : %s  (kill to stop)\n" "$SERVER_PID"
echo ""
