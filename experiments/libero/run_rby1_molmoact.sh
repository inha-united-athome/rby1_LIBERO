#!/bin/bash
# =============================================================================
# MolmoAct + RBY1 LIBERO Inference Script
# =============================================================================
# Usage:
#   ./run_rby1_molmoact.sh spatial 0          # Run task 0 of libero_spatial
#   ./run_rby1_molmoact.sh object 3           # Run task 3 of libero_object
#   ./run_rby1_molmoact.sh goal 5             # Run task 5 of libero_goal
#   ./run_rby1_molmoact.sh 10 2               # Run task 2 of libero_10
#   ./run_rby1_molmoact.sh spatial             # Run ALL tasks (0-9) of libero_spatial
# =============================================================================

set -e

# Configuration
TASK_TYPE="${1:-spatial}"        # spatial | object | goal | 10
TASK_ID="${2:-}"                  # 0-9 or empty for all
ROBOT_NAME="${ROBOT_NAME:-RBY1RightArm}"

# Checkpoint mapping
case "$TASK_TYPE" in
    spatial)
        CHECKPOINT="${CHECKPOINT:-allenai/MolmoAct-7B-D-LIBERO-Spatial-0812}"
        ;;
    object)
        CHECKPOINT="${CHECKPOINT:-allenai/MolmoAct-7B-D-LIBERO-Object-0812}"
        ;;
    goal)
        CHECKPOINT="${CHECKPOINT:-allenai/MolmoAct-7B-D-LIBERO-Goal-0812}"
        ;;
    10)
        CHECKPOINT="${CHECKPOINT:-allenai/MolmoAct-7B-D-LIBERO-10-0812}"
        ;;
    *)
        echo "Unknown task type: $TASK_TYPE"
        echo "Supported: spatial, object, goal, 10"
        exit 1
        ;;
esac

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Environment setup
export DISPLAY="${DISPLAY:-:1}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TF_CPP_MIN_LOG_LEVEL=3

echo "============================================="
echo "  MolmoAct + RBY1 LIBERO Inference"
echo "============================================="
echo "Task type:   libero_${TASK_TYPE}"
echo "Task ID:     ${TASK_ID:-all (0-9)}"
echo "Robot:       ${ROBOT_NAME}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "DISPLAY:     ${DISPLAY}"
echo "============================================="

# Auto-patch MolmoAct model code for transformers compatibility
# The HuggingFace-cached modeling_molmoact.py has two bugs:
#   1) attn_implementation can be None (should default to "eager")
#   2) self.config.float32_attention should be self.float32_attention (in ViT attention)
echo "[patch] Checking MolmoAct model code for known issues..."
MOLMOACT_CACHE="$HOME/.cache/huggingface/modules/transformers_modules/allenai"
for MODEL_DIR in "$MOLMOACT_CACHE"/MolmoAct-*/; do
    if [ -d "$MODEL_DIR" ]; then
        for MFILE in "$MODEL_DIR"*/modeling_molmoact.py; do
            if [ -f "$MFILE" ]; then
                # Fix 1: attn_implementation default to "eager" when None
                if grep -q 'self\.attn_implementation = attn_implementation$' "$MFILE" 2>/dev/null; then
                    sed -i 's/self\.attn_implementation = attn_implementation$/self.attn_implementation = attn_implementation or "eager"/' "$MFILE"
                    echo "[patch] Fixed attn_implementation None default in $(basename "$MODEL_DIR")"
                fi
                # Fix 2: self.config.float32_attention -> self.float32_attention
                if grep -q 'self\.config\.float32_attention' "$MFILE" 2>/dev/null; then
                    sed -i 's/self\.config\.float32_attention/self.float32_attention/g' "$MFILE"
                    echo "[patch] Fixed float32_attention attribute in $(basename "$MODEL_DIR")"
                fi
            fi
        done
    fi
done
echo "[patch] Done."
echo ""

# Build command
CMD="python3 ${SCRIPT_DIR}/rby1_libero_eval.py --task ${TASK_TYPE} --checkpoint ${CHECKPOINT}"
if [ -n "$TASK_ID" ]; then
    CMD="${CMD} --task_id ${TASK_ID}"
fi

echo "Running: ${CMD}"
echo ""
exec ${CMD}
