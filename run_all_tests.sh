#!/bin/bash
# =============================================================================
# LabelLens — Complete Test Suite Runner
# =============================================================================
# Runs tests for ALL phases using each phase's isolated virtual environment.
#
# Phase layout:
#   Consolidated root suite covering OCR through frontend adapters.
# =============================================================================

set -e

export PYTHONPATH=$(pwd)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_ERROR=0
TOTAL_SKIP=0

run_phase_tests() {
    local phase_name="$1"
    local pytest_bin="$2"
    local test_dir="$3"

    echo ""
    echo "=========================================="
    echo " Running ${phase_name} tests..."
    echo "=========================================="

    if [ ! -f "$pytest_bin" ]; then
        echo "  SKIPPED: pytest not found at ${pytest_bin}"
        return
    fi
    if [ ! -d "$test_dir" ]; then
        echo "  SKIPPED: test directory not found at ${test_dir}"
        return
    fi

    $pytest_bin "$test_dir" -v 2>&1 || true
}

echo "============================================"
echo " LabelLens — Full Regression Test Suite"
echo "============================================"

if [ -x ".venv/bin/pytest" ]; then
    .venv/bin/pytest -q tests -m "not model and not private_data"
else
    python3 -m pytest -q tests -m "not model and not private_data"
fi


echo ""
echo "============================================"
echo " All phase tests complete."
echo "============================================"
