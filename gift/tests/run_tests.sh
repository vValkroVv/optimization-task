#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(mktemp -d)}"
SUITES="${SUITES:-features models}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    B=$'\e[1m'; DIM=$'\e[2m'; RED=$'\e[31m'; GREEN=$'\e[32m'; CYAN=$'\e[36m'; R=$'\e[0m'
else
    B=; DIM=; RED=; GREEN=; CYAN=; R=
fi

step() { printf '\n%s==>%s %s%s\n' "$CYAN$B" "$R$B" "$*" "$R"; }
die()  { printf '%s✗ %s%s\n' "$RED$B" "$*" "$R" >&2; exit 1; }

usage() {
    cat <<EOF
${B}Usage:${R} ${0##*/} [unit|integration|all]   ${DIM}(default: all)${R}

Run torchtitan unit and/or integration tests.

${B}Environment overrides:${R}
  ${B}NGPU${R}        GPUs for integration tests   ${DIM}(default: detected = ${NGPU})${R}
  ${B}SUITES${R}      integration suites to run    ${DIM}(default: "${SUITES}")${R}
  ${B}OUTPUT_DIR${R}  integration artifacts dir    ${DIM}(default: a temp dir)${R}
EOF
}

run_unit() {
    step "Unit tests ${DIM}pytest tests/unit_tests${R}"
    python -m pytest tests/unit_tests
}

run_integration() {
    step "Integration tests ${DIM}${NGPU} GPU(s) -> ${OUTPUT_DIR}${R}"
    # features.py reloads an HF checkpoint from $RUNNER_TEMP/artifacts-to-be-uploaded,
    # so mirror the CI layout; other suites have no such path coupling.
    export RUNNER_TEMP="$OUTPUT_DIR"
    for suite in $SUITES; do
        step "suite: ${B}${suite}${R}"
        local out="${OUTPUT_DIR}/${suite}"
        [[ "$suite" == features ]] && out="${OUTPUT_DIR}/artifacts-to-be-uploaded"
        rm -rf "${out:?}"
        python -m tests.integration_tests.run_tests \
            --test_suite "$suite" --ngpu "$NGPU" "$out"
    done
}

mode="${1:-all}"
start=$(date +%s)
case "$mode" in
    unit)        run_unit ;;
    integration) run_integration ;;
    all)         run_unit; run_integration ;;
    -h|--help)   usage; exit 0 ;;
    *)           usage >&2; die "unknown mode: ${mode}" ;;
esac

printf '\n%s✓ all green%s %s(%s, %ds)%s\n' "$GREEN$B" "$R" "$DIM" "$mode" "$(( $(date +%s) - start ))" "$R"
