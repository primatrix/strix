#!/usr/bin/env bash
# run_benchmark.sh — Orchestrate a TPU benchmark Job on GKE.
#
# Usage:
#   bash scripts/run_benchmark.sh <kernel_module> \
#     --shape <shape> [--chunk-size <n>] [--tpu-type <type>] [--tpu-topology <topo>]
#
# The script renders benchmark_job.yaml via envsubst, deploys the Job,
# waits for completion, downloads results from GCS, and cleans up.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_TEMPLATE="${SCRIPT_DIR}/benchmark_job.yaml"

# ---- Defaults ----
TPU_TYPE="v7x"
TPU_TOPOLOGY="2x2x1"
CHUNK_SIZE=""

# ---- Usage ----
usage() {
  cat <<EOF
Usage: $(basename "$0") <kernel_module> --shape <shape> [options]

Arguments:
  kernel_module        Kernel module path (e.g. kernels.chunk_kda_fwd)

Options:
  --shape <dims>       Comma-separated shape (required)
  --chunk-size <n>     Chunk size for the kernel
  --tpu-type <type>    TPU type (default: v7x)
  --tpu-topology <t>   TPU topology (default: 2x2x1)
  -h, --help           Show this help
EOF
  exit 1
}

# ---- Parse arguments ----
if [[ $# -lt 1 ]]; then
  usage
fi

KERNEL_MODULE="${1:?kernel module is required}"
shift

SHAPE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shape)
      SHAPE="${2:?--shape requires a value}"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="${2:?--chunk-size requires a value}"
      shift 2
      ;;
    --tpu-type)
      TPU_TYPE="${2:?--tpu-type requires a value}"
      shift 2
      ;;
    --tpu-topology)
      TPU_TOPOLOGY="${2:?--tpu-topology requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage
      ;;
  esac
done

if [[ -z "${SHAPE}" ]]; then
  echo "Error: --shape is required" >&2
  usage
fi

# ---- Generate job name ----
KERNEL_SLUG="${KERNEL_MODULE//[._]/-}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
export JOB_NAME="strix-benchmark-${KERNEL_SLUG}-${TIMESTAMP}"

# ---- Resolve branch ----
export BRANCH
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
# Sanitise branch for K8s label (max 63 chars, alphanumeric + dash/dot/underscore)
export BRANCH_LABEL
BRANCH_LABEL="$(echo "${BRANCH}" | tr '/' '-' | cut -c1-63)"

# ---- Compute TPU chip count from topology (e.g. 2x2x1 = 4) ----
export TPU_CHIPS
TPU_CHIPS=$(( ${TPU_TOPOLOGY//x/*} ))

# ---- Derive GKE accelerator label (v7x -> tpu7x) ----
export TPU_ACCELERATOR="tpu${TPU_TYPE#v}"

# ---- Export for envsubst ----
export KERNEL_MODULE SHAPE CHUNK_SIZE TPU_TYPE TPU_TOPOLOGY

# ---- GCS config ----
export GCS_BUCKET="${GCS_BUCKET:-gs://poc_profile/}"
GCS_PATH="${GCS_BUCKET}${JOB_NAME}/"
OUTPUT_DIR="benchmark_results"

# ---- Cleanup trap ----
cleanup() {
  echo "[cleanup] Deleting K8s Job ${JOB_NAME}..."
  kubectl delete job "${JOB_NAME}" --ignore-not-found=true 2>/dev/null || true
}
trap cleanup EXIT

# ---- Render and deploy ----
echo "[run_benchmark] Rendering Job YAML for ${JOB_NAME}..."
ENVSUBST_VARS='$JOB_NAME $BRANCH $BRANCH_LABEL $KERNEL_MODULE $SHAPE $CHUNK_SIZE $TPU_TYPE $TPU_TOPOLOGY $TPU_CHIPS $TPU_ACCELERATOR $GCS_BUCKET'
RENDERED_YAML="$(envsubst "${ENVSUBST_VARS}" < "${YAML_TEMPLATE}")"

echo "[run_benchmark] Deploying Job..."
echo "${RENDERED_YAML}" | kubectl apply -f -

# ---- Wait for completion (fail fast if Job fails) ----
echo "[run_benchmark] Waiting for Job to complete..."
kubectl wait --for=condition=complete "job/${JOB_NAME}" --timeout=1800s &
WAIT_COMPLETE=$!
kubectl wait --for=condition=failed "job/${JOB_NAME}" --timeout=1800s &
WAIT_FAILED=$!

# Whichever finishes first wins
if wait -n "$WAIT_COMPLETE" "$WAIT_FAILED" 2>/dev/null; then
  # One condition was met — check which one
  if ! kubectl get job "${JOB_NAME}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep -q True; then
    echo "Error: Job ${JOB_NAME} failed" >&2
    kubectl logs "job/${JOB_NAME}" --tail=50 2>/dev/null || true
    exit 1
  fi
else
  echo "Error: timed out waiting for Job ${JOB_NAME}" >&2
  exit 1
fi
# Kill the remaining background wait
kill "$WAIT_COMPLETE" "$WAIT_FAILED" 2>/dev/null || true

# ---- Download results ----
echo "[run_benchmark] Downloading results from ${GCS_PATH}..."
mkdir -p "${OUTPUT_DIR}"
gcloud storage cp "${GCS_PATH}${JOB_NAME}.tar.gz" "${OUTPUT_DIR}/"

# ---- Extract ----
echo "[run_benchmark] Extracting results..."
tar xzf "${OUTPUT_DIR}/${JOB_NAME}.tar.gz" -C "${OUTPUT_DIR}"

echo "[run_benchmark] Done. Results in ${OUTPUT_DIR}/"
