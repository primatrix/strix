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

# ---- Compute TPU chip count from topology (e.g. 2x2x1 = 4) ----
export TPU_CHIPS
TPU_CHIPS="$(echo "${TPU_TOPOLOGY}" | tr 'x' '*' | bc)"

# ---- Export for envsubst ----
export KERNEL_MODULE SHAPE CHUNK_SIZE TPU_TYPE TPU_TOPOLOGY

# ---- GCS config ----
GCS_BUCKET="gs://poc_profile/"
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
RENDERED_YAML="$(envsubst < "${YAML_TEMPLATE}")"

echo "[run_benchmark] Deploying Job..."
echo "${RENDERED_YAML}" | kubectl apply -f -

# ---- Wait for completion ----
echo "[run_benchmark] Waiting for Job to complete..."
kubectl wait --for=condition=complete "job/${JOB_NAME}" --timeout=1800s

# ---- Download results ----
echo "[run_benchmark] Downloading results from ${GCS_PATH}..."
mkdir -p "${OUTPUT_DIR}"
gcloud storage cp "${GCS_PATH}${JOB_NAME}.tar.gz" "${OUTPUT_DIR}/"

# ---- Extract ----
echo "[run_benchmark] Extracting results..."
tar xzf "${OUTPUT_DIR}/${JOB_NAME}.tar.gz" -C "${OUTPUT_DIR}"

echo "[run_benchmark] Done. Results in ${OUTPUT_DIR}/"
