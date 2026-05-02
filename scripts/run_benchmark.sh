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
JOB_TIMEOUT=${JOB_TIMEOUT:-7200}

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

# ---- Preflight checks ----
echo "[preflight] Checking kubectl connectivity..."
if ! kubectl cluster-info > /dev/null 2>&1; then
  echo "Error: cannot connect to Kubernetes cluster" >&2
  exit 1
fi

# Resolve the GCP service account bound to the K8s SA via Workload Identity
K8S_SA="gcs-account"
GCP_SA="$(kubectl get sa "${K8S_SA}" -o jsonpath='{.metadata.annotations.iam\.gke\.io/gcp-service-account}' 2>/dev/null || true)"
if [[ -n "${GCP_SA}" ]]; then
  echo "[preflight] Checking GCS write access for ${GCP_SA} on ${GCS_BUCKET}..."
  BUCKET_POLICY="$(gcloud storage buckets get-iam-policy "${GCS_BUCKET}" --format=json 2>/dev/null || true)"
  if [[ -n "${BUCKET_POLICY}" ]]; then
    if ! echo "${BUCKET_POLICY}" | grep -q "${GCP_SA}"; then
      echo "Error: ${GCP_SA} has no IAM binding on ${GCS_BUCKET}." >&2
      echo "Fix: gcloud storage buckets add-iam-policy-binding ${GCS_BUCKET} \\" >&2
      echo "  --member='serviceAccount:${GCP_SA}' --role='roles/storage.objectAdmin'" >&2
      exit 1
    fi
  fi
fi
echo "[preflight] OK"

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

# ---- Wait for completion (polling loop, bash 3.2 compatible) ----
echo "[run_benchmark] Waiting for Job to complete (timeout: ${JOB_TIMEOUT}s)..."
ELAPSED=0
INTERVAL=30
while [[ $ELAPSED -lt $JOB_TIMEOUT ]]; do
  STATUS=$(kubectl get job "${JOB_NAME}" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || true)
  if echo "$STATUS" | grep -q True; then
    break
  fi
  STATUS=$(kubectl get job "${JOB_NAME}" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || true)
  if echo "$STATUS" | grep -q True; then
    echo "Error: Job ${JOB_NAME} failed" >&2
    kubectl logs "job/${JOB_NAME}" --tail=50 2>/dev/null || true
    exit 1
  fi
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done
if [[ $ELAPSED -ge $JOB_TIMEOUT ]]; then
  echo "Error: timed out waiting for Job ${JOB_NAME}" >&2
  exit 1
fi

# ---- Download results ----
echo "[run_benchmark] Downloading results from ${GCS_PATH}..."
mkdir -p "${OUTPUT_DIR}"
gcloud storage cp "${GCS_PATH}${JOB_NAME}.tar.gz" "${OUTPUT_DIR}/"

# ---- Extract ----
echo "[run_benchmark] Extracting results..."
tar xzf "${OUTPUT_DIR}/${JOB_NAME}.tar.gz" -C "${OUTPUT_DIR}"

echo "[run_benchmark] Done. Results in ${OUTPUT_DIR}/"
