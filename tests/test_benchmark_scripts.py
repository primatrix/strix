"""Tests for K8s Job YAML template and run_benchmark.sh (Issue #210).

Acceptance criteria:
1. benchmark_job.yaml uses gcs-account SA, correct TPU resource limits and topology
2. run_benchmark.sh accepts --kernel, --shape, --chunk-size parameters
3. envsubst-rendered YAML passes structure validation
4. Script includes Job resource cleanup logic
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Repo root is one level above tests/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_YAML_TEMPLATE = _SCRIPTS_DIR / "benchmark_job.yaml"
_SHELL_SCRIPT = _SCRIPTS_DIR / "run_benchmark.sh"


# ---------------------------------------------------------------------------
# Helper: render YAML template via envsubst
# ---------------------------------------------------------------------------

def _render_yaml(
    job_name: str = "strix-benchmark-test-20260501-120000",
    branch: str = "main",
    shape: str = "1,2048,4,128,128",
    chunk_size: str = "64",
    kernel_module: str = "kernels.chunk_kda_fwd",
    tpu_type: str = "v7x",
    tpu_topology: str = "2x2x1",
    tpu_chips: str | None = None,
) -> str:
    """Render the YAML template by substituting ${VAR} placeholders.

    Uses Python string substitution to replicate envsubst behavior,
    avoiding a dependency on GNU gettext being installed.
    ``tpu_chips`` defaults to the product of the topology dimensions.
    """
    if tpu_chips is None:
        import math
        tpu_chips = str(math.prod(int(d) for d in tpu_topology.split("x")))
    # Derive accelerator label: v7x -> tpu7x (strip leading 'v')
    tpu_accelerator = "tpu" + (tpu_type[1:] if tpu_type.startswith("v") else tpu_type)
    mapping = {
        "JOB_NAME": job_name,
        "BRANCH": branch,
        "SHAPE": shape,
        "CHUNK_SIZE": chunk_size,
        "KERNEL_MODULE": kernel_module,
        "TPU_TYPE": tpu_type,
        "TPU_TOPOLOGY": tpu_topology,
        "TPU_CHIPS": tpu_chips,
        "TPU_ACCELERATOR": tpu_accelerator,
    }
    text = _YAML_TEMPLATE.read_text()
    for key, value in mapping.items():
        text = text.replace(f"${{{key}}}", value)
        text = text.replace(f"${key}", value)
    return text


# ===================================================================
# Part 1: benchmark_job.yaml template tests
# ===================================================================


class TestYamlTemplateVariables:
    """YAML template contains all required envsubst variables."""

    def test_template_file_exists(self):
        assert _YAML_TEMPLATE.exists(), f"{_YAML_TEMPLATE} does not exist"

    def test_contains_job_name_variable(self):
        content = _YAML_TEMPLATE.read_text()
        assert "$JOB_NAME" in content or "${JOB_NAME}" in content

    def test_contains_branch_variable(self):
        content = _YAML_TEMPLATE.read_text()
        assert "$BRANCH" in content or "${BRANCH}" in content

    def test_contains_shape_variable(self):
        content = _YAML_TEMPLATE.read_text()
        assert "$SHAPE" in content or "${SHAPE}" in content

    def test_contains_chunk_size_variable(self):
        content = _YAML_TEMPLATE.read_text()
        assert "$CHUNK_SIZE" in content or "${CHUNK_SIZE}" in content

    def test_contains_kernel_module_variable(self):
        content = _YAML_TEMPLATE.read_text()
        assert "$KERNEL_MODULE" in content or "${KERNEL_MODULE}" in content

    def test_contains_tpu_type_variable(self):
        content = _YAML_TEMPLATE.read_text()
        assert "$TPU_ACCELERATOR" in content or "${TPU_ACCELERATOR}" in content

    def test_contains_tpu_topology_variable(self):
        content = _YAML_TEMPLATE.read_text()
        assert "$TPU_TOPOLOGY" in content or "${TPU_TOPOLOGY}" in content

    def test_contains_tpu_chips_variable(self):
        content = _YAML_TEMPLATE.read_text()
        assert "$TPU_CHIPS" in content or "${TPU_CHIPS}" in content


class TestYamlRenderedStructure:
    """envsubst-rendered YAML has valid K8s Job structure."""

    @pytest.fixture()
    def rendered(self):
        import yaml  # PyYAML is needed for parsing; installed via dev deps
        return yaml.safe_load(_render_yaml())

    def test_api_version(self, rendered):
        assert rendered["apiVersion"] == "batch/v1"

    def test_kind_is_job(self, rendered):
        assert rendered["kind"] == "Job"

    def test_metadata_name(self, rendered):
        assert rendered["metadata"]["name"] == "strix-benchmark-test-20260501-120000"

    def test_service_account(self, rendered):
        pod_spec = rendered["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == "gcs-account"

    def test_restart_policy_never(self, rendered):
        pod_spec = rendered["spec"]["template"]["spec"]
        assert pod_spec["restartPolicy"] == "Never"

    def test_backoff_limit_zero(self, rendered):
        assert rendered["spec"]["backoffLimit"] == 0

    def test_tpu_resource_limit(self, rendered):
        container = rendered["spec"]["template"]["spec"]["containers"][0]
        resources = container["resources"]["limits"]
        assert "google.com/tpu" in resources
        assert resources["google.com/tpu"] == 4  # 2x2x1 = 4 chips

    def test_tpu_topology_annotation(self, rendered):
        annotations = rendered["spec"]["template"]["metadata"]["annotations"]
        assert annotations.get("cloud.google.com/gke-tpu-topology") == "2x2x1"

    def test_tpu_type_selector(self, rendered):
        node_selector = rendered["spec"]["template"]["spec"]["nodeSelector"]
        assert node_selector.get("cloud.google.com/gke-tpu-accelerator") == "tpu7x"

    def test_tpu_topology_in_node_selector(self, rendered):
        node_selector = rendered["spec"]["template"]["spec"]["nodeSelector"]
        assert node_selector.get("cloud.google.com/gke-tpu-topology") == "2x2x1"

    def test_container_has_env_vars(self, rendered):
        container = rendered["spec"]["template"]["spec"]["containers"][0]
        env_names = {e["name"] for e in container.get("env", [])}
        # At minimum, the runner needs to know the kernel module, shape, etc.
        assert "KERNEL_MODULE" in env_names
        assert "SHAPE" in env_names

    def test_rendered_yaml_no_unsubstituted_vars(self):
        """After envsubst, no ${VAR} placeholders should remain."""
        rendered_text = _render_yaml()
        remaining = re.findall(r'\$\{\w+\}', rendered_text)
        assert remaining == [], f"Unsubstituted variables remain: {remaining}"


class TestYamlCustomTopology:
    """Topology changes propagate correctly through the template."""

    def test_4x4x1_topology_annotation(self):
        import yaml
        rendered = yaml.safe_load(_render_yaml(tpu_topology="4x4x1"))
        annotations = rendered["spec"]["template"]["metadata"]["annotations"]
        assert annotations["cloud.google.com/gke-tpu-topology"] == "4x4x1"

    def test_4x4x1_topology_chip_count(self):
        import yaml
        rendered = yaml.safe_load(_render_yaml(tpu_topology="4x4x1"))
        container = rendered["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["limits"]["google.com/tpu"] == 16

    def test_2x2x1_topology_chip_count(self):
        import yaml
        rendered = yaml.safe_load(_render_yaml(tpu_topology="2x2x1"))
        container = rendered["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["limits"]["google.com/tpu"] == 4


# ===================================================================
# Part 2: run_benchmark.sh tests
# ===================================================================


class TestShellScriptExists:
    """run_benchmark.sh exists and is executable."""

    def test_script_file_exists(self):
        assert _SHELL_SCRIPT.exists(), f"{_SHELL_SCRIPT} does not exist"

    def test_script_is_executable(self):
        mode = os.stat(_SHELL_SCRIPT).st_mode
        assert mode & stat.S_IXUSR, "Script is not executable by owner"


class TestShellScriptContent:
    """run_benchmark.sh contains required structural elements."""

    @pytest.fixture()
    def script_content(self):
        return _SHELL_SCRIPT.read_text()

    def test_has_shebang(self, script_content):
        assert script_content.startswith("#!/")

    def test_has_set_euo_pipefail(self, script_content):
        assert "set -euo pipefail" in script_content

    def test_accepts_kernel_positional_arg(self, script_content):
        # The script should read the first positional arg as the kernel module
        # Typically: KERNEL_MODULE="${1:?...}" or similar
        assert "KERNEL_MODULE" in script_content

    def test_accepts_shape_option(self, script_content):
        assert "--shape" in script_content

    def test_accepts_chunk_size_option(self, script_content):
        assert "--chunk-size" in script_content

    def test_accepts_tpu_type_option(self, script_content):
        assert "--tpu-type" in script_content

    def test_accepts_tpu_topology_option(self, script_content):
        assert "--tpu-topology" in script_content

    def test_generates_job_name(self, script_content):
        # Should have the strix-benchmark-{slug}-{timestamp} pattern
        assert "strix-benchmark" in script_content

    def test_computes_tpu_chips(self, script_content):
        assert "TPU_CHIPS" in script_content

    def test_uses_envsubst(self, script_content):
        assert "envsubst" in script_content

    def test_uses_kubectl_apply(self, script_content):
        assert "kubectl apply" in script_content

    def test_uses_kubectl_wait(self, script_content):
        assert "kubectl wait" in script_content

    def test_uses_gcloud_storage_cp(self, script_content):
        assert "gcloud storage cp" in script_content

    def test_uses_tar_extraction(self, script_content):
        assert "tar" in script_content

    def test_has_cleanup_trap(self, script_content):
        # Script must include a trap for cleanup
        assert "trap" in script_content

    def test_cleanup_deletes_k8s_job(self, script_content):
        assert "kubectl delete" in script_content

    def test_references_gcs_bucket(self, script_content):
        assert "gs://poc_profile/" in script_content

    def test_references_yaml_template(self, script_content):
        assert "benchmark_job.yaml" in script_content


class TestShellScriptArgParsing:
    """run_benchmark.sh parses arguments correctly (dry run via --help or parse-only)."""

    def test_script_prints_usage_without_args(self):
        """Script should fail with usage message when no args provided."""
        result = subprocess.run(
            ["bash", str(_SHELL_SCRIPT)],
            capture_output=True,
            text=True,
        )
        # Should exit non-zero when no kernel argument given
        assert result.returncode != 0

    def test_script_rejects_unknown_flag(self):
        result = subprocess.run(
            ["bash", str(_SHELL_SCRIPT), "kernels.test", "--bogus"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


class TestYamlEmptyChunkSize:
    """Empty chunk-size renders valid YAML."""

    def test_empty_chunk_size_still_valid(self):
        import yaml
        rendered = yaml.safe_load(_render_yaml(chunk_size=""))
        container = rendered["spec"]["template"]["spec"]["containers"][0]
        env_map = {e["name"]: e["value"] for e in container["env"]}
        assert env_map["CHUNK_SIZE"] == ""
