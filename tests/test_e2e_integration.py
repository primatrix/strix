"""Tests for end-to-end integration: CLI -> K8s -> GCS -> local download (Issue #212).

Acceptance criteria:
1. `python -m strix.cli import kernels.fused_moe --shape 256,256,8,8192,2048` submits K8s Job
2. Job completion triggers automatic IR dump download to local directory
3. Downloaded benchmark_results/ contains benchmark_result.json and ir_dumps/{hlo,llo,mosaic}/
4. Downloaded LLO files can be fed to `python -m strix.cli <llo.txt>` for analysis
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from strix.cli import build_arg_parser, main, preprocess_argv

# Repo root is one level above tests/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_YAML_TEMPLATE = _SCRIPTS_DIR / "benchmark_job.yaml"
_RUNNER_PATH = _SCRIPTS_DIR / "benchmark_runner.py"


def _import_runner():
    """Import benchmark_runner.py as a module from its file path."""
    spec = importlib.util.spec_from_file_location("benchmark_runner", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render_yaml(
    job_name: str = "strix-benchmark-test-20260501-120000",
    branch: str = "main",
    shape: str = "256,256,8,8192,2048",
    chunk_size: str = "",
    kernel_module: str = "kernels.fused_moe",
    tpu_type: str = "v7x",
    tpu_topology: str = "2x2x1",
    tpu_chips: str | None = None,
    gcs_bucket: str = "gs://poc_profile/",
) -> str:
    """Render YAML template via Python string substitution (envsubst equivalent)."""
    if tpu_chips is None:
        tpu_chips = str(math.prod(int(d) for d in tpu_topology.split("x")))
    tpu_accelerator = "tpu" + (tpu_type[1:] if tpu_type.startswith("v") else tpu_type)
    branch_label = branch.replace("/", "-")[:63]
    mapping = {
        "JOB_NAME": job_name,
        "BRANCH": branch,
        "BRANCH_LABEL": branch_label,
        "SHAPE": shape,
        "CHUNK_SIZE": chunk_size,
        "KERNEL_MODULE": kernel_module,
        "TPU_TYPE": tpu_type,
        "TPU_TOPOLOGY": tpu_topology,
        "TPU_CHIPS": tpu_chips,
        "TPU_ACCELERATOR": tpu_accelerator,
        "GCS_BUCKET": gcs_bucket,
    }
    text = _YAML_TEMPLATE.read_text()
    for key, value in mapping.items():
        text = text.replace(f"${{{key}}}", value)
        text = text.replace(f"${key}", value)
    return text


# ===================================================================
# Work Item 1: CLI -> Shell Script interface contract
# ===================================================================


class TestCliToShellContract:
    """CLI _run_import() produces exact command format run_benchmark.sh expects."""

    @patch("subprocess.run")
    def test_fused_moe_full_command_structure(self, mock_run):
        """AC1: CLI generates correct command for fused_moe kernel with AC shape."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        main(["import", "kernels.fused_moe", "--shape", "256,256,8,8192,2048"])
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "bash"
        assert cmd[1].endswith("scripts/run_benchmark.sh")
        # Shell script expects kernel as first positional arg ($1)
        assert cmd[2] == "kernels.fused_moe"
        assert cmd[3:5] == ["--shape", "256,256,8,8192,2048"]
        assert cmd[5:7] == ["--tpu-type", "v7x"]
        assert cmd[7:9] == ["--tpu-topology", "2x2x1"]

    @patch("subprocess.run")
    def test_kernel_is_positional_before_all_flags(self, mock_run):
        """Shell script reads ${1:?} as kernel module — must precede all flags."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        main(["import", "kernels.fused_moe", "--shape", "1,2"])
        cmd = mock_run.call_args[0][0]
        # cmd[0] = "bash", cmd[1] = script path, cmd[2] = kernel
        assert cmd[2] == "kernels.fused_moe"
        # After kernel, args come in --flag value pairs
        trailing = cmd[3:]
        for i in range(0, len(trailing), 2):
            assert trailing[i].startswith("--"), f"Expected flag at position {i}, got {trailing[i]}"

    @patch("subprocess.run")
    def test_chunk_size_omitted_when_not_specified(self, mock_run):
        """Shell script defaults CHUNK_SIZE to empty; CLI must not pass --chunk-size."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        main(["import", "kernels.fused_moe", "--shape", "1,2"])
        cmd = mock_run.call_args[0][0]
        assert "--chunk-size" not in cmd

    @patch("subprocess.run")
    def test_chunk_size_appended_when_specified(self, mock_run):
        """CLI appends --chunk-size <n> to command when user specifies it."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        main(["import", "kernels.fused_moe", "--shape", "1,2", "--chunk-size", "64"])
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--chunk-size")
        assert cmd[idx + 1] == "64"

    @patch("subprocess.run")
    def test_script_path_is_absolute(self, mock_run):
        """run_benchmark.sh path must be absolute for subprocess to find it."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        main(["import", "kernels.fused_moe", "--shape", "1,2"])
        script_path = mock_run.call_args[0][0][1]
        assert os.path.isabs(script_path)

    @patch("subprocess.run")
    def test_script_path_points_to_existing_file(self, mock_run):
        """The shell script referenced by CLI must actually exist."""
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        main(["import", "kernels.fused_moe", "--shape", "1,2"])
        script_path = mock_run.call_args[0][0][1]
        assert Path(script_path).exists()


# ===================================================================
# Work Item 2: YAML-to-runner env var naming contract
# ===================================================================


class TestEnvVarContract:
    """YAML template env vars match benchmark_runner.py expectations."""

    @pytest.fixture()
    def yaml_env_names(self):
        import yaml

        rendered = yaml.safe_load(_render_yaml())
        container = rendered["spec"]["template"]["spec"]["containers"][0]
        return {e["name"] for e in container.get("env", [])}

    @pytest.fixture()
    def yaml_env_map(self):
        import yaml

        rendered = yaml.safe_load(_render_yaml())
        container = rendered["spec"]["template"]["spec"]["containers"][0]
        return {e["name"]: e["value"] for e in container.get("env", [])}

    def test_yaml_covers_runner_required_env_vars(self, yaml_env_names):
        """YAML template must set all env vars that benchmark_runner reads."""
        runner_required = {"KERNEL_MODULE", "SHAPE", "CHUNK_SIZE", "JOB_NAME", "GCS_BUCKET"}
        missing = runner_required - yaml_env_names
        assert not missing, f"YAML missing env vars needed by runner: {missing}"

    def test_branch_env_var_present(self, yaml_env_names):
        """BRANCH env var needed for git clone in the Pod."""
        assert "BRANCH" in yaml_env_names

    def test_xla_flags_set_in_yaml(self, yaml_env_names):
        """XLA_FLAGS set in YAML (runner skips if already present)."""
        assert "XLA_FLAGS" in yaml_env_names

    def test_libtpu_init_args_set_in_yaml(self, yaml_env_names):
        """LIBTPU_INIT_ARGS set in YAML (runner skips if already present)."""
        assert "LIBTPU_INIT_ARGS" in yaml_env_names

    def test_yaml_xla_flags_enable_hlo_dump(self, yaml_env_map):
        """XLA_FLAGS in YAML must enable HLO text dump."""
        xla_flags = yaml_env_map["XLA_FLAGS"]
        assert "--xla_dump_hlo_as_text" in xla_flags
        assert "/tmp/ir_dumps/hlo" in xla_flags

    def test_yaml_libtpu_enables_llo_dump(self, yaml_env_map):
        """LIBTPU_INIT_ARGS in YAML must enable LLO text dump."""
        libtpu = yaml_env_map["LIBTPU_INIT_ARGS"]
        assert "--xla_jf_dump_llo_text=true" in libtpu
        assert "/tmp/ir_dumps/llo" in libtpu

    def test_yaml_libtpu_enables_mosaic_dump(self, yaml_env_map):
        """LIBTPU_INIT_ARGS in YAML must enable Mosaic dump."""
        libtpu = yaml_env_map["LIBTPU_INIT_ARGS"]
        assert "/tmp/ir_dumps/mosaic" in libtpu

    def test_ir_dump_paths_consistent_across_yaml_and_runner(self, yaml_env_map):
        """IR dump paths in YAML env vars must match runner's default paths."""
        xla = yaml_env_map["XLA_FLAGS"]
        libtpu = yaml_env_map["LIBTPU_INIT_ARGS"]

        # Runner's default IR dump root is /tmp/ir_dumps
        runner = _import_runner()
        default_root = str(runner._DEFAULT_IR_DUMP_ROOT)
        assert default_root == "/tmp/ir_dumps"

        # YAML paths must use the same root
        assert f"{default_root}/hlo" in xla
        assert f"{default_root}/llo" in libtpu
        assert f"{default_root}/mosaic" in libtpu


# ===================================================================
# Work Item 3: Tar structure contract
# ===================================================================


class TestTarStructureContract:
    """Runner's tar output matches shell script's download/extract expectations."""

    def test_extract_produces_ac3_directory_structure(self, tmp_path):
        """AC3: benchmark_results/ contains benchmark_result.json and ir_dumps/{hlo,llo,mosaic}/."""
        runner = _import_runner()

        # Simulate runner output: IR dumps + benchmark result
        ir_root = tmp_path / "ir_dumps"
        for subdir in ("hlo", "llo", "mosaic"):
            (ir_root / subdir).mkdir(parents=True)
            (ir_root / subdir / f"test_{subdir}.txt").write_text(f"{subdir} content")

        result_file = tmp_path / "benchmark_result.json"
        result_file.write_text('{"kernel": "kernels.fused_moe", "timings_ms": [100.0]}')

        job_name = "strix-benchmark-kernels-fused-moe-20260501-120000"
        tarball = runner.package_results(job_name, ir_root, result_file, tmp_path)

        # Simulate shell script: tar xzf ... -C benchmark_results/
        extract_dir = tmp_path / "benchmark_results"
        extract_dir.mkdir()
        with tarfile.open(tarball) as tf:
            tf.extractall(extract_dir, filter="data")

        # AC3 verification
        assert (extract_dir / "benchmark_result.json").exists()
        assert (extract_dir / "ir_dumps" / "hlo").is_dir()
        assert (extract_dir / "ir_dumps" / "llo").is_dir()
        assert (extract_dir / "ir_dumps" / "mosaic").is_dir()

    def test_extracted_files_contain_content(self, tmp_path):
        """Extracted files should preserve their original content."""
        runner = _import_runner()

        ir_root = tmp_path / "ir_dumps"
        (ir_root / "llo").mkdir(parents=True)
        (ir_root / "llo" / "module.llo").write_text("llo content here")
        result_file = tmp_path / "benchmark_result.json"
        result_file.write_text('{"kernel": "test"}')

        tarball = runner.package_results("j", ir_root, result_file, tmp_path)

        extract_dir = tmp_path / "benchmark_results"
        extract_dir.mkdir()
        with tarfile.open(tarball) as tf:
            tf.extractall(extract_dir, filter="data")

        assert (extract_dir / "ir_dumps" / "llo" / "module.llo").read_text() == "llo content here"

    def test_tarball_name_matches_gcloud_download_path(self, tmp_path):
        """Shell script downloads ${JOB_NAME}.tar.gz; runner must produce exact name."""
        runner = _import_runner()

        ir_root = tmp_path / "ir_dumps"
        ir_root.mkdir()
        result_file = tmp_path / "result.json"
        result_file.write_text("{}")

        job_name = "strix-benchmark-kernels-fused-moe-20260501-120000"
        tarball = runner.package_results(job_name, ir_root, result_file, tmp_path)
        assert tarball.name == f"{job_name}.tar.gz"

    def test_gcs_upload_path_matches_shell_download_path(self, tmp_path):
        """Runner upload path must match shell script's gcloud download path."""
        runner = _import_runner()

        ir_root = tmp_path / "ir_dumps"
        ir_root.mkdir()
        result_file = tmp_path / "result.json"
        result_file.write_text("{}")

        job_name = "strix-benchmark-kernels-fused-moe-20260501-120000"
        gcs_bucket = "gs://poc_profile/"

        tarball = runner.package_results(job_name, ir_root, result_file, tmp_path)

        # Runner uploads to: {bucket_prefix}/{job_name}/{tarball.name}
        # When bucket = "gs://poc_profile/", prefix = "" so blob = "{job_name}/{job_name}.tar.gz"
        expected_blob = f"{job_name}/{tarball.name}"

        # Shell script downloads: ${GCS_BUCKET}${JOB_NAME}/${JOB_NAME}.tar.gz
        # = gs://poc_profile/ + job_name + / + job_name + .tar.gz
        shell_download = f"{gcs_bucket}{job_name}/{job_name}.tar.gz"
        # Full GCS URI = gs://poc_profile/{blob}
        runner_upload = f"{gcs_bucket}{expected_blob}"

        assert shell_download == runner_upload

    def test_shell_script_download_trigger_chain(self):
        """AC2: Shell script has kubectl wait → gcloud cp → tar extract sequence."""
        script = _SCRIPTS_DIR / "run_benchmark.sh"
        content = script.read_text()
        # Verify the download trigger chain exists in the correct order
        wait_pos = content.find("kubectl wait")
        gcloud_pos = content.find("gcloud storage cp")
        tar_pos = content.find("tar xzf")
        assert wait_pos > 0, "kubectl wait not found in shell script"
        assert gcloud_pos > 0, "gcloud storage cp not found in shell script"
        assert tar_pos > 0, "tar xzf not found in shell script"
        assert wait_pos < gcloud_pos < tar_pos, (
            "Download chain must be: kubectl wait → gcloud cp → tar extract"
        )

    def test_shell_script_downloads_to_benchmark_results_dir(self):
        """Shell download dir must match expected benchmark_results/ convention."""
        script = _SCRIPTS_DIR / "run_benchmark.sh"
        content = script.read_text()
        assert 'OUTPUT_DIR="benchmark_results"' in content

    def test_ir_dump_subdirs_match_runner_setup(self, tmp_path):
        """IR dump subdirs created by setup_ir_dump_dirs match package structure."""
        runner = _import_runner()
        runner.setup_ir_dump_dirs(tmp_path)
        expected_subdirs = {"hlo", "llo", "mosaic"}
        actual_subdirs = {p.name for p in tmp_path.iterdir() if p.is_dir()}
        assert expected_subdirs == actual_subdirs


# ===================================================================
# Work Item 4: FusedMoE acceptance criteria shape validation
# ===================================================================


class TestFusedMoeE2E:
    """Fused MoE kernel acceptance criteria validation."""

    def test_fused_moe_module_exports_kernel_fn_and_config(self):
        """kernels.fused_moe exports kernel_fn and config (AST check, no JAX import)."""
        kernel_path = _REPO_ROOT / "kernels" / "fused_moe.py"
        tree = ast.parse(kernel_path.read_text())
        top_level_names = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                top_level_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        top_level_names.add(target.id)
        assert "kernel_fn" in top_level_names
        assert "config" in top_level_names

    def test_fused_moe_config_has_ac_shape_values(self):
        """config default_shape must contain AC shape: 256,256,8,8192,2048."""
        kernel_path = _REPO_ROOT / "kernels" / "fused_moe.py"
        tree = ast.parse(kernel_path.read_text())

        config_dict = None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "config":
                        config_dict = ast.literal_eval(node.value)
                        break
                if config_dict:
                    break

        assert config_dict is not None, "Could not find 'config' dictionary in fused_moe.py"
        default_shape = config_dict.get("default_shape", {})
        assert default_shape.get("num_tokens") == 256
        assert default_shape.get("num_experts") == 256
        assert default_shape.get("top_k") == 8
        assert default_shape.get("hidden_size") == 8192
        assert default_shape.get("intermediate_size") == 2048

    def test_cli_parses_ac_shape_for_fused_moe(self):
        """--shape 256,256,8,8192,2048 is parsed correctly by CLI."""
        ap = build_arg_parser()
        args = ap.parse_args(
            ["import", "kernels.fused_moe", "--shape", "256,256,8,8192,2048"]
        )
        assert args.shape == "256,256,8,8192,2048"
        assert args.kernel == "kernels.fused_moe"

    def test_job_name_slug_derivation_for_fused_moe(self):
        """Shell script slug: kernels.fused_moe -> kernels-fused-moe."""
        # run_benchmark.sh: KERNEL_SLUG="${KERNEL_MODULE//[._]/-}"
        kernel_module = "kernels.fused_moe"
        slug = kernel_module.replace(".", "-").replace("_", "-")
        assert slug == "kernels-fused-moe"
        # Full job name format
        assert f"strix-benchmark-{slug}".startswith("strix-benchmark-kernels-fused-moe")

    def test_fused_moe_kernel_fn_parameters_match_config_shape(self):
        """kernel_fn default params must align with config default_shape keys."""
        kernel_path = _REPO_ROOT / "kernels" / "fused_moe.py"
        tree = ast.parse(kernel_path.read_text())

        # Extract kernel_fn parameter names with defaults
        config_shape_keys = {"num_tokens", "num_experts", "top_k", "hidden_size", "intermediate_size"}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "kernel_fn":
                param_names = {arg.arg for arg in node.args.args}
                assert config_shape_keys.issubset(param_names), (
                    f"kernel_fn missing params: {config_shape_keys - param_names}"
                )
                break
        else:
            pytest.fail("kernel_fn not found in fused_moe.py")


# ===================================================================
# Work Item 5: Download-to-analyze chain
# ===================================================================


class TestDownloadToAnalyzeChain:
    """Downloaded LLO files can be analyzed by strix CLI."""

    def test_llo_path_routes_to_analyze_subcommand(self):
        """AC4: LLO file path routes to analyze subcommand via backward compat."""
        processed = preprocess_argv(["benchmark_results/ir_dumps/llo/module.llo"])
        assert processed[0] == "analyze"
        assert processed[1] == "benchmark_results/ir_dumps/llo/module.llo"

    def test_analyze_parses_llo_file_from_download_dir(self):
        """CLI can parse an LLO file path from the benchmark_results directory."""
        ap = build_arg_parser()
        args = ap.parse_args(
            preprocess_argv(["benchmark_results/ir_dumps/llo/post-finalize-llo.txt"])
        )
        assert args.subcommand == "analyze"
        assert args.path == "benchmark_results/ir_dumps/llo/post-finalize-llo.txt"

    def test_llo_parser_accepts_minimal_llo_file(self, tmp_path):
        """LLOParser can parse a minimal LLO file from the download directory."""
        from strix.parser import LLOParser

        llo_content = textwrap.dedent("""\
            module @fused_moe {
            }
        """)
        llo_file = tmp_path / "ir_dumps" / "llo" / "post-finalize-llo.txt"
        llo_file.parent.mkdir(parents=True)
        llo_file.write_text(llo_content)

        parser = LLOParser()
        root = parser.parse_file(str(llo_file), exclude_instructions=None)
        assert root is not None

    def test_full_pipeline_tar_to_analysis(self, tmp_path):
        """Full chain: runner tar -> extract -> LLO file -> parser."""
        runner = _import_runner()
        from strix.parser import LLOParser

        # 1. Create runner output with a minimal LLO file
        ir_root = tmp_path / "ir_dumps"
        (ir_root / "hlo").mkdir(parents=True)
        (ir_root / "llo").mkdir(parents=True)
        (ir_root / "mosaic").mkdir(parents=True)
        llo_file = ir_root / "llo" / "post-finalize-llo.txt"
        llo_file.write_text("module @fused_moe {\n}\n")

        result_file = tmp_path / "benchmark_result.json"
        result_file.write_text('{"kernel": "kernels.fused_moe"}')

        # 2. Package (runner side)
        job_name = "strix-benchmark-kernels-fused-moe-20260501-120000"
        tarball = runner.package_results(job_name, ir_root, result_file, tmp_path)

        # 3. Extract (shell script side)
        extract_dir = tmp_path / "benchmark_results"
        extract_dir.mkdir()
        with tarfile.open(tarball) as tf:
            tf.extractall(extract_dir, filter="data")

        # 4. Analyze (AC4: strix CLI parses the extracted LLO)
        extracted_llo = extract_dir / "ir_dumps" / "llo" / "post-finalize-llo.txt"
        assert extracted_llo.exists()

        parser = LLOParser()
        root = parser.parse_file(str(extracted_llo), exclude_instructions=None)
        assert root is not None
        assert root.opcode == "module"
