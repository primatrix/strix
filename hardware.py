from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareSpec:
    """
    Extremely simple hardware model for a TPU-like device.

    The default values are a coarse approximation of a TPU v6e-like chip.
    We expose two kinds of parameters:
      * time-based throughput knobs used by the cost model:
          - peak_bf16_flops_per_s
          - peak_hbm_bytes_per_s
          - time_unit_s (e.g., 1e-9 for nanoseconds)
          - dispatch_overhead_ns / dma_setup_overhead_ns
      * descriptive fields mirroring public v6e specs.
    """

    # Human-readable identifier.
    name: str = "TPU v6e"

    # Per-chip specs.
    peak_bf16_tflops: float = 918.0
    peak_int8_tops: float = 1836.0
    hbm_capacity_gb: int = 32
    hbm_bandwidth_gbps: float = 1600.0
    ici_bandwidth_gbps: float = 3200.0
    ici_ports_per_chip: int = 4

    # Host / system-level specs.
    host_dram_gib: int = 1536
    chips_per_host: int = 8
    pod_size_chips: int = 256
    pod_bf16_pflops: float = 234.9
    pod_allreduce_tbps: float = 102.4
    pod_bisection_tbps: float = 3.2
    host_nic_config: str = "4 x 200 Gbps NIC"
    pod_dc_bandwidth_tbps: float = 25.6

    # Special accelerator features.
    special_features: str = "SparseCore"

    # ------------------------------------------------------------------
    # Core knobs used by the simulator / cost model (time-based).
    # ------------------------------------------------------------------

    # Time unit used for estimates (seconds). We model everything in
    # nanoseconds by default (1e-9 s).
    time_unit_s: float = 1e-9

    @property
    def peak_bf16_flops_per_s(self) -> float:
        """Peak BF16 FLOPs per second."""
        return self.peak_bf16_tflops * 1e12

    @property
    def peak_hbm_bytes_per_s(self) -> float:
        """Peak HBM bandwidth in bytes per second."""
        return self.hbm_bandwidth_gbps * 1e9

    @property
    def flops_per_time_unit(self) -> float:
        """Peak BF16 FLOPs in one time unit."""
        return self.peak_bf16_flops_per_s * self.time_unit_s

    @property
    def hbm_bytes_per_time_unit(self) -> float:
        """Peak HBM bytes in one time unit."""
        return self.peak_hbm_bytes_per_s * self.time_unit_s

    # Fixed per-dispatch overheads (in nanoseconds).
    dispatch_overhead_ns: int = 50
    dma_setup_overhead_ns: int = 200

    # MXU (Matrix Multiply Unit) specifications for v6e.
    # v6e has 256x256 ALUs arranged in a systolic array.
    mxu_width: int = 256  # MXU width (K and N dimensions)
    mxu_height: int = 256  # MXU height (maximum M dimension)

    # Each vmatmul.mubr operation:
    # - RHS (weights) is always 256x256 (fills the MXU width)
    # - LHS (activations) is M x 256, where M is inferred from vector shape
    #   e.g., vector<8x128x2xbf16> -> M = 8*2 = 16 rows
    # - The 128 in vector type gets padded to 256 to match MXU width
    # FLOPs per vmatmul.mubr = 2 * M * 256 * 256, where M varies
    # Note: flops_per_vmatmul_mubr is removed, calculate dynamically instead
