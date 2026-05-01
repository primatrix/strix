"""Kernel modules for the Strix benchmark pipeline.

Each kernel file in this directory must export two names:

``kernel_fn``
    A callable that constructs inputs, compiles the kernel via JAX, and
    returns a zero-argument closure suitable for benchmarking::

        def kernel_fn(
            num_tokens=1024,
            hidden_size=4096,
            dtype=jnp.bfloat16,
            **kwargs,
        ) -> Callable[[], Any]:
            ...
            def run():
                return compiled_fn(inputs)
            return run

``config``
    A plain ``dict`` holding default parameters for the kernel::

        config = {
            "default_shape": {...},
            "dtype": "bfloat16",
            "tpu_type": "v7x",
            "tpu_topology": "2x2x1",
            "description": "Short human-readable summary",
        }
"""
