"""Shared memory hygiene for the pipeline stages."""


def free_memory():
    import gc
    gc.collect()
