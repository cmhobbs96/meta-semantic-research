# scripts/util/clear_memory.py
import gc
import torch

def clear_memory(*objects_to_clear):
    """
    Clears memory from passed objects, garbage collection,
    and empties CUDA cache if available.
    """
    for obj in objects_to_clear:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
