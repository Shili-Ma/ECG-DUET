import os


def resolve_results_paths(results_dir: str, save_dir: str | None, cache_root: str | None):
    os.makedirs(results_dir, exist_ok=True)
    if save_dir is None:
        save_dir = os.path.join(results_dir, "main")
    elif not os.path.isabs(save_dir) and not os.path.normpath(save_dir).startswith(os.path.normpath(results_dir) + os.sep):
        save_dir = os.path.join(results_dir, save_dir)
    if cache_root is None:
        cache_root = os.path.join(results_dir, "cache")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_root, exist_ok=True)
    return save_dir, cache_root
