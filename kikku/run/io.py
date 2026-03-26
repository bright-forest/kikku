"""Solution persistence.

Saves/loads solution nests as numpy archives + JSON metadata.
No dynx/stagecraft/heptapodx dependency.
Only requires numpy + json (standard library).
"""

import json
from pathlib import Path
import numpy as np


def _flatten_arrays(obj, prefix=''):
    """Recursively extract arrays from a nested dict/list structure.

    Returns (arrays_dict, skeleton) where skeleton mirrors the
    original structure with arrays replaced by their dotted keys.
    """
    arrays = {}

    if isinstance(obj, np.ndarray):
        arrays[prefix] = obj
        return arrays, f'__array__:{prefix}'

    if isinstance(obj, list):
        skeleton = []
        for i, item in enumerate(obj):
            sub_prefix = f'{prefix}.{i}' if prefix else str(i)
            sub_arrays, sub_skel = _flatten_arrays(item, sub_prefix)
            arrays.update(sub_arrays)
            skeleton.append(sub_skel)
        return arrays, skeleton

    if isinstance(obj, dict):
        skeleton = {}
        for k, v in obj.items():
            sub_prefix = f'{prefix}.{k}' if prefix else k
            sub_arrays, sub_skel = _flatten_arrays(v, sub_prefix)
            arrays.update(sub_arrays)
            skeleton[k] = sub_skel
        return arrays, skeleton

    # Scalar/string: keep as-is in skeleton
    return arrays, _to_json_safe(obj)


def _to_json_safe(val):
    """Convert numpy scalars to Python types for JSON."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _rebuild_from_skeleton(skeleton, arrays):
    """Reconstruct the original nested structure from skeleton + arrays."""
    if isinstance(skeleton, str) and skeleton.startswith('__array__:'):
        key = skeleton[len('__array__:'):]
        return arrays[key]

    if isinstance(skeleton, list):
        return [_rebuild_from_skeleton(item, arrays) for item in skeleton]

    if isinstance(skeleton, dict):
        return {
            k: _rebuild_from_skeleton(v, arrays)
            for k, v in skeleton.items()
        }

    return skeleton


def save_solution(
    path,
    nest: dict,
    metadata: dict | None = None,
) -> Path:
    """Save solution nest to disk.

    Creates:
      {path}/
        solutions.npz   - numpy arrays (flattened from nest)
        metadata.json   - metadata + nest structure skeleton

    Arrays are extracted recursively from nest and saved with dotted
    keys (e.g. '0.keeper_cons.dcsn.c' for solutions[0]).
    Non-array values (scalars, strings) go to metadata.

    Parameters
    ----------
    path : output directory (created if needed)
    nest : solution dict from model's solve().
    metadata : optional dict (method, grid_size, timestamp, etc.)

    Returns
    -------
    Path to the saved directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    arrays, skeleton = _flatten_arrays(nest)

    np.savez_compressed(path / 'solutions.npz', **arrays)

    meta = {
        'skeleton': skeleton,
        'user_metadata': metadata or {},
        'array_keys': sorted(arrays.keys()),
    }
    with open(path / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2, default=_to_json_safe)

    return path


def load_solution(path) -> tuple:
    """Load solution nest from disk.

    Returns
    -------
    (nest, metadata)
        nest has the same structure as the original.
    """
    path = Path(path)

    with open(path / 'metadata.json') as f:
        meta = json.load(f)

    npz = np.load(path / 'solutions.npz', allow_pickle=False)
    arrays = dict(npz)

    nest = _rebuild_from_skeleton(meta['skeleton'], arrays)

    return nest, meta.get('user_metadata', {})
