"""MPI helpers for DDSL runners.

Thin wrappers that degrade gracefully when mpi4py is absent.
No model-specific code.
"""

try:
    from mpi4py import MPI as _MPI
    _HAS_MPI = True
except (ImportError, RuntimeError, OSError):
    _MPI = None
    _HAS_MPI = False


def get_comm():
    """Return MPI communicator or None if mpi4py unavailable."""
    if _HAS_MPI:
        return _MPI.COMM_WORLD
    return None


def rank_size(comm=None) -> tuple:
    """(rank, world_size). Returns (0, 1) if no MPI."""
    if comm is None:
        return (0, 1)
    return (comm.Get_rank(), comm.Get_size())


def is_root(comm=None) -> bool:
    """True if rank 0 or no MPI."""
    r, _ = rank_size(comm)
    return r == 0


def scatter_items(items: list, comm=None) -> list:
    """Distribute items across ranks. Each rank gets its slice."""
    if comm is None:
        return list(items)

    r, size = rank_size(comm)
    n = len(items)
    chunk = n // size
    remainder = n % size

    # First `remainder` ranks get one extra item
    if r < remainder:
        start = r * (chunk + 1)
        end = start + chunk + 1
    else:
        start = remainder * (chunk + 1) + (r - remainder) * chunk
        end = start + chunk

    return items[start:end]


def gather_results(local_results: list, comm=None) -> list:
    """Gather results from all ranks to rank 0.

    Returns full list on root, [] on others.
    """
    if comm is None:
        return list(local_results)

    all_results = comm.gather(local_results, root=0)
    if is_root(comm):
        flat = []
        for chunk in all_results:
            flat.extend(chunk)
        return flat
    return []


def bcast_item(item, comm=None, root=0):
    """Broadcast item from root to all ranks. Returns item if no MPI."""
    if comm is None:
        return item
    return comm.bcast(item, root=root)


def mpi_map(fn, items, comm=None) -> list:
    """Distribute fn(item) across ranks.

    Falls back to [fn(x) for x in items] if no MPI.
    """
    local = scatter_items(items, comm)
    results = [fn(x) for x in local]
    return gather_results(results, comm)
