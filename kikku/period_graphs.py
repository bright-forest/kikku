"""Period graph construction and solve-order topology derivation.

The period graph is a ``networkx.DiGraph`` derived from a canonical
period dict.  Kikku's job ends when it returns the solve-order
topology; the application layer pulls operators from its own
numModel and solves.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


def _extract_interface(mod: Any) -> dict:
    """Extract prestate / poststate interface from a SymbolicModel.

    For branching stages (``kind == 'branching'``), the primary
    representation is ``branch_poststates`` (branch label -> field
    names).  The flat ``poststate`` list is deduplicated and kept
    for informational / backward-compat purposes only — graph
    wiring must always go through the per-branch structure.

    Returns
    -------
    dict with keys:
        prestate : list[str]
        poststate : list[str]          (flat, deduplicated)
        branch_poststates : dict | None  (branch -> list[str] if branching)
        kind : str                       ('branching' or 'standard')
    """
    symbols = mod.symbols
    prestate = list(symbols.get("prestate", []))

    kind = getattr(mod, "kind", "standard") or "standard"

    branch_poststates = None
    if kind == "branching" and hasattr(mod, "branch_poststates"):
        raw = mod.branch_poststates
        branch_poststates = {
            branch: list(fields.keys()) if isinstance(fields, dict) else list(fields)
            for branch, fields in raw.items()
        }

        seen: dict[str, str] = {}
        for branch, fields in branch_poststates.items():
            for f in fields:
                if f in seen:
                    raise ValueError(
                        f"Poststate field '{f}' appears in branches "
                        f"'{seen[f]}' and '{branch}' of stage "
                        f"'{getattr(mod, 'name', '?')}'. "
                        f"Field names must be disjoint across branches "
                        f"(spec 0.1l flat-disjointness rule)."
                    )
                seen[f] = branch

        poststate = list(dict.fromkeys(
            f for fields in branch_poststates.values() for f in fields
        ))
    else:
        poststate = list(symbols.get("poststates", []))

    return {
        "prestate": prestate,
        "poststate": poststate,
        "branch_poststates": branch_poststates,
        "kind": kind,
    }


def period_to_graph(period_dict: dict) -> nx.DiGraph:
    """Build a directed graph from a canonical period dict.

    Connectivity is determined by two sources (no ordering input):

    1. **Explicit connectors** from ``period_dict["connectors"]``.
       Each entry ``{src_field: tgt_field}`` creates an edge from the
       stage whose poststate contains *src_field* to the stage whose
       prestate contains *tgt_field*, with ``rename={src: tgt}``.

    2. **Common-namespace matching.**  For every pair (A, B) where a
       poststate field of A matches a prestate field of B and no
       explicit connector already covers that field, an identity edge
       is added with ``rename={}``.

    Parameters
    ----------
    period_dict : dict
        Canonical period dict with ``"stages"`` and ``"connectors"``.

    Returns
    -------
    nx.DiGraph
        Nodes carry ``prestate``, ``poststate``, ``kind``,
        ``branch_poststates``, and optionally ``mod``.
        Edges carry ``rename`` (dict) and optionally ``branch`` (str).
    """
    stages = period_dict["stages"]
    connectors = period_dict.get("connectors", [])

    G = nx.DiGraph()

    interfaces: dict[str, dict] = {}
    for name, mod in stages.items():
        iface = _extract_interface(mod)
        interfaces[name] = iface
        G.add_node(
            name,
            prestate=iface["prestate"],
            poststate=iface["poststate"],
            kind=iface["kind"],
            branch_poststates=iface["branch_poststates"],
            mod=mod,
        )

    covered_fields: set[str] = set()

    for conn in connectors:
        for src_field, tgt_field in conn.items():
            src_stage = _find_stage_with_poststate(interfaces, src_field)
            tgt_stage = _find_stage_with_prestate(interfaces, tgt_field)
            if src_stage and tgt_stage:
                attrs = {"rename": {src_field: tgt_field}}
                bp = interfaces[src_stage]["branch_poststates"]
                if bp:
                    for branch, fields in bp.items():
                        if src_field in fields:
                            attrs["branch"] = branch
                            break
                G.add_edge(src_stage, tgt_stage, **attrs)
                covered_fields.add(src_field)

    for name_a, iface_a in interfaces.items():
        bp = iface_a["branch_poststates"]
        if bp:
            for branch, fields in bp.items():
                for field in fields:
                    if field in covered_fields:
                        continue
                    tgt = _find_stage_with_prestate(interfaces, field, exclude=name_a)
                    if tgt:
                        G.add_edge(name_a, tgt, rename={}, branch=branch)
                        covered_fields.add(field)
        else:
            for field in iface_a["poststate"]:
                if field in covered_fields:
                    continue
                tgt = _find_stage_with_prestate(interfaces, field, exclude=name_a)
                if tgt:
                    G.add_edge(name_a, tgt, rename={})
                    covered_fields.add(field)

    if not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        raise ValueError(f"Period graph has a cycle: {cycle}")

    return G


def backward_paths(
    graph: nx.DiGraph, inter_connector: dict
) -> list[list[str]]:
    """Return independently solvable stage chains for backward iteration.

    Terminal stages (forward) are those whose poststates overlap with
    the inter-period connector keys.  These are initial in backward
    traversal.  The function returns a list of "waves" — each wave is
    a list of stages that can be solved in parallel once all previous
    waves are complete.

    Parameters
    ----------
    graph : nx.DiGraph
        Period graph from :func:`period_to_graph`.
    inter_connector : dict
        Inter-period connector, e.g. ``{"b": "a", "b_ret": "a_ret"}``.

    Returns
    -------
    list[list[str]]
        Ordered waves.  For the retirement model:
        ``[["work_cons", "retire_cons"], ["labour_mkt_decision"]]``
        (first wave is parallel, second depends on both).
    """
    entry_fields = set(inter_connector.keys())

    initial: set[str] = set()
    for node, data in graph.nodes(data=True):
        post = set(data.get("poststate", []))
        if post & entry_fields:
            initial.add(node)

    reversed_G = graph.reverse(copy=True)

    resolved: set[str] = set()
    waves: list[list[str]] = []
    all_nodes = set(graph.nodes)

    while resolved != all_nodes:
        wave = []
        for node in all_nodes - resolved:
            preds_in_reversed = set(reversed_G.predecessors(node))
            unresolved_deps = preds_in_reversed - resolved
            if not unresolved_deps:
                if node in initial or _all_successors_resolved(
                    reversed_G, node, resolved
                ):
                    wave.append(node)

        if not wave:
            remaining = all_nodes - resolved
            wave = sorted(remaining)

        wave.sort()
        waves.append(wave)
        resolved.update(wave)

    return waves


def forward_order(graph: nx.DiGraph) -> list[str]:
    """Return stages in forward (topological) order.

    Parameters
    ----------
    graph : nx.DiGraph
        Period graph from :func:`period_to_graph`.

    Returns
    -------
    list[str]
        Topologically sorted stage names.
    """
    return list(nx.topological_sort(graph))


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _find_stage_with_poststate(
    interfaces: dict[str, dict], field: str, *, exclude: str | None = None
) -> str | None:
    for name, iface in interfaces.items():
        if name == exclude:
            continue
        if field in iface["poststate"]:
            return name
    return None


def _find_stage_with_prestate(
    interfaces: dict[str, dict], field: str, *, exclude: str | None = None
) -> str | None:
    for name, iface in interfaces.items():
        if name == exclude:
            continue
        if field in iface["prestate"]:
            return name
    return None


def _all_successors_resolved(
    reversed_G: nx.DiGraph, node: str, resolved: set[str]
) -> bool:
    """In the reversed graph, check if all successors of *node* are resolved."""
    return all(s in resolved for s in reversed_G.successors(node))
