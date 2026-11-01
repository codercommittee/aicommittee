"""Alignment: the pre-match a referee decides, and the units its decisions form.

Candidate pairs come from the pairwise matching of every two members over pair similarity at or
above 0.60; each is banded ``high`` (at or above the referee's ``accept_at``) or ``uncertain``
and flagged when it sits inside an inconsistent group. The referee confirms or rejects every
candidate pair and nothing else. Confirmed edges form units; a unit stands only when it holds
at most one relation of each member and every within-unit pair is confirmed; while that fails,
the weakest recorded edge inside the offending group drops (content-unordered key, exact ties
dropped together). Every component is returned, singletons included, with its support and its
medoid; the keep rule is applied afterwards by :func:`codercommittee.combine.combine`.

The pre-match's sha256 is taken over the exact dictionary shape the study stored, so stored
referee decisions stay bound to the pre-match they answered.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .data import ItemId, Member, Relation, Unit
from .dedupe import medoid, relation_key
from .matching import canonical_pair_matching
from .similarity import Similarity
from .strictjson import canonical_json

if TYPE_CHECKING:
    from .protocols import Referee

LOW_THRESHOLD = 0.60
Band = Literal["high", "uncertain"]


@dataclass(frozen=True)
class Edge:
    """One candidate pair: ``pair`` is ``"a|b"`` with the two node labels in sorted order."""

    pair: str
    a: str
    b: str
    similarity: float
    fields: Mapping[str, float]
    band: Band
    in_conflict: bool


def node_label(member: Member) -> str:
    """The node label of a member: ``name:index``."""
    return f"{member.name}:{member.index}"


def pair_label(a: Member, b: Member) -> str:
    """The pair label of two members: both node labels sorted and joined by ``|``."""
    low, high = sorted((node_label(a), node_label(b)))
    return f"{low}|{high}"


@dataclass(frozen=True)
class Prematch:
    """The quantitative pre-match: nodes, candidate edges, bands, and what the matching did."""

    low: float
    high: float
    nodes: tuple[Member, ...]
    edges: tuple[Edge, ...]
    match_counts: Mapping[str, int]
    fallback_pairs: tuple[str, ...]
    conflict_components: int
    item: ItemId | None = None
    label: str | None = None
    text: str = ""

    @property
    def candidate_pairs(self) -> tuple[str, ...]:
        """The candidate pairs' labels, in the edges' sorted order."""
        return tuple(edge.pair for edge in self.edges)

    def as_dict(self) -> dict[str, Any]:
        """The hashed shape: the dictionary the study stored and the referee's record names."""
        return {
            "low_threshold": float(self.low),
            "high_threshold": float(self.high),
            "nodes": [
                {
                    "node": node_label(member),
                    "member_id": member.name,
                    "index": member.index,
                    "relation": member.relation.as_dict(),
                }
                for member in self.nodes
            ],
            "edges": [
                {
                    "pair": edge.pair,
                    "a": edge.a,
                    "b": edge.b,
                    "similarity": edge.similarity,
                    **{f"{name}_similarity": value for name, value in edge.fields.items()},
                    "band": edge.band,
                    "in_conflict": edge.in_conflict,
                }
                for edge in self.edges
            ],
            "candidate_pairs": list(self.candidate_pairs),
            "pairwise_match_counts": dict(self.match_counts),
            "enumeration_fallback_pairs": list(self.fallback_pairs),
            "conflict_components": self.conflict_components,
        }

    @property
    def sha256(self) -> str:
        """The sha256 of the hashed shape; stored referee decisions are bound to it."""
        return hashlib.sha256(canonical_json(self.as_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Alignment:
    """What alignment produced: every unit, the pre-match, the decisions and the dropped edges."""

    units: tuple[Unit, ...]
    prematch: Prematch
    decisions: Mapping[str, bool]
    dropped: tuple[tuple[str, float], ...]


WeightedEdge = tuple[float, Member, Member]


def _candidates(members: Mapping[str, Sequence[Relation]], coders: Mapping[str, str] | None) -> dict[str, list[Member]]:
    """Members in sorted-name order, each relation a candidate; the coder is the name unless given."""
    for name in members:
        if "|" in name:
            raise ValueError(f"member name {name!r} holds '|', which the pair labels reserve")
    coder_of = coders or {}
    return {
        name: [Member(name, index, relation, coder_of.get(name, name)) for index, relation in enumerate(members[name])]
        for name in sorted(members)
    }


def _edge_graph(
    by_member: Mapping[str, list[Member]], sim: Similarity
) -> tuple[list[Member], list[WeightedEdge], dict[str, int], list[str]]:
    names = list(by_member)
    nodes = [member for name in names for member in by_member[name]]
    edges: list[WeightedEdge] = []
    match_counts: dict[str, int] = {}
    fallback: list[str] = []
    for name_a, name_b in itertools.combinations(names, 2):
        left, right = by_member[name_a], by_member[name_b]
        matching, used_fallback = canonical_pair_matching(left, right, LOW_THRESHOLD, sim)
        match_counts[f"{name_a}|{name_b}"] = len(matching)
        if used_fallback:
            fallback.append(f"{name_a}|{name_b}")
        for row, col, value in matching:
            edges.append((value, left[row], right[col]))
    return nodes, edges, match_counts, fallback


def _positions(nodes: Sequence[Member]) -> dict[int, int]:
    return {id(member): index for index, member in enumerate(nodes)}


def _components(nodes: Sequence[Member], active: Sequence[WeightedEdge]) -> list[list[Member]]:
    position = _positions(nodes)
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(nodes))}
    for _, a, b in active:
        adjacency[position[id(a)]].add(position[id(b)])
        adjacency[position[id(b)]].add(position[id(a)])
    seen: set[int] = set()
    components: list[list[Member]] = []
    for index in range(len(nodes)):
        if index in seen:
            continue
        stack = [index]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(adjacency[current] - seen)
        components.append([nodes[i] for i in sorted(component)])
    return components


def _edge_set(nodes: Sequence[Member], active: Sequence[WeightedEdge]) -> set[frozenset[int]]:
    position = _positions(nodes)
    return {frozenset((position[id(a)], position[id(b)])) for _, a, b in active}


def _is_unit(component: Sequence[Member], present: set[frozenset[int]], nodes: Sequence[Member]) -> bool:
    names = [member.name for member in component]
    if len(set(names)) != len(names):
        return False
    position = _positions(nodes)
    pairs = itertools.combinations(component, 2)
    return all(frozenset((position[id(a)], position[id(b)])) in present for a, b in pairs)


def prematch(
    members: Mapping[str, Sequence[Relation]],
    sim: Similarity,
    *,
    accept_at: float,
    coders: Mapping[str, str] | None = None,
    item: ItemId | None = None,
    label: str | None = None,
    text: str = "",
) -> Prematch:
    """The candidate pairs across members, banded at ``accept_at`` and flagged inside inconsistent groups."""
    if not 0.0 < LOW_THRESHOLD <= accept_at <= 1.0:
        raise ValueError("the accept threshold must lie between the candidate threshold and one")
    by_member = _candidates(members, coders)
    nodes, edges, match_counts, fallback = _edge_graph(by_member, sim)
    present = _edge_set(nodes, edges)
    conflict_nodes: set[int] = set()
    conflict_components = 0
    for component in _components(nodes, edges):
        if len(component) > 1 and not _is_unit(component, present, nodes):
            conflict_components += 1
            conflict_nodes.update(id(member) for member in component)
    records = sorted(
        (
            Edge(
                pair=pair_label(a, b),
                a=min(node_label(a), node_label(b)),
                b=max(node_label(a), node_label(b)),
                similarity=value,
                fields=sim.per_field(a.relation, b.relation),
                band="high" if value >= accept_at else "uncertain",
                in_conflict=id(a) in conflict_nodes and id(b) in conflict_nodes,
            )
            for value, a, b in edges
        ),
        key=lambda edge: edge.pair,
    )
    return Prematch(
        LOW_THRESHOLD,
        accept_at,
        tuple(nodes),
        tuple(records),
        match_counts,
        tuple(fallback),
        conflict_components,
        item=item,
        label=label,
        text=text,
    )


def resolve_units(
    nodes: Sequence[Member], kept: Sequence[WeightedEdge], names: Sequence[str]
) -> tuple[list[list[Member]], list[tuple[str, float]]]:
    """Components of the kept edges once every component is a clique with distinct members.

    While a component violates this, the weakest edge inside a violating component drops, by the
    key (similarity, sorted content keys of its ends); every edge tying on the exact key drops
    with it. Every final component is returned, singletons included.
    """
    edges = list(kept)
    dropped: list[tuple[str, float]] = []
    while True:
        present = _edge_set(nodes, edges)
        components = _components(nodes, edges)
        bad = [c for c in components if len(c) > 1 and not _is_unit(c, present, nodes)]
        if not bad:
            break
        bad_nodes = {id(member) for component in bad for member in component}
        removable = [edge for edge in edges if id(edge[1]) in bad_nodes and id(edge[2]) in bad_nodes]

        def drop_key(edge: WeightedEdge) -> tuple[Any, ...]:
            low, high = sorted((relation_key(edge[1].relation, names), relation_key(edge[2].relation, names)))
            return (edge[0], low, high)

        weakest = min(drop_key(edge) for edge in removable)
        for edge in [e for e in removable if drop_key(e) == weakest]:
            edges.remove(edge)
            dropped.append((pair_label(edge[1], edge[2]), edge[0]))
    return _components(nodes, edges), dropped


def align(
    members: Mapping[str, Sequence[Relation]],
    sim: Similarity,
    referee: Referee,
    *,
    coders: Mapping[str, str] | None = None,
    item: ItemId | None = None,
    label: str | None = None,
    text: str = "",
) -> Alignment:
    """Align the members' relations into units under the referee's decisions.

    Every unit is returned, singletons included, each with its support, its distinct coders and
    its medoid. The referee must decide every candidate pair exactly; the decisions are recorded.
    """
    pre = prematch(members, sim, accept_at=referee.accept_at, coders=coders, item=item, label=label, text=text)
    by_member = _candidates(members, coders)
    nodes, edges, _, _ = _edge_graph(by_member, sim)
    decisions = dict(referee.decide(pre))
    if set(decisions) != set(pre.candidate_pairs):
        raise ValueError("the referee must decide every candidate pair exactly once and nothing else")
    by_pair = {pair_label(a, b): (value, a, b) for value, a, b in edges}
    kept = [by_pair[pair] for pair in sorted(pair for pair, confirmed in decisions.items() if confirmed)]
    components, dropped = resolve_units(nodes, kept, sim.names)
    units = tuple(Unit(tuple(component), medoid(component, sim)) for component in components)
    return Alignment(units, pre, decisions, tuple(dropped))


__all__ = ["LOW_THRESHOLD", "Alignment", "Band", "Edge", "Prematch", "align", "node_label", "pair_label", "prematch"]
