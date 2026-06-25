from __future__ import annotations
from collections import deque
from typing import Dict, List, Tuple


def build_directed_links(topology: str, nodes: List[str]) -> List[Tuple[str, str]]:
    topo = (topology or "mesh").lower()
    links: List[Tuple[str, str]] = []
    if len(nodes) <= 1:
        return links
    if topo == "full":
        for a in nodes:
            for b in nodes:
                if a != b:
                    links.append((a, b))
        return links
    if topo == "ring":
        for i, a in enumerate(nodes):
            b = nodes[(i + 1) % len(nodes)]
            links.append((a, b))
            links.append((b, a))
        return links
    # mesh fallback
    w = max(1, int(len(nodes) ** 0.5))
    while w * w < len(nodes):
        w += 1
    coords: Dict[str, Tuple[int, int]] = {}
    by_coord: Dict[Tuple[int, int], str] = {}
    for idx, n in enumerate(nodes):
        c = (idx % w, idx // w)
        coords[n] = c
        by_coord[c] = n
    for n, (x, y) in coords.items():
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            m = by_coord.get((nx, ny))
            if m is not None:
                links.append((n, m))
    return links


def xy_path(nodes: List[str], src: str, dst: str) -> List[str]:
    if src == dst:
        return [src]
    w = max(1, int(len(nodes) ** 0.5))
    while w * w < len(nodes):
        w += 1
    coords = {n: (i % w, i // w) for i, n in enumerate(nodes)}
    inv = {v: k for k, v in coords.items()}
    x, y = coords[src]
    tx, ty = coords[dst]
    path = [src]
    while x != tx:
        x += 1 if tx > x else -1
        nxt = inv.get((x, y))
        if nxt is None:
            break
        path.append(nxt)
    while y != ty:
        y += 1 if ty > y else -1
        nxt = inv.get((x, y))
        if nxt is None:
            break
        path.append(nxt)
    if path[-1] != dst:
        path.append(dst)
    return path


def shortest_path(topology: str, nodes: List[str], src: str, dst: str) -> List[str]:
    topo = (topology or "mesh").lower()
    if src == dst:
        return [src]
    if topo == "mesh":
        return xy_path(nodes, src, dst)
    links = build_directed_links(topo, nodes)
    graph: Dict[str, List[str]] = {}
    for a, b in links:
        graph.setdefault(a, []).append(b)
    q = deque([[src]])
    seen = {src}
    while q:
        path = q.popleft()
        u = path[-1]
        for v in graph.get(u, []):
            if v == dst:
                return path + [v]
            if v not in seen:
                seen.add(v)
                q.append(path + [v])
    return [src, dst]


def path_to_links(path: List[str]) -> List[Tuple[str, str]]:
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]
