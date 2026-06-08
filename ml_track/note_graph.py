"""
Build a note co-occurrence graph and centrality report.

Usage:
    python -m ml_track.note_graph
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from .features import CSV_PATH, OUT_DIR, load_perfumes


def build_graph(df: pd.DataFrame, min_note_count: int, min_pair_count: int) -> nx.Graph:
    note_counts: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    total = len(df)

    for notes in df["all_notes"]:
        unique = sorted(set(notes))
        for note in unique:
            note_counts[note] = note_counts.get(note, 0) + 1
        for a, b in itertools.combinations(unique, 2):
            pair = (a, b)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    graph = nx.Graph()
    for note, count in note_counts.items():
        if count >= min_note_count:
            graph.add_node(note, count=count, prevalence=count / total)

    for (a, b), count in pair_counts.items():
        if count < min_pair_count or a not in graph or b not in graph:
            continue
        expected = (note_counts[a] * note_counts[b]) / total
        lift = count / expected if expected else 0.0
        pmi = math.log(lift) if lift > 0 else 0.0
        if pmi <= 0:
            continue
        graph.add_edge(a, b, count=count, lift=lift, pmi=pmi, weight=pmi)

    return graph


def centrality_frame(graph: nx.Graph) -> pd.DataFrame:
    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {}
    try:
        eigen = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)
    except Exception:
        eigen = {node: 0.0 for node in graph.nodes}
    if graph.number_of_nodes() > 2:
        k = min(250, graph.number_of_nodes())
        between = nx.betweenness_centrality(graph, k=k, seed=42, weight=None)
    else:
        between = {node: 0.0 for node in graph.nodes}

    rows = []
    for node, attrs in graph.nodes(data=True):
        weighted_degree = sum(data.get("weight", 1.0) for _, _, data in graph.edges(node, data=True))
        rows.append({
            "note": node,
            "count": attrs.get("count", 0),
            "prevalence": attrs.get("prevalence", 0.0),
            "degree": graph.degree(node),
            "weighted_degree": weighted_degree,
            "pagerank": pagerank.get(node, 0.0),
            "eigenvector": eigen.get(node, 0.0),
            "betweenness": between.get(node, 0.0),
        })
    return pd.DataFrame(rows).sort_values(["pagerank", "weighted_degree"], ascending=False)


def community_frame(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_edges() == 0:
        return pd.DataFrame(columns=["community", "note", "count", "degree"])

    communities = nx.community.greedy_modularity_communities(graph, weight="weight")
    rows = []
    for idx, community in enumerate(communities):
        for note in sorted(community):
            rows.append({
                "community": idx,
                "note": note,
                "count": graph.nodes[note].get("count", 0),
                "degree": graph.degree(note),
            })
    return pd.DataFrame(rows).sort_values(["community", "count"], ascending=[True, False])


def write_report(graph: nx.Graph, centrality: pd.DataFrame, communities: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# AromaLatent Note Graph Report",
        "",
        f"- Nodes: {graph.number_of_nodes()}",
        f"- Edges: {graph.number_of_edges()}",
        "",
        "## Top PageRank Notes",
        "",
    ]
    for _, row in centrality.head(20).iterrows():
        lines.append(
            f"- {row['note']}: pagerank={row['pagerank']:.5f}, "
            f"count={int(row['count'])}, degree={int(row['degree'])}"
        )

    lines.extend(["", "## Largest Communities", ""])
    if not communities.empty:
        sizes = communities.groupby("community").size().sort_values(ascending=False).head(10)
        for community, size in sizes.items():
            notes = communities[communities["community"] == community].head(12)["note"].tolist()
            lines.append(f"- Community {community} ({size} notes): {', '.join(notes)}")

    (out_dir / "note_graph_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(csv_path: Path, out_dir: Path, min_note_count: int, min_pair_count: int) -> nx.Graph:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_perfumes(csv_path)
    graph = build_graph(df, min_note_count=min_note_count, min_pair_count=min_pair_count)
    centrality = centrality_frame(graph)
    communities = community_frame(graph)

    centrality.to_csv(out_dir / "note_centrality.csv", index=False, encoding="utf-8")
    communities.to_csv(out_dir / "note_communities.csv", index=False, encoding="utf-8")
    nx.write_graphml(graph, out_dir / "note_graph.graphml")
    write_report(graph, centrality, communities, out_dir)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--min-note-count", type=int, default=25)
    parser.add_argument("--min-pair-count", type=int, default=8)
    args = parser.parse_args()

    graph = run(
        csv_path=args.csv,
        out_dir=args.out,
        min_note_count=args.min_note_count,
        min_pair_count=args.min_pair_count,
    )
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()

