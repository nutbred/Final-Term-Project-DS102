"""
Find descriptor-space perfume alternatives from weighted note/accord vectors.

Usage:
    python -m ml_track.dupe_detector --top-k 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .features import OUT_DIR, build_and_save


def parse_json_list(raw: Any) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except Exception:
        return []
    return [str(v).lower().strip() for v in value if isinstance(v, str) and v.strip()]


def weighted_note_map(row: pd.Series) -> dict[str, float]:
    weights = {}
    for note in parse_json_list(row["notes_top"]):
        weights[note] = weights.get(note, 0.0) + 1.0
    for note in parse_json_list(row["notes_middle"]):
        weights[note] = weights.get(note, 0.0) + 1.25
    for note in parse_json_list(row["notes_base"]):
        weights[note] = weights.get(note, 0.0) + 1.5
    return weights


def weighted_jaccard(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    numerator = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    denominator = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return numerator / denominator if denominator else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def accord_rank_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    wa = {accord: 1.0 / (rank + 1) for rank, accord in enumerate(a)}
    wb = {accord: 1.0 / (rank + 1) for rank, accord in enumerate(b)}
    keys = set(wa) | set(wb)
    numerator = sum(min(wa.get(k, 0.0), wb.get(k, 0.0)) for k in keys)
    denominator = sum(max(wa.get(k, 0.0), wb.get(k, 0.0)) for k in keys)
    return numerator / denominator if denominator else 0.0


def ensure_artifacts(out_dir: Path) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    matrix_path = out_dir / "perfume_features.npz"
    index_path = out_dir / "perfume_index.csv"
    if not matrix_path.exists() or not index_path.exists():
        build_and_save(out_dir=out_dir)
    index = pd.read_csv(index_path)
    matrix = sparse.load_npz(matrix_path).tocsr()
    return index, matrix


def find_dupes(out_dir: Path, top_k: int, limit: int | None = None) -> pd.DataFrame:
    index, matrix = ensure_artifacts(out_dir)
    query_count = len(index) if limit is None else min(limit, len(index))
    neighbors = NearestNeighbors(n_neighbors=top_k + 1, metric="cosine", algorithm="brute")
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix[:query_count])

    note_maps = [weighted_note_map(row) for _, row in index.iterrows()]
    base_sets = [set(parse_json_list(row["notes_base"])) for _, row in index.iterrows()]
    accord_lists = [parse_json_list(row["accords"]) for _, row in index.iterrows()]

    records = []
    for query_idx in range(query_count):
        query = index.iloc[query_idx]
        for distance, cand_idx in zip(distances[query_idx], indices[query_idx]):
            if cand_idx == query_idx:
                continue
            candidate = index.iloc[cand_idx]
            tfidf_cosine = max(0.0, 1.0 - float(distance))
            note_overlap = weighted_jaccard(note_maps[query_idx], note_maps[cand_idx])
            base_overlap = jaccard(base_sets[query_idx], base_sets[cand_idx])
            accord_overlap = accord_rank_overlap(accord_lists[query_idx], accord_lists[cand_idx])
            dupe_score = (
                0.45 * tfidf_cosine
                + 0.25 * note_overlap
                + 0.20 * accord_overlap
                + 0.10 * base_overlap
            )
            records.append({
                "query_perfume": query["perfume"],
                "query_brand": query["brand"],
                "candidate_perfume": candidate["perfume"],
                "candidate_brand": candidate["brand"],
                "dupe_score": round(dupe_score, 5),
                "tfidf_cosine": round(tfidf_cosine, 5),
                "weighted_note_jaccard": round(note_overlap, 5),
                "accord_rank_overlap": round(accord_overlap, 5),
                "base_note_jaccard": round(base_overlap, 5),
                "query_url": query["url"],
                "candidate_url": candidate["url"],
            })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(["query_perfume", "dupe_score"], ascending=[True, False])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Limit query rows for quick tests.")
    parser.add_argument("--output-name", default=None, help="Override output CSV filename.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    result = find_dupes(args.out, top_k=args.top_k, limit=args.limit)
    output_name = args.output_name
    if output_name is None:
        output_name = "dupe_candidates_sample.csv" if args.limit else "dupe_candidates.csv"
    path = args.out / output_name
    result.to_csv(path, index=False, encoding="utf-8")
    print(f"Wrote {len(result)} rows to {path}")


if __name__ == "__main__":
    main()
