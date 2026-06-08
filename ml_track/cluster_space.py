"""
Cluster perfumes in descriptor-space and generate latent-space artifacts.

Usage:
    python -m ml_track.cluster_space --clusters 14
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .features import OUT_DIR, build_and_save


try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None


def ensure_artifacts(out_dir: Path) -> tuple[pd.DataFrame, sparse.csr_matrix, list[str]]:
    matrix_path = out_dir / "perfume_features.npz"
    index_path = out_dir / "perfume_index.csv"
    names_path = out_dir / "feature_names.json"
    if not matrix_path.exists() or not index_path.exists() or not names_path.exists():
        build_and_save(out_dir=out_dir)
    index = pd.read_csv(index_path)
    matrix = sparse.load_npz(matrix_path).tocsr()
    with names_path.open("r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return index, matrix, feature_names


def reduce_vectors(matrix: sparse.csr_matrix, components: int, seed: int) -> np.ndarray:
    n_components = min(components, matrix.shape[0] - 1, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    reduced = svd.fit_transform(matrix)
    return StandardScaler().fit_transform(reduced)


def make_coordinates(vectors: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    if umap is not None:
        reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.08, random_state=seed)
        coords = reducer.fit_transform(vectors)
        return coords, "umap"

    coords = np.zeros((vectors.shape[0], 3))
    n = min(3, vectors.shape[1])
    coords[:, :n] = vectors[:, :n]
    return coords, "svd"


def cluster_vectors(vectors: np.ndarray, clusters: int, seed: int) -> tuple[np.ndarray, str]:
    if hdbscan is not None:
        model = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10)
        return model.fit_predict(vectors), "hdbscan"

    model = KMeans(n_clusters=clusters, n_init=20, random_state=seed)
    return model.fit_predict(vectors), "kmeans"


def metric_summary(vectors: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> dict[str, float | int | None]:
    valid_mask = labels >= 0
    unique = sorted(set(labels[valid_mask]))
    summary: dict[str, float | int | None] = {
        "clusters": len(unique),
        "noise_points": int((labels < 0).sum()),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if len(unique) < 2:
        return summary

    rng = np.random.default_rng(seed)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) > sample_size:
        valid_indices = rng.choice(valid_indices, size=sample_size, replace=False)
    sample_vectors = vectors[valid_indices]
    sample_labels = labels[valid_indices]

    summary["silhouette"] = float(silhouette_score(sample_vectors, sample_labels))
    summary["davies_bouldin"] = float(davies_bouldin_score(sample_vectors, sample_labels))
    summary["calinski_harabasz"] = float(calinski_harabasz_score(sample_vectors, sample_labels))
    return summary


def top_cluster_features(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    feature_names: list[str],
    top_n: int = 12,
) -> dict[int, list[str]]:
    output = {}
    for label in sorted(set(labels)):
        mask = labels == label
        if label < 0 or not mask.any():
            continue
        mean_vec = np.asarray(matrix[mask].mean(axis=0)).ravel()
        top_indices = mean_vec.argsort()[-top_n:][::-1]
        output[int(label)] = [feature_names[i] for i in top_indices if mean_vec[i] > 0]
    return output


def write_plot(clusters_df: pd.DataFrame, out_dir: Path, coordinate_method: str) -> None:
    if px is None:
        return
    hover = ["perfume", "brand", "gender", "rating", "votes"]
    fig2 = px.scatter(
        clusters_df,
        x="x",
        y="y",
        color="cluster",
        hover_data=hover,
        title=f"AromaLatent latent scent space ({coordinate_method})",
    )
    fig2.write_html(out_dir / "latent_space_2d.html")

    fig3 = px.scatter_3d(
        clusters_df,
        x="x",
        y="y",
        z="z",
        color="cluster",
        hover_data=hover,
        title=f"AromaLatent latent scent space 3D ({coordinate_method})",
    )
    fig3.write_html(out_dir / "latent_space_3d.html")


def run(out_dir: Path, clusters: int, svd_components: int, sample_size: int, seed: int) -> tuple[pd.DataFrame, str]:
    index, matrix, feature_names = ensure_artifacts(out_dir)
    vectors = reduce_vectors(matrix, components=svd_components, seed=seed)
    coords, coordinate_method = make_coordinates(vectors, seed=seed)
    labels, cluster_method = cluster_vectors(vectors, clusters=clusters, seed=seed)

    clusters_df = index[["perfume", "brand", "url", "gender", "rating", "votes", "year"]].copy()
    clusters_df["cluster"] = labels.astype(str)
    clusters_df["x"] = coords[:, 0]
    clusters_df["y"] = coords[:, 1]
    clusters_df["z"] = coords[:, 2]
    clusters_df.to_csv(out_dir / "clusters.csv", index=False, encoding="utf-8")
    write_plot(clusters_df, out_dir, coordinate_method)

    metrics = metric_summary(vectors, labels, sample_size=sample_size, seed=seed)
    feature_summary = top_cluster_features(matrix, labels, feature_names)

    report_lines = [
        "# AromaLatent ML Cluster Report",
        "",
        f"- Rows: {len(index)}",
        f"- Feature matrix: {matrix.shape[0]} x {matrix.shape[1]}",
        f"- Vector reducer: TruncatedSVD to {min(svd_components, matrix.shape[1] - 1)} dimensions",
        f"- Coordinate method: {coordinate_method}",
        f"- Cluster method: {cluster_method}",
        f"- Clusters: {metrics['clusters']}",
        f"- Noise points: {metrics['noise_points']}",
        f"- Silhouette: {metrics['silhouette']}",
        f"- Davies-Bouldin: {metrics['davies_bouldin']}",
        f"- Calinski-Harabasz: {metrics['calinski_harabasz']}",
        "",
        "## Top Cluster Features",
        "",
    ]
    for label, terms in feature_summary.items():
        size = int((labels == label).sum())
        report_lines.append(f"### Cluster {label} ({size} perfumes)")
        report_lines.append(", ".join(terms))
        report_lines.append("")
    report = "\n".join(report_lines)
    (out_dir / "cluster_report.md").write_text(report, encoding="utf-8")
    return clusters_df, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--clusters", type=int, default=14)
    parser.add_argument("--svd-components", type=int, default=50)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    clusters_df, _ = run(
        out_dir=args.out,
        clusters=args.clusters,
        svd_components=args.svd_components,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Wrote {len(clusters_df)} rows to {args.out / 'clusters.csv'}")
    print(f"Wrote report to {args.out / 'cluster_report.md'}")


if __name__ == "__main__":
    main()

