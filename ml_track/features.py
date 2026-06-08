"""
Build weighted note/accord feature matrices for the AromaLatent ML track.

Usage:
    python -m ml_track.features
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "fragrantica_dataset" / "fra_cleaned.csv"
OUT_DIR = ROOT / "output" / "ml"

TIER_WEIGHTS = {
    "top": 1.0,
    "middle": 1.25,
    "base": 1.5,
}
ACCORD_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]


def clean_token(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text or text == "nan":
        return ""
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_tokens(raw: object) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    tokens = []
    for part in str(raw).split(","):
        token = clean_token(part)
        if token:
            tokens.append(token)
    return tokens


def parse_decimal(raw: object) -> float | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip().replace(",", ".")
    if not text or text.lower() == "nan":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(raw: object) -> int | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = re.sub(r"[^\d]", "", str(raw))
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def load_perfumes(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="latin-1")
    out = pd.DataFrame()
    out["perfume"] = df["Perfume"].fillna("").astype(str)
    out["brand"] = df["Brand"].fillna("").astype(str)
    out["url"] = df["url"].fillna("").astype(str)
    out["country"] = df.get("Country", "").fillna("").astype(str)
    out["gender"] = df.get("Gender", "").fillna("").astype(str)
    out["year"] = df.get("Year", "").map(parse_int)
    out["rating"] = df.get("Rating Value", "").map(parse_decimal)
    out["votes"] = df.get("Rating Count", "").map(parse_int)
    out["notes_top"] = df["Top"].map(split_tokens)
    out["notes_middle"] = df["Middle"].map(split_tokens)
    out["notes_base"] = df["Base"].map(split_tokens)
    out["accords"] = df[[f"mainaccord{i}" for i in range(1, 6)]].apply(
        lambda row: [clean_token(v) for v in row if clean_token(v)],
        axis=1,
    )
    out["all_notes"] = out.apply(
        lambda row: sorted(set(row["notes_top"] + row["notes_middle"] + row["notes_base"])),
        axis=1,
    )
    out["perfume_key"] = out.apply(
        lambda row: f"{clean_token(row['brand'])}::{clean_token(row['perfume'])}::{row['url']}",
        axis=1,
    )
    return out


def row_feature_dict(row: pd.Series) -> dict[str, float]:
    features: defaultdict[str, float] = defaultdict(float)
    for tier, col in (
        ("top", "notes_top"),
        ("middle", "notes_middle"),
        ("base", "notes_base"),
    ):
        weight = TIER_WEIGHTS[tier]
        for note in row[col]:
            features[f"note:{note}"] += weight
            features[f"{tier}:{note}"] += weight * 0.35

    for idx, accord in enumerate(row["accords"][:5]):
        features[f"accord:{accord}"] += ACCORD_WEIGHTS[idx]

    return dict(features)


def build_feature_matrix(df: pd.DataFrame, use_tfidf: bool = True) -> tuple[sparse.csr_matrix, list[str]]:
    vectorizer = DictVectorizer(sparse=True)
    raw = vectorizer.fit_transform(row_feature_dict(row) for _, row in df.iterrows())
    if use_tfidf:
        matrix = TfidfTransformer(norm="l2", sublinear_tf=True).fit_transform(raw)
    else:
        matrix = raw.tocsr()
    feature_names = [str(name) for name in vectorizer.get_feature_names_out()]
    return matrix.tocsr(), feature_names


def json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=True)


def write_artifacts(
    df: pd.DataFrame,
    matrix: sparse.csr_matrix,
    feature_names: list[str],
    out_dir: Path = OUT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "perfume_features.npz", matrix)
    with (out_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    index = df.copy()
    for col in ["notes_top", "notes_middle", "notes_base", "accords", "all_notes"]:
        index[col] = index[col].map(json_list)
    index.to_csv(out_dir / "perfume_index.csv", index=False, encoding="utf-8")


def build_and_save(
    csv_path: Path = CSV_PATH,
    out_dir: Path = OUT_DIR,
    use_tfidf: bool = True,
) -> tuple[pd.DataFrame, sparse.csr_matrix, list[str]]:
    df = load_perfumes(csv_path)
    matrix, feature_names = build_feature_matrix(df, use_tfidf=use_tfidf)
    write_artifacts(df, matrix, feature_names, out_dir=out_dir)
    return df, matrix, feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--raw-counts", action="store_true", help="Skip TF-IDF weighting.")
    args = parser.parse_args()

    df, matrix, feature_names = build_and_save(
        csv_path=args.csv,
        out_dir=args.out,
        use_tfidf=not args.raw_counts,
    )
    print(f"Rows: {len(df)}")
    print(f"Matrix: {matrix.shape[0]} x {matrix.shape[1]} ({matrix.nnz} non-zero)")
    print(f"Features: {len(feature_names)}")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()

