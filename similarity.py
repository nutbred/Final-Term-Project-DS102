"""
similarity.py — ABSA-RRF fragrance similarity module

Formula:
  base_sim = 0.40 * text_sim + 0.35 * note_sim + 0.25 * accord_sim
  final_sim = base_sim + dupe_bonus * (1.0 - base_sim)

dupe_bonus:
  0.30 — A is direct dupe of B (or vice versa)
  0.20 — A and B are both dupes of same OG
  0.00 — no known structural relationship
"""
import json, math, sqlite3
from uuid import uuid4
from datetime import datetime

DB_PATH = None  # set by caller via init()

ALL_SEASONS   = ["spring","summer","fall","winter"]
ALL_TIMES     = ["morning","midday","afternoon","evening","night"]
ALL_OCCASIONS = ["office","date","casual","clubbing","outdoor","formal"]
ALL_ACCORDS   = [
    "aromatic","woody","fresh","citrus","floral","sweet","amber","spicy",
    "fresh spicy","warm spicy","soft spicy","musky","powdery","green",
    "aquatic","leather","vanilla","oud","smoky","earthy","balsamic","fruity",
]
KNOWN_VIBES = [
    "fresh","clean","masculine","feminine","office-safe","date-night","casual",
    "sporty","sweet","smoky","woody","earthy","citrus","floral","powdery",
    "warm","cold","heavy","light","long-lasting","beast-mode","crowd-pleaser",
    "unique","safe","polarizing","versatile","unisex","bold","subtle",
]


def init(db_path: str):
    global DB_PATH
    DB_PATH = db_path
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fragrance_similarity (
            id TEXT PRIMARY KEY,
            frag_a TEXT NOT NULL,
            frag_b TEXT NOT NULL,
            sim_score REAL NOT NULL,
            text_sim  REAL,
            note_sim  REAL,
            accord_sim REAL,
            dupe_bonus REAL,
            method TEXT DEFAULT 'absa_rrf',
            og_anchor TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# ── Feature vector ────────────────────────────────────────────────────────────

def build_vector(profile: dict, accords: list) -> dict:
    vec = {}
    vec["longevity"] = (profile.get("longevity_score") or 5) / 10.0
    vec["sillage"]   = (profile.get("sillage_score")   or 5) / 10.0

    seasons = [s.lower() for s in (profile.get("season") or [])]
    for s in ALL_SEASONS:
        vec[f"s_{s}"] = 1.0 if s in seasons else 0.0

    times = [t.lower() for t in (profile.get("time_of_day") or [])]
    for t in ALL_TIMES:
        vec[f"t_{t}"] = 1.0 if t in times else 0.0

    occ = [o.lower() for o in (profile.get("occasion") or [])]
    for o in ALL_OCCASIONS:
        vec[f"o_{o}"] = 1.0 if o in occ else 0.0

    vibes = [v.lower() for v in (profile.get("vibes") or [])]
    for v in KNOWN_VIBES:
        vec[f"v_{v}"] = 1.0 if any(v in vibe or vibe in v for vibe in vibes) else 0.0

    acc_lower = [a.lower() for a in accords if isinstance(a, str)]
    for a in ALL_ACCORDS:
        vec[f"a_{a.replace(' ','_')}"] = 1.0 if a in acc_lower else 0.0

    return vec


def cosine(va: dict, vb: dict) -> float:
    keys = set(va) | set(vb)
    dot  = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na   = math.sqrt(sum(va.get(k, 0) ** 2 for k in keys))
    nb   = math.sqrt(sum(vb.get(k, 0) ** 2 for k in keys))
    return dot / (na * nb + 1e-9)


def jaccard(set_a: list, set_b: list) -> float:
    a = {x.lower().strip() for x in set_a if isinstance(x, str) and x}
    b = {x.lower().strip() for x in set_b if isinstance(x, str) and x}
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Dupe graph helpers ────────────────────────────────────────────────────────

def _get_dupes_col(name: str) -> list:
    """Return the `dupes` JSON array stored for a fragrance."""
    conn = _conn()
    row = conn.execute(
        "SELECT dupes FROM perfumes WHERE LOWER(name)=?", (name.lower(),)
    ).fetchone()
    conn.close()
    if not row:
        return []
    return json.loads(row["dupes"] or "[]")


def _get_mentioned_dupe_of(name: str) -> str | None:
    """Return `mentioned_dupe_of` extracted from the first review's llm_profile."""
    conn = _conn()
    row = conn.execute(
        "SELECT reviews FROM perfumes WHERE LOWER(name)=?", (name.lower(),)
    ).fetchone()
    conn.close()
    if not row:
        return None
    reviews = json.loads(row["reviews"] or "[]")
    for r in reviews:
        prof = r.get("llm_profile", {}) or {}
        dupe_of = prof.get("mentioned_dupe_of")
        if dupe_of:
            return dupe_of.strip()
    return None


def dupe_bonus(name_a: str, name_b: str) -> float:
    """
    0.30 — direct dupe relationship (A is dupe of B or vice versa)
    0.20 — both are dupes of same OG
    0.00 — no structural relationship
    """
    a_l = name_a.lower()
    b_l = name_b.lower()

    # Check A→B
    dupes_a = [d.lower() for d in _get_dupes_col(name_a) if isinstance(d, str)]
    if any(b_l in d or d in b_l for d in dupes_a):
        return 0.30

    # Check B→A
    dupes_b = [d.lower() for d in _get_dupes_col(name_b) if isinstance(d, str)]
    if any(a_l in d or d in a_l for d in dupes_b):
        return 0.30

    # Check mentioned_dupe_of
    og_a = _get_mentioned_dupe_of(name_a)
    og_b = _get_mentioned_dupe_of(name_b)
    if og_a and og_b and og_a.lower() == og_b.lower():
        return 0.20

    # Check if one is the OG of the other via mentioned_dupe_of
    if og_a and og_a.lower() == b_l:
        return 0.30
    if og_b and og_b.lower() == a_l:
        return 0.30

    return 0.00


# ── Main scoring ──────────────────────────────────────────────────────────────

def _load_frag(name: str) -> dict | None:
    conn = _conn()
    row = conn.execute(
        "SELECT name, accords, notes_top, notes_middle, notes_base, reviews "
        "FROM perfumes WHERE LOWER(name)=?",
        (name.lower(),)
    ).fetchone()
    conn.close()
    if not row:
        return None
    reviews = json.loads(row["reviews"] or "[]")
    profile = {}
    for r in reviews:
        p = r.get("llm_profile", {})
        if p:
            profile = p
            break
    def _flat(raw):
        lst = json.loads(raw or "[]")
        return [x for x in lst if isinstance(x, str)]

    return {
        "name":    row["name"],
        "accords": _flat(row["accords"]),
        "notes":   _flat(row["notes_top"]) + _flat(row["notes_middle"]) + _flat(row["notes_base"]),
        "profile": profile,
    }


def compute(name_a: str, name_b: str, og_anchor: str = None) -> dict | None:
    fa = _load_frag(name_a)
    fb = _load_frag(name_b)
    if not fa or not fb:
        return None

    va = build_vector(fa["profile"], fa["accords"])
    vb = build_vector(fb["profile"], fb["accords"])

    text_s  = round(cosine(va, vb), 4)
    note_s  = round(jaccard(fa["notes"], fb["notes"]), 4)
    accord_s = round(jaccard(fa["accords"], fb["accords"]), 4)
    bonus   = round(dupe_bonus(name_a, name_b), 2)

    base = 0.40 * text_s + 0.35 * note_s + 0.25 * accord_s
    final = round(base + bonus * (1.0 - base), 4)

    # Canonical ordering: always store (lower-alpha, higher-alpha) so (A,B)==(B,A)
    name_lo = min(fa["name"], fb["name"], key=str.lower)
    name_hi = max(fa["name"], fb["name"], key=str.lower)
    pair_id  = f"{name_lo.lower()}|||{name_hi.lower()}"   # deterministic PK

    result = {
        "frag_a": name_lo, "frag_b": name_hi,
        "sim_score": final,
        "text_sim": text_s, "note_sim": note_s,
        "accord_sim": accord_s, "dupe_bonus": bonus,
        "og_anchor": og_anchor,
    }

    # Persist — INSERT OR REPLACE deduplicates because pair_id is deterministic
    conn = _conn()
    conn.execute("""
        INSERT OR REPLACE INTO fragrance_similarity
        (id, frag_a, frag_b, sim_score, text_sim, note_sim, accord_sim,
         dupe_bonus, method, og_anchor, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        pair_id, name_lo, name_hi, final,
        text_s, note_s, accord_s, bonus, "absa_rrf",
        og_anchor, datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    return result


def compute_all_pairs(names: list, og_anchor: str = None) -> list:
    """Compute pairwise similarity for a list of fragrances."""
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = compute(names[i], names[j], og_anchor)
            if r:
                results.append(r)
    return results


def knn(name: str, k: int = 5) -> list:
    """Return top-k most similar fragrances from the similarity table."""
    conn = _conn()
    rows = conn.execute("""
        SELECT frag_a, frag_b, sim_score, text_sim, note_sim, accord_sim, dupe_bonus
        FROM fragrance_similarity
        WHERE LOWER(frag_a)=? OR LOWER(frag_b)=?
        ORDER BY sim_score DESC
        LIMIT ?
    """, (name.lower(), name.lower(), k * 2)).fetchall()
    conn.close()

    seen = set()
    results = []
    for r in rows:
        neighbor = r["frag_b"] if r["frag_a"].lower() == name.lower() else r["frag_a"]
        if neighbor.lower() not in seen:
            seen.add(neighbor.lower())
            results.append({
                "name": neighbor,
                "sim_score": r["sim_score"],
                "text_sim": r["text_sim"],
                "note_sim": r["note_sim"],
                "accord_sim": r["accord_sim"],
                "dupe_bonus": r["dupe_bonus"],
            })
        if len(results) >= k:
            break
    return results
