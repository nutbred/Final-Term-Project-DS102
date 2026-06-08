"""
pipeline_v2.py — Seed-based review pipeline + similarity scoring

Flow:
  1. Auto-select 15 popular seeds from fra_cleaned.csv
  2. Fetch Reddit reviews (exact name, no augmentation)
  3. LLM: reviews → profile JSON (includes mentioned_dupe_of)
  4. Depth-1 cascade: if mentioned_dupe_of → queue & process dupe
  5. Similarity pass: cosine similarity within each OG chain
"""
import sys, os, re, json, time, sqlite3, math, requests
from uuid import uuid4
from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import similarity as sim_module

sys.stdout.reconfigure(encoding="utf-8")

HERE     = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(HERE, "github_project_references", "PerfumAPI", "perfumes.db")
CSV_PATH = os.path.join(HERE, "fragrantica_dataset", "fra_cleaned.csv")
ENV_PATH = os.path.join(HERE, "llm", ".env")
OUT_DIR  = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

load_dotenv(ENV_PATH)
llm = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

HEADERS   = {"User-Agent": "AromaLatentBot/1.0 (academic; student@uit.edu.vn)"}
SUBREDDIT = "fragrance"
N_SEEDS   = 655

# ── Curated known dupe chains (verified by community) ────────────────────────
# Format: anchor_name -> [list of known dupe names]
# These are processed with dupe_of context so LLM can confirm properly.
KNOWN_CHAINS = {
    "Creed Aventus": [
        "Club De Nuit Intense Man",   # armaf — slug: club-de-nuit-intense-man
        "Explorer",                    # montblanc
        "L Aventure",                  # al-haramain-perfumes
        "Asad",                        # lattafa-perfumes
    ],
    "YSL Y EDP": [
        "Yeah",                        # maison-alhambra
        "Odyssey Mega Man",            # armaf
    ],
    "Dior Sauvage": [
        "Afnan 9PM",                   # afnan
        "Sauvage Elixir",              # dior (higher concentration reference)
    ],
    "Baccarat Rouge 540": [
        "Bade E Al Oud",              # lattafa-perfumes
        "Citrus 08",                   # kayali-fragrances
    ],
}

# Anchor lookup names for CSV (exactly as they appear in fra_cleaned.csv)
ANCHOR_CSV_LOOKUP = {
    "Creed Aventus": ("aventus", "creed"),
    "YSL Y EDP": ("y eau de parfum", "yves-saint-laurent"),
    "Dior Sauvage": ("sauvage", "dior"),
    "Baccarat Rouge 540": ("baccarat rouge 540", ""),
}

# ── Popular designer brands to constrain seed selection ──────────────────────
TARGET_BRANDS = {
    "dior", "chanel", "tom-ford", "yves-saint-laurent", "creed",
    "versace", "armani", "paco-rabanne", "dolce-gabbana", "givenchy",
    "burberry", "gucci", "viktor-rolf", "maison-margiela", "thierry-mugler",
    "lancome", "hermes", "cartier", "bvlgari", "mont-blanc",
    "calvin-klein", "hugo-boss", "lacoste", "ralph-lauren",
}

# ── CSV ───────────────────────────────────────────────────────────────────────
_df = None

def load_csv() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(CSV_PATH, sep=";", encoding="latin-1")
        # Normalize votes to numeric
        _df["_votes"] = pd.to_numeric(_df["Rating Count"], errors="coerce").fillna(0)
    return _df

def select_seeds(n: int = N_SEEDS) -> list:
    """Pick top-n by vote count from target designer brands (min 1000 votes)."""
    df = load_csv()
    mask = df["Brand"].str.lower().isin(TARGET_BRANDS) & (df["_votes"] >= 1000)
    top = df[mask].nlargest(n, "_votes")
    seeds = []
    for _, r in top.iterrows():
        seeds.append({
            "display_name": r["Perfume"].replace("-", " ").title(),
            "brand":        r["Brand"].replace("-", " ").title(),
            "url":          r["url"],
            "notes_top":    [n.strip() for n in str(r.get("Top","")).split(",") if n.strip() and n.strip()!="nan"],
            "notes_middle": [n.strip() for n in str(r.get("Middle","")).split(",") if n.strip() and n.strip()!="nan"],
            "notes_base":   [n.strip() for n in str(r.get("Base","")).split(",") if n.strip() and n.strip()!="nan"],
            "accords":      [str(r.get(f"mainaccord{i}","")) for i in range(1,6) if str(r.get(f"mainaccord{i}","")) not in ("","nan")],
            "rating":       str(r.get("Rating Value","")),
            "votes":        str(r.get("Rating Count","")),
            "release_year": str(r.get("Year","")),
            "gender":       str(r.get("Gender","")),
        })
    return seeds

def csv_lookup(name: str, brand: str = "") -> dict:
    """
    Search across combined 'perfume_slug + brand_slug' with hyphen-stripped normalization.
    Handles aliases like:
      'Armaf Asad'            → slug='asad',         brand='armaf'
      'Montblanc Explorer'    → slug='explorer',     brand='mont-blanc'
      'Al Haramain L Aventure'→ slug='l-aventure',   brand='al-haramain-perfumes'
    """
    df = load_csv()

    # Build a normalized combined column: remove hyphens so mont-blanc == montblanc
    if "_combined" not in df.columns:
        df["_combined"] = (
            df["Perfume"].str.lower().str.replace("-", "", regex=False) + " " +
            df["Brand"].str.lower().str.replace("-", "", regex=False)
        )

    # Normalize query words — keep words of length >= 2 to handle '9PM', 'Y', etc.
    words = [w.replace("-", "") for w in name.lower().split() if len(w) >= 2]
    if not words:
        return {}

    mask = df["_combined"].str.contains(words[0], na=False)
    for w in words[1:]:
        mask &= df["_combined"].str.contains(w, na=False)

    # Optional extra brand filter
    if brand:
        brand_norm = brand.lower().replace("-", "")
        brand_words = [w for w in brand_norm.split() if len(w) > 2]
        for bw in brand_words:
            mask &= df["_combined"].str.contains(bw, na=False)

    rows = df[mask]
    if rows.empty:
        return {}

    rows = rows.sort_values("_votes", ascending=False)
    r = rows.iloc[0]
    return {
        "display_name": r["Perfume"].replace("-", " ").title(),
        "brand":        r["Brand"].replace("-", " ").title(),
        "url":          str(r.get("url", "")),
        "notes_top":    [n.strip() for n in str(r.get("Top", "")).split(",") if n.strip() and n.strip() != "nan"],
        "notes_middle": [n.strip() for n in str(r.get("Middle", "")).split(",") if n.strip() and n.strip() != "nan"],
        "notes_base":   [n.strip() for n in str(r.get("Base", "")).split(",") if n.strip() and n.strip() != "nan"],
        "accords":      [str(r.get(f"mainaccord{i}", "")) for i in range(1, 6)
                         if str(r.get(f"mainaccord{i}", "")) not in ("", "nan")],
        "rating":       str(r.get("Rating Value", "")),
        "votes":        str(r.get("Rating Count", "")),
        "release_year": str(r.get("Year", "")),
        "gender":       str(r.get("Gender", "")),
    }

# ── DB ────────────────────────────────────────────────────────────────────────
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_similarity_table():
    # Delegated to similarity module
    pass

def _ensure_similarity_table_orig():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fragrance_similarity (
            id TEXT PRIMARY KEY,
            frag_a TEXT NOT NULL,
            frag_b TEXT NOT NULL,
            sim_score REAL NOT NULL,
            method TEXT NOT NULL,
            og_anchor TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def existing_names() -> set:
    conn = get_conn()
    rows = conn.execute("SELECT name FROM perfumes").fetchall()
    conn.close()
    return {r["name"].lower().strip() for r in rows}

def has_reviews(name: str) -> bool:
    conn = get_conn()
    row = conn.execute("SELECT reviews FROM perfumes WHERE LOWER(name)=?", (name.lower(),)).fetchone()
    conn.close()
    if not row: return False
    return len(json.loads(row["reviews"] or "[]")) > 0

def get_profile(name: str) -> dict:
    conn = get_conn()
    row = conn.execute("SELECT reviews FROM perfumes WHERE LOWER(name)=?", (name.lower(),)).fetchone()
    conn.close()
    if not row: return {}
    reviews = json.loads(row["reviews"] or "[]")
    if reviews and reviews[0].get("llm_profile"):
        return reviews[0]["llm_profile"]
    return {}

def upsert(seed: dict, reviews: list, profile: dict, dupe_of: str = None):
    enriched = [dict(r, llm_profile=profile,
                     mentioned_dupe_of=dupe_of) for r in reviews]
    frag_url = seed.get("url") or f"discovered://{seed['display_name'].lower().replace(' ','-')}"
    conn = get_conn()
    conn.execute("""
        INSERT INTO perfumes (
            id, name, brand, release_year, gender,
            notes_top, notes_middle, notes_base,
            rating, votes, description, longevity, sillage,
            image_url, perfume_url, accords, dupes, reviews
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(perfume_url) DO UPDATE SET
            name=excluded.name, brand=excluded.brand,
            release_year=excluded.release_year, gender=excluded.gender,
            notes_top=excluded.notes_top, notes_middle=excluded.notes_middle,
            notes_base=excluded.notes_base, rating=excluded.rating,
            votes=excluded.votes, accords=excluded.accords,
            dupes=excluded.dupes, reviews=excluded.reviews
    """, (
        str(uuid4()),
        seed["display_name"], seed.get("brand","Unknown"),
        seed.get("release_year"), seed.get("gender"),
        json.dumps(seed.get("notes_top",[])),
        json.dumps(seed.get("notes_middle",[])),
        json.dumps(seed.get("notes_base",[])),
        seed.get("rating"), seed.get("votes"),
        None, None, None, None,
        frag_url,
        json.dumps(seed.get("accords",[])),
        json.dumps([dupe_of] if dupe_of else []),
        json.dumps(enriched, ensure_ascii=False),
    ))
    conn.commit()
    conn.close()

def save_similarity(a: str, b: str, score: float, method: str, anchor: str = None):
    conn = get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO fragrance_similarity
        (id, frag_a, frag_b, sim_score, method, og_anchor, created_at)
        VALUES (?,?,?,?,?,?,?)
    """, (str(uuid4()), a, b, round(score, 4), method, anchor,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()

# ── Reddit ────────────────────────────────────────────────────────────────────
def reddit_get(url, params=None):
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 40))
            print(f"    Rate-limited — sleeping {wait}s ...")
            time.sleep(wait)
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            return r.json() if r.status_code == 200 else None
    except Exception as e:
        print(f"    Reddit error: {e}")
    return None

def clean(text: str) -> str:
    text = re.sub(r"http\S+", "[link]", text or "")
    text = re.sub(r"^>+\s?", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def fetch_reviews(name: str, limit: int = 5) -> list:
    """Exact name search — no query augmentation."""
    for q in [f"{name} review", name]:
        data = reddit_get(
            f"https://www.reddit.com/r/{SUBREDDIT}/search.json",
            {"q": q, "restrict_sr":"1", "sort":"relevance",
             "t":"all", "limit": limit + 5, "type":"link"},
        )
        time.sleep(2.5)
        if not data:
            continue

        kws = [w.lower() for w in name.split() if len(w) > 2]
        results = []
        for stub in data["data"]["children"]:
            if len(results) >= limit: break
            p = stub["data"]
            title = p.get("title","")
            if not any(kw in title.lower() for kw in kws):
                continue

            pdata = reddit_get(
                f"https://www.reddit.com/r/{SUBREDDIT}/comments/{p['id']}.json",
                {"limit":40, "depth":2, "sort":"top"},
            )
            time.sleep(1.5)
            comments = []
            if pdata and isinstance(pdata, list) and len(pdata) >= 2:
                for c in pdata[1]["data"]["children"]:
                    if c["kind"] != "t1": continue
                    body = clean(c["data"].get("body",""))
                    if len(body) >= 60:
                        comments.append({
                            "author": c["data"].get("author","[deleted]"),
                            "score":  c["data"].get("score", 0),
                            "body":   body[:1000],
                        })

            text = clean(p.get("selftext",""))
            if len(text) >= 60 or comments:
                results.append({
                    "source": "reddit", "alias_used": q, "title": title,
                    "url":    f"https://www.reddit.com{p.get('permalink','')}",
                    "author": p.get("author","[deleted]"),
                    "date":   datetime.fromtimestamp(p.get("created_utc",0), tz=timezone.utc).strftime("%Y-%m-%d"),
                    "score":  p.get("score",0),
                    "text":   text[:2000],
                    "comments": comments[:6],
                })

        if results:
            return results
    return []

# ── LLM — summarize only ─────────────────────────────────────────────────────
PROFILE_SYSTEM = """You are a fragrance analyst. Summarize Reddit reviews into structured JSON.
Return ONLY valid JSON, no markdown fences.
{
  "vibes": ["3-6 short descriptors like fresh, smoky, office-safe"],
  "longevity_score": 1-10,
  "sillage_score": 1-10,
  "time_of_day": ["morning","midday","afternoon","evening","night"] (subset),
  "season": ["spring","summer","fall","winter"] (subset),
  "occasion": ["office","date","casual","clubbing","outdoor","formal"] (subset),
  "mentioned_dupe_of": "fragrance name string if reviewers call this a clone/dupe of something, else null",
  "sentiment": "positive" | "mixed" | "negative",
  "top_quotes": ["up to 3 short memorable quotes"]
}"""

def llm_summarize(name: str, reviews: list, dupe_of: str = None) -> dict:
    parts = [f"Fragrance: {name}\n"]
    if dupe_of:
        parts.append(f"(Discussed as a dupe/alternative of {dupe_of})\n")
    for i, r in enumerate(reviews, 1):
        parts.append(f"\n--- Post {i}: {r.get('title','')} ---")
        if r.get("text"): parts.append(r["text"][:700])
        for c in r.get("comments",[])[:4]:
            parts.append(f"[u/{c['author']}]: {c['body'][:350]}")
    corpus = "\n".join(parts)[:6000]

    try:
        resp = llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role":"system","content": PROFILE_SYSTEM},
                {"role":"user",  "content": corpus},
            ],
            stream=False,
            extra_body={"thinking": {"type":"disabled"}},
            temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*","",raw); raw = re.sub(r"\s*```$","",raw)
        return json.loads(raw)
    except Exception as e:
        print(f"    LLM error: {e}")
        return {}

# ── Similarity ────────────────────────────────────────────────────────────────
ALL_SEASONS   = ["spring","summer","fall","winter"]
ALL_TIMES     = ["morning","midday","afternoon","evening","night"]
ALL_OCCASIONS = ["office","date","casual","clubbing","outdoor","formal"]
ALL_ACCORDS   = [
    "aromatic","woody","fresh","citrus","floral","sweet","amber","spicy",
    "fresh spicy","warm spicy","soft spicy","musky","powdery","green",
    "aquatic","leather","vanilla","oud","smoky","earthy","balsamic",
]
KNOWN_VIBES = [
    "fresh","clean","masculine","feminine","office-safe","date-night",
    "casual","sporty","sweet","smoky","woody","earthy","citrus","floral",
    "powdery","warm","cold","heavy","light","long-lasting","beast-mode",
    "crowd-pleaser","unique","safe","polarizing","versatile",
]

def build_vector(profile: dict, accords: list, notes: list) -> dict:
    """Build a normalized feature dict from LLM profile + CSV data."""
    vec = {}
    # Numerics
    vec["longevity"] = (profile.get("longevity_score") or 5) / 10.0
    vec["sillage"]   = (profile.get("sillage_score")   or 5) / 10.0

    # One-hot: seasons
    seasons = [s.lower() for s in (profile.get("season") or [])]
    for s in ALL_SEASONS:
        vec[f"season_{s}"] = 1.0 if s in seasons else 0.0

    # One-hot: time of day
    times = [t.lower() for t in (profile.get("time_of_day") or [])]
    for t in ALL_TIMES:
        vec[f"time_{t}"] = 1.0 if t in times else 0.0

    # One-hot: occasions
    occ = [o.lower() for o in (profile.get("occasion") or [])]
    for o in ALL_OCCASIONS:
        vec[f"occ_{o}"] = 1.0 if o in occ else 0.0

    # One-hot: vibes (controlled vocab)
    vibes = [v.lower() for v in (profile.get("vibes") or [])]
    for v in KNOWN_VIBES:
        vec[f"vibe_{v}"] = 1.0 if any(v in vibe or vibe in v for vibe in vibes) else 0.0

    # One-hot: accords from CSV
    acc_lower = [a.lower() for a in accords]
    for a in ALL_ACCORDS:
        vec[f"accord_{a.replace(' ','_')}"] = 1.0 if a in acc_lower else 0.0

    # Note-overlap factor (Jaccard-style count)
    note_count = len(set(n.lower() for n in notes))
    vec["note_richness"] = min(note_count / 15.0, 1.0)

    return vec

def cosine_sim(va: dict, vb: dict) -> float:
    keys = set(va) | set(vb)
    dot  = sum(va.get(k,0) * vb.get(k,0) for k in keys)
    na   = math.sqrt(sum(va.get(k,0)**2 for k in keys))
    nb   = math.sqrt(sum(vb.get(k,0)**2 for k in keys))
    return round(dot / (na * nb + 1e-9), 4)

def note_jaccard(notes_a: list, notes_b: list) -> float:
    a = {n.lower().strip() for n in notes_a if n}
    b = {n.lower().strip() for n in notes_b if n}
    if not a and not b: return 0.0
    return round(len(a & b) / len(a | b), 4)

def compute_chain_similarity(chain: dict):
    """
    chain = {"anchor": name, "members": [{"name", "profile", "accords", "notes"}]}
    Computes pairwise similarity for all members (including anchor).
    """
    members = chain["members"]
    anchor  = chain["anchor"]
    n = len(members)
    if n < 2:
        return

    print(f"\n  Computing similarity for chain: {anchor} ({n} members)")
    vectors = {}
    for m in members:
        all_notes = m.get("notes_top",[]) + m.get("notes_middle",[]) + m.get("notes_base",[])
        vectors[m["name"]] = {
            "vec":    build_vector(m.get("profile",{}), m.get("accords",[]), all_notes),
            "notes":  all_notes,
            "accords": m.get("accords",[]),
        }

    for i in range(n):
        for j in range(i+1, n):
            a = members[i]["name"]
            b = members[j]["name"]
            va = vectors[a]["vec"]
            vb = vectors[b]["vec"]

            profile_sim = cosine_sim(va, vb)
            note_sim    = note_jaccard(vectors[a]["notes"], vectors[b]["notes"])
            # Weighted blend: 60% profile, 40% notes
            blended = round(0.6 * profile_sim + 0.4 * note_sim, 4)

            save_similarity(a, b, blended, "blended_cosine", anchor)
            print(f"    {a:35s} <-> {b:35s}  sim={blended:.3f}  "
                  f"(profile={profile_sim:.3f}, notes={note_sim:.3f})")

# ── Main pipeline ──────────────────────────────────────────────────────────────
def process_one(seed: dict, known: set, dupe_of: str = None) -> dict | None:
    name = seed["display_name"]
    print(f"\n  Processing: {name} ({'dupe of ' + dupe_of if dupe_of else 'OG seed'})")

    if has_reviews(name):
        print(f"    Already in DB — loading profile")
        profile = get_profile(name)
        return {"name": name, "profile": profile,
                "accords": seed.get("accords",[]),
                "notes_top": seed.get("notes_top",[]),
                "notes_middle": seed.get("notes_middle",[]),
                "notes_base": seed.get("notes_base",[])}

    reviews = fetch_reviews(name, limit=5)
    time.sleep(3.0)

    if not reviews:
        print(f"    No reviews found — skipping")
        return None

    profile = llm_summarize(name, reviews, dupe_of)
    upsert(seed, reviews, profile, dupe_of)
    known.add(name.lower())
    print(f"    Saved: {len(reviews)} reviews | "
          f"longevity={profile.get('longevity_score')}, "
          f"sillage={profile.get('sillage_score')}, "
          f"dupe_of={profile.get('mentioned_dupe_of')}")

    return {"name": name, "profile": profile,
            "accords": seed.get("accords",[]),
            "notes_top": seed.get("notes_top",[]),
            "notes_middle": seed.get("notes_middle",[]),
            "notes_base": seed.get("notes_base",[])}

def run():
    print("\n" + "="*60)
    print("  Pipeline v2 — Seeds + Cascade + Similarity")
    print("="*60)

    sim_module.init(DB_PATH)
    known = existing_names()

    # Select seeds
    seeds = select_seeds(N_SEEDS)
    print(f"\n  Selected {len(seeds)} seeds:")
    for s in seeds:
        print(f"    - {s['display_name']} ({s['brand']}) | {s['votes']} votes")

    chains = {}       # anchor_name -> [member dicts]
    dupe_queue = {}   # dupe_name -> anchor_name

    # ── Phase 0: Curated known dupe chains ─────────────────────────────────
    print(f"\n[ PHASE 0 ] Processing curated known dupe chains ...\n")
    phase0_urls = set()       # Fragrantica URLs processed in Phase 0
    phase0_names = set()      # Canonical display names (lower) processed in Phase 0

    for anchor_name, dupe_names in KNOWN_CHAINS.items():
        # Use CSV canonical name — do NOT override with anchor label
        csv_name, csv_brand = ANCHOR_CSV_LOOKUP.get(anchor_name, (anchor_name.lower(), ""))
        anchor_meta = csv_lookup(csv_name, csv_brand)
        if not anchor_meta:
            anchor_meta = {"display_name": anchor_name, "brand": "",
                           "notes_top":[], "notes_middle":[], "notes_base":[],
                           "accords":[], "rating":None, "votes":None,
                           "release_year":None, "gender":None, "url":""}
        else:
            # Keep CSV canonical name but print both for clarity
            csv_display = anchor_meta["display_name"]
            if csv_display.lower() != anchor_name.lower():
                print(f"  Anchor '{anchor_name}' -> CSV canonical: '{csv_display}'")

        phase0_urls.add(anchor_meta.get("url", ""))
        phase0_names.add(anchor_meta["display_name"].lower())

        anchor_result = process_one(anchor_meta, known)
        if anchor_result:
            chains[anchor_name] = [anchor_result]
            for dupe_name in dupe_names:
                dupe_meta = csv_lookup(dupe_name)
                if not dupe_meta:
                    dupe_meta = {"display_name": dupe_name, "brand": "Unknown",
                                 "notes_top":[], "notes_middle":[], "notes_base":[],
                                 "accords":[], "rating":None, "votes":None,
                                 "release_year":None, "gender":None, "url":""}
                dupe_meta["display_name"] = dupe_name
                phase0_urls.add(dupe_meta.get("url", ""))
                phase0_names.add(dupe_meta["display_name"].lower())
                dupe_result = process_one(dupe_meta, known, dupe_of=anchor_meta["display_name"])
                if dupe_result:
                    chains[anchor_name].append(dupe_result)
                time.sleep(3.0)
        time.sleep(3.0)

    # ── Phase 1: Process seeds ──────────────────────────────────────────────
    print(f"\n[ PHASE 1 ] Processing {len(seeds)} seeds ...\n")
    for seed in seeds:
        # Skip seeds already fully handled in Phase 0 (same URL or same canonical name)
        seed_url  = seed.get("url", "")
        seed_name = seed["display_name"].lower()
        if (seed_url and seed_url in phase0_urls) or seed_name in phase0_names:
            print(f"  Skipping (Phase 0 duplicate): {seed['display_name']}")
            continue
        result = process_one(seed, known)
        if not result:
            continue

        name = result["name"]
        chains[name] = [result]   # anchor is first member

        # Validate cascade claim — skip if dupe_of doesn't loosely match the seed
        dupe = (result["profile"] or {}).get("mentioned_dupe_of")
        if dupe and isinstance(dupe, str) and dupe.strip():
            dupe_clean = dupe.strip()
            # Sanity check: the mentioned OG should loosely match the current anchor seed
            # to avoid spurious cross-chain dupe claims (e.g. Arabians Tonka -> LV Symphony)
            anchor_words = set(name.lower().split())
            dupe_words   = set(dupe_clean.lower().split())
            # Accept if 1+ meaningful word overlaps OR dupe is about a known anchor
            known_anchors_lower = {k.lower() for k in KNOWN_CHAINS}
            plausible = (
                len(anchor_words & dupe_words) >= 1 or
                any(dupe_clean.lower() in a or a in dupe_clean.lower()
                    for a in known_anchors_lower)
            )
            if not plausible:
                print(f"    --> Skipping implausible cascade: '{name}' claims dupe of '{dupe_clean}'")
            elif dupe_clean.lower() not in known:
                dupe_queue[dupe_clean] = name
                print(f"    --> Queued dupe: '{dupe_clean}' (mentioned in {name} reviews)")
            else:
                # Still add to chain for similarity computation
                dupe_meta = csv_lookup(dupe_clean)
                if dupe_meta:
                    dupe_profile = get_profile(dupe_clean)
                    chains[name].append({
                        "name": dupe_clean,
                        "profile": dupe_profile,
                        "accords": dupe_meta.get("accords",[]),
                        "notes_top": dupe_meta.get("notes_top",[]),
                        "notes_middle": dupe_meta.get("notes_middle",[]),
                        "notes_base": dupe_meta.get("notes_base",[]),
                    })
                    print(f"    --> Already in DB: '{dupe_clean}' added to similarity chain")

        time.sleep(3.0)

    # ── Phase 2: Cascade (depth 1) ─────────────────────────────────────────
    print(f"\n[ PHASE 2 ] Cascade — processing {len(dupe_queue)} queued dupes ...\n")
    for dupe_name, anchor_name in dupe_queue.items():
        # Try CSV lookup for metadata
        dupe_meta = csv_lookup(dupe_name)
        if dupe_meta:
            dupe_seed = dupe_meta
            dupe_seed["display_name"] = dupe_meta.get("display_name", dupe_name)
        else:
            dupe_seed = {
                "display_name": dupe_name, "brand": "Unknown",
                "notes_top":[], "notes_middle":[], "notes_base":[],
                "accords":[], "rating":None, "votes":None,
                "release_year":None, "gender":None, "url":"",
            }

        result = process_one(dupe_seed, known, dupe_of=anchor_name)
        if result and anchor_name in chains:
            chains[anchor_name].append(result)
        time.sleep(3.0)

    # ── Phase 3: Similarity ────────────────────────────────────────────────
    print(f"\n[ PHASE 3 ] Computing similarity scores ...\n")
    for anchor_name, members in chains.items():
        if len(members) < 2:
            continue
        names_in_chain = [m["name"] for m in members]
        print(f"  Chain: {anchor_name} — {len(members)} members")
        results = sim_module.compute_all_pairs(names_in_chain, og_anchor=anchor_name)
        for r in results:
            print(f"    {r['frag_a']:30s} <-> {r['frag_b']:30s} "
                  f"sim={r['sim_score']:.3f} "
                  f"(text={r['text_sim']:.3f} note={r['note_sim']:.3f} "
                  f"accord={r['accord_sim']:.3f} bonus={r['dupe_bonus']:.2f})")

    # Also compute cross-chain similarity for all fragrances in DB
    print(f"\n  Computing cross-chain similarity for all {len(known)} DB entries ...")
    all_names = list(known)[:40]  # cap at 40 to limit API calls
    cross_results = sim_module.compute_all_pairs(all_names)
    print(f"  Stored {len(cross_results)} cross-chain similarity pairs")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM perfumes").fetchone()[0]
    sim_count = conn.execute("SELECT COUNT(*) FROM fragrance_similarity").fetchone()[0]
    conn.close()
    print(f"  Total fragrances in DB : {total}")
    print(f"  Similarity pairs stored: {sim_count}")
    print(f"  Chains processed       : {len(chains)}")
    for anchor, members in chains.items():
        print(f"    {anchor}: {len(members)} members ({', '.join(m['name'] for m in members[1:])} or none)")

    print(f"\n  Refresh viewer:")
    print(f"  python github_project_references/PerfumAPI/reddit_reviews.py --from-db\n")

if __name__ == "__main__":
    run()
