"""
semantic_similarity_v3.py — BM25 + Vector RRF Hybrid Search
Computes Lexical rank via BM25 (Notes + Accords) and Semantic rank via SentenceTransformers.
Applies Reciprocal Rank Fusion (RRF), normalizes to [0,1], adds structural Dupe Bonus, and stores as 'hybrid_rrf_v3'.
"""
import sys, os, sqlite3, json, math
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "github_project_references", "PerfumAPI", "perfumes.db")

print("Loading semantic model 'all-MiniLM-L6-v2' ...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def load_all_profiles(names=None):
    """Load all profiles, optionally filtered by a list of names."""
    conn = _conn()
    q = "SELECT name, brand, notes_top, notes_middle, notes_base, accords, reviews FROM perfumes"
    if names:
        placeholders = ",".join("?" for _ in names)
        q += f" WHERE LOWER(name) IN ({placeholders})"
        args = [n.lower() for n in names]
        rows = conn.execute(q, args).fetchall()
    else:
        rows = conn.execute(q).fetchall()
    conn.close()

    profiles = {}
    for r in rows:
        try: n_top = json.loads(r["notes_top"] or "[]")
        except: n_top = []
        try: n_mid = json.loads(r["notes_middle"] or "[]")
        except: n_mid = []
        try: n_bas = json.loads(r["notes_base"] or "[]")
        except: n_bas = []
        try: acc = json.loads(r["accords"] or "[]")
        except: acc = []
        try: revs = json.loads(r["reviews"] or "[]")
        except: revs = []

        all_notes = set(str(n).lower() for n in n_top + n_mid + n_bas if n)
        all_accords = set(str(a).lower() for a in acc if a)

        # 1. Lexical Tokenization for BM25 (Notes + Accords are heavily weighted)
        bm25_tokens = list(all_notes) * 2 + list(all_accords)
        
        # 2. Semantic Text for Vector
        vibes, occasions, quotes = [], [], []
        for rev in revs:
            p = rev.get("llm_profile") or {}
            vibes.extend(p.get("vibes") or [])
            occasions.extend(p.get("occasion") or [])
            quotes.extend(p.get("top_quotes") or [])
        
        text_blocks = []
        if vibes: text_blocks.append("Vibes: " + ", ".join(set(vibes)))
        if occasions: text_blocks.append("Occasions: " + ", ".join(set(occasions)))
        if all_accords: text_blocks.append("Accords: " + ", ".join(all_accords))
        if quotes: text_blocks.append("Reviews say: " + " ".join(set(quotes[:5])))
        semantic_text = " ".join(text_blocks)

        profiles[r["name"]] = {
            "name": r["name"],
            "bm25_tokens": bm25_tokens,
            "semantic_text": semantic_text
        }
    return profiles

def get_rrf_score(rank):
    return 1.0 / (60.0 + rank)

def compute_all_pairs_v3(profiles_dict, og_anchor=None):
    """
    Computes all-pairs hybrid similarity using symmetric RRF normalized to 0-1.
    """
    names = list(profiles_dict.keys())
    N = len(names)
    if N < 2: return []

    # 1. Precompute semantic vectors
    print("  Encoding vectors ...")
    vectors = {}
    for n in names:
        vectors[n] = model.encode(profiles_dict[n]["semantic_text"], convert_to_tensor=True)

    # 2. BM25 Corpus Preparation
    # To compute BM25 efficiently per query, we can build the corpus once
    tokenized_corpus = [profiles_dict[n]["bm25_tokens"] for n in names]
    bm25 = BM25Okapi(tokenized_corpus)

    # 3. Compute RRF per query
    # rrf_matrix[i][j] will store RRF(j | query=i)
    rrf_matrix = {n: {} for n in names}
    
    print("  Calculating rank matrices ...")
    for i, q_name in enumerate(names):
        q_prof = profiles_dict[q_name]
        q_vec = vectors[q_name]
        
        # BM25 scores against all
        bm25_scores = bm25.get_scores(q_prof["bm25_tokens"])
        
        # Vector scores against all
        # util.cos_sim can take (1, D) and (N, D)
        # But we can just loop since N=150 is small
        v_scores = []
        for j, d_name in enumerate(names):
            if i == j:
                v_scores.append(-1.0) # ignore self
                bm25_scores[j] = -1.0
            else:
                score = float(util.cos_sim(q_vec, vectors[d_name])[0][0])
                v_scores.append(score)

        # Rank them (1 is best, highest score)
        # We pair index with score, sort descending, assign rank
        b_sorted = sorted([(score, j) for j, score in enumerate(bm25_scores) if i != j], key=lambda x: x[0], reverse=True)
        v_sorted = sorted([(score, j) for j, score in enumerate(v_scores) if i != j], key=lambda x: x[0], reverse=True)

        b_ranks = {idx: rank for rank, (sc, idx) in enumerate(b_sorted, 1)}
        v_ranks = {idx: rank for rank, (sc, idx) in enumerate(v_sorted, 1)}

        for j, d_name in enumerate(names):
            if i == j: continue
            rrf = get_rrf_score(b_ranks[j]) + get_rrf_score(v_ranks[j])
            rrf_matrix[q_name][d_name] = rrf

    # 4. Symmetrize and Normalize
    # Find min/max RRF across all to normalize
    all_rrfs = []
    sym_rrf_matrix = {}
    for i in range(N):
        n1 = names[i]
        for j in range(i+1, N):
            n2 = names[j]
            # Average RRFs
            sym_rrf = (rrf_matrix[n1][n2] + rrf_matrix[n2][n1]) / 2.0
            all_rrfs.append(sym_rrf)
            sym_rrf_matrix[(n1, n2)] = sym_rrf
            sym_rrf_matrix[(n2, n1)] = sym_rrf

    if all_rrfs:
        min_rrf = min(all_rrfs)
        max_rrf = max(all_rrfs)
    else:
        min_rrf, max_rrf = 0, 1

    # Avoid division by zero
    if max_rrf == min_rrf: max_rrf = min_rrf + 1e-9

    results = []
    conn = _conn()
    
    print("  Normalizing and applying dupe bonus ...")
    for i in range(N):
        for j in range(i+1, N):
            n1 = names[i]
            n2 = names[j]
            sym_rrf = sym_rrf_matrix[(n1, n2)]
            
            # Normalize to 0-1
            norm_score = (sym_rrf - min_rrf) / (max_rrf - min_rrf)
            
            # 5. Dupe Bonus
            bonus = 0.0
            is_chain_member = bool(og_anchor)
            if is_chain_member:
                name_lo = min(n1, n2, key=str.lower)
                name_hi = max(n1, n2, key=str.lower)
                if name_lo.lower() == og_anchor.lower() or name_hi.lower() == og_anchor.lower():
                    bonus = 0.30  # OG <-> Dupe
                else:
                    bonus = 0.20  # Dupe <-> Dupe

            # Final Score (Blend normalized RRF and Bonus)
            final_score = round(norm_score + bonus * (1.0 - norm_score), 4)

            name_lo = min(n1, n2, key=str.lower)
            name_hi = max(n1, n2, key=str.lower)
            pair_id = f"hybrid_rrf_v3|||{name_lo.lower()}|||{name_hi.lower()}"

            conn.execute("""
                INSERT OR REPLACE INTO fragrance_similarity
                (id, frag_a, frag_b, sim_score, text_sim, note_sim, accord_sim,
                 dupe_bonus, method, og_anchor, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                pair_id, name_lo, name_hi, final_score,
                norm_score, 0.0, 0.0, bonus, "hybrid_rrf_v3",
                og_anchor, datetime.now().isoformat()
            ))
            
            results.append({
                "frag_a": name_lo, "frag_b": name_hi,
                "sim_score": final_score, "rrf_raw": sym_rrf,
                "dupe_bonus": bonus
            })
    conn.commit()
    conn.close()
    return results
