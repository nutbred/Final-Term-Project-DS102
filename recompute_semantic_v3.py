"""
recompute_semantic_v3.py
Computes the new BM25 + Vector RRF Hybrid similarity for all fragrances in the DB.
Does NOT wipe the old absa_rrf or hybrid_v1 scores.
"""
import sys, os, sqlite3
sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DB   = os.path.join(HERE, "github_project_references", "PerfumAPI", "perfumes.db")

os.chdir(HERE)
import semantic_similarity_v3 as sim_module
import pipeline_v3 as p

# 1. Clean out any previous 'hybrid_rrf_v3' runs (just in case of re-runs)
conn = sqlite3.connect(DB)
conn.execute("DELETE FROM fragrance_similarity WHERE method='hybrid_rrf_v3'")
conn.commit()

# Load all names
rows = conn.execute("SELECT name FROM perfumes").fetchall()
conn.close()
all_names = [r[0] for r in rows]
print(f"Computing HYBRID RRF V3 pairs for {len(all_names)} fragrances ...")

# 2. Load profiles
profiles = sim_module.load_all_profiles()

# 3. Process Known Chains (for Dupe Bonus)
processed_chains = set()
for anchor_name, dupe_names in p.KNOWN_CHAINS.items():
    csv_name, csv_brand = p.ANCHOR_CSV_LOOKUP.get(anchor_name, (anchor_name.lower(), ""))
    meta = p.csv_lookup(csv_name, csv_brand)
    canonical = meta["display_name"] if meta else anchor_name
    chain_members = [canonical] + list(dupe_names)
    
    # Only members that exist in DB
    chain_members = [n for n in chain_members if any(n.lower() == db.lower() for db in all_names)]
    if len(chain_members) >= 2:
        print(f"  Chain: {canonical} ({len(chain_members)} members)")
        # Filter profiles dict
        chain_profs = {k: v for k, v in profiles.items() if k.lower() in [c.lower() for c in chain_members]}
        results = sim_module.compute_all_pairs_v3(chain_profs, og_anchor=canonical)
        for r in results:
            print(f"    {r['frag_a']:30s} <-> {r['frag_b']:30s} rrf_v3={r['sim_score']:.3f}")
        processed_chains.add(canonical)

# 4. Process Cross-chain
print(f"\nComputing global cross-chain RRF pairs for all {len(all_names)} entries (this may take a minute) ...")
sim_module.compute_all_pairs_v3(profiles)

conn = sqlite3.connect(DB)
old_v1 = conn.execute("SELECT COUNT(*) FROM fragrance_similarity WHERE method='absa_rrf'").fetchone()[0]
old_v2 = conn.execute("SELECT COUNT(*) FROM fragrance_similarity WHERE method='hybrid_v1'").fetchone()[0]
new_v3 = conn.execute("SELECT COUNT(*) FROM fragrance_similarity WHERE method='hybrid_rrf_v3'").fetchone()[0]
conn.close()

print(f"\nDone! V1 (Lexical): {old_v1} | V2 (Hybrid): {old_v2} | V3 (RRF Hybrid): {new_v3}")
