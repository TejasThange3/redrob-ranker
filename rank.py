#!/usr/bin/env python3
"""
Redrob AI Candidate Ranker — Main Pipeline
============================================
Produces a top-100 ranked CSV of candidates for the Senior AI Engineer JD.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Constraints (must satisfy for Stage 3):
  - ≤ 5 minutes wall-clock
  - ≤ 16 GB RAM
  - CPU only (no GPU)
  - No network calls

Architecture:
  1. Load candidates + pre-computed embeddings
  2. Compute semantic similarity (JD ↔ candidate)
  3. Compute rule-based scores (title, skill, experience, location)
  4. Apply behavioral signal multipliers
  5. Apply consulting-career penalty
  6. Filter honeypots
  7. Combine → final score → rank top 100
  8. Generate reasoning
  9. Write CSV
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from scorer.title_scorer import score_title
from scorer.skill_scorer import score_skills
from scorer.experience_scorer import score_experience
from scorer.location_scorer import score_location
from scorer.signal_scorer import score_signals
from scorer.honeypot_detector import is_honeypot
from scorer.consulting_checker import check_consulting_career
from scorer.reasoning_generator import generate_reasoning

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

# Score weights for the final combination
WEIGHTS = {
    "semantic":    0.20,  # Embedding cosine similarity
    "title":       0.30,  # Title & career relevance (most important!)
    "skill":       0.20,  # Skill match
    "experience":  0.12,  # Experience fit
    "location":    0.12,  # Location & logistics
    "education":   0.06,  # Education signal
}

TOP_K = 100  # Number of candidates to output


def score_education(candidate: dict) -> float:
    """
    Score education relevance.
    Factors: degree field, institution tier, relevance to AI/ML.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3  # No education data — neutral-low

    best_score = 0.0

    for edu in education:
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()
        tier = edu.get("tier", "unknown")

        # Field relevance
        if any(f in field for f in [
            "computer science", "machine learning", "artificial intelligence",
            "data science", "information technology", "software",
            "computational", "statistics",
        ]):
            field_score = 1.0
        elif any(f in field for f in [
            "mathematics", "physics", "electronics",
            "electrical", "information systems",
        ]):
            field_score = 0.7
        elif any(f in field for f in [
            "engineering",  # Generic engineering
        ]):
            field_score = 0.4
        else:
            field_score = 0.15

        # Degree level
        if "ph.d" in degree or "phd" in degree:
            degree_score = 1.0
        elif any(d in degree for d in ["m.tech", "m.s", "m.sc", "m.e.", "master", "mba"]):
            degree_score = 0.8
        elif any(d in degree for d in ["b.tech", "b.e.", "b.sc", "b.s", "bachelor"]):
            degree_score = 0.6
        else:
            degree_score = 0.4

        # Institution tier
        tier_scores = {
            "tier_1": 1.0,
            "tier_2": 0.7,
            "tier_3": 0.45,
            "tier_4": 0.25,
            "unknown": 0.35,
        }
        tier_score = tier_scores.get(tier, 0.35)

        # Combine for this education entry
        edu_score = 0.45 * field_score + 0.25 * degree_score + 0.30 * tier_score
        best_score = max(best_score, edu_score)

    return round(best_score, 4)


def load_candidates(path: str) -> list[dict]:
    """Load candidates from JSON array or JSONL file (including .gz)."""
    candidates = []
    p = Path(path)

    if p.suffix == ".gz":
        import gzip
        opener = lambda: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda: open(p, "r", encoding="utf-8")

    with opener() as f:
        # Peek at first non-whitespace character to detect format
        first_char = ""
        while True:
            ch = f.read(1)
            if not ch:
                break
            if ch.strip():
                first_char = ch
                break

        if first_char == "[":
            # JSON array — need to read whole file
            content = first_char + f.read()
            candidates = json.loads(content)
        else:
            # JSONL — stream line by line (memory efficient for 465MB file)
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

    return candidates


def load_embeddings():
    """Load pre-computed embeddings if available."""
    embeddings_path = DATA_DIR / "embeddings.npy"
    ids_path = DATA_DIR / "candidate_ids.json"
    jd_path = DATA_DIR / "jd_embedding.npy"

    if not embeddings_path.exists():
        return None, None, None

    embeddings = np.load(embeddings_path)
    with open(ids_path, "r") as f:
        candidate_ids = json.load(f)
    jd_embedding = np.load(jd_path)

    return embeddings, candidate_ids, jd_embedding


def compute_semantic_scores(embeddings, candidate_ids, jd_embedding, id_to_idx):
    """Compute cosine similarity between JD and all candidates."""
    if embeddings is None:
        return {}

    # Dot product (embeddings are already normalized)
    similarities = embeddings @ jd_embedding

    # Normalize to 0-1 range
    min_sim = similarities.min()
    max_sim = similarities.max()
    if max_sim > min_sim:
        similarities = (similarities - min_sim) / (max_sim - min_sim)
    else:
        similarities = np.zeros_like(similarities)

    # Build lookup
    semantic_scores = {}
    for i, cid in enumerate(candidate_ids):
        semantic_scores[cid] = float(similarities[i])

    return semantic_scores


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates for the Senior AI Engineer JD"
    )
    parser.add_argument(
        "--candidates", "-c",
        required=True,
        help="Path to candidates.jsonl (or .jsonl.gz)"
    )
    parser.add_argument(
        "--out", "-o",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K,
        help=f"Number of top candidates to output (default: {TOP_K})"
    )
    args = parser.parse_args()

    start_time = time.time()

    # --- 1. Load data ---
    print("Loading candidates...")
    candidates = load_candidates(args.candidates)
    print(f"  Loaded {len(candidates)} candidates")

    print("Loading pre-computed embeddings...")
    embeddings, emb_ids, jd_embedding = load_embeddings()
    has_embeddings = embeddings is not None
    if has_embeddings:
        print(f"  Loaded embeddings: {embeddings.shape}")
    else:
        print("  No pre-computed embeddings found — using rule-based scoring only")
        # Adjust weights when no embeddings
        WEIGHTS["semantic"] = 0.0
        WEIGHTS["title"] = 0.38
        WEIGHTS["skill"] = 0.28
        WEIGHTS["experience"] = 0.14
        WEIGHTS["location"] = 0.14
        WEIGHTS["education"] = 0.06

    # Build ID → index mapping for embeddings
    id_to_idx = {}
    if emb_ids:
        id_to_idx = {cid: i for i, cid in enumerate(emb_ids)}

    # Compute semantic scores
    semantic_scores = {}
    if has_embeddings:
        print("Computing semantic similarities...")
        semantic_scores = compute_semantic_scores(
            embeddings, emb_ids, jd_embedding, id_to_idx
        )

    # --- 2. Score all candidates ---
    print("Scoring candidates...")
    scored_candidates = []

    for candidate in tqdm(candidates, desc="Scoring"):
        cid = candidate.get("candidate_id", "")

        # Check honeypot first (saves time on scoring)
        honeypot = is_honeypot(candidate)

        # Component scores
        title_sc = score_title(candidate)
        skill_sc = score_skills(candidate)
        exp_sc = score_experience(candidate)
        loc_sc = score_location(candidate)
        edu_sc = score_education(candidate)
        signal_mult = score_signals(candidate)
        consulting_mult = check_consulting_career(candidate)
        semantic_sc = semantic_scores.get(cid, 0.0)

        # Weighted base score
        base_score = (
            WEIGHTS["semantic"] * semantic_sc +
            WEIGHTS["title"] * title_sc +
            WEIGHTS["skill"] * skill_sc +
            WEIGHTS["experience"] * exp_sc +
            WEIGHTS["location"] * loc_sc +
            WEIGHTS["education"] * edu_sc
        )

        # Apply multipliers
        final_score = base_score * signal_mult * consulting_mult

        # Honeypot elimination
        if honeypot:
            final_score *= 0.01  # Effectively kill their score

        scored_candidates.append({
            "candidate": candidate,
            "candidate_id": cid,
            "final_score": final_score,
            "components": {
                "semantic": semantic_sc,
                "title": title_sc,
                "skill": skill_sc,
                "experience": exp_sc,
                "location": loc_sc,
                "education": edu_sc,
                "signal_mult": signal_mult,
                "consulting_mult": consulting_mult,
                "is_honeypot": honeypot,
            },
        })

    # --- 3. Sort and select top-K ---
    print(f"Ranking top {args.top_k}...")
    scored_candidates.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    top_k = scored_candidates[:args.top_k]

    # --- 4. Generate reasoning ---
    print("Generating reasoning...")
    results = []
    for rank_idx, entry in enumerate(top_k):
        rank = rank_idx + 1
        reasoning = generate_reasoning(
            entry["candidate"],
            rank,
            entry["final_score"],
            entry["components"],
        )
        results.append({
            "candidate_id": entry["candidate_id"],
            "rank": rank,
            "score": round(entry["final_score"], 4),
            "reasoning": reasoning,
        })

    # --- 5. Write CSV ---
    print(f"Writing {args.out}...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time

    # --- 6. Summary ---
    print(f"\n{'='*60}")
    print(f"DONE — {args.out}")
    print(f"{'='*60}")
    print(f"  Candidates scored:  {len(candidates)}")
    print(f"  Top {args.top_k} written:    {len(results)}")
    print(f"  Honeypots filtered: {sum(1 for s in scored_candidates if s['components']['is_honeypot'])}")
    print(f"  Score range:        {results[0]['score']:.4f} (rank 1) → {results[-1]['score']:.4f} (rank {args.top_k})")
    print(f"  Time elapsed:       {elapsed:.1f}s")

    if elapsed > 300:
        print(f"\n  ⚠️  WARNING: Exceeded 5-minute limit ({elapsed:.0f}s)!")
    else:
        print(f"  ✅ Within 5-minute limit ({elapsed:.1f}s / 300s)")

    # Show top 5 for quick inspection
    print(f"\nTop 5 candidates:")
    for r in results[:5]:
        cand = top_k[r["rank"]-1]["candidate"]
        title = cand["profile"]["current_title"]
        company = cand["profile"]["current_company"]
        print(f"  #{r['rank']}: {r['candidate_id']} — {title} @ {company} "
              f"(score: {r['score']:.4f})")

    # Warn if many honeypots in top 100
    honeypot_count = sum(
        1 for s in top_k if s["components"]["is_honeypot"]
    )
    if honeypot_count > 0:
        print(f"\n  ⚠️  {honeypot_count} potential honeypots in top {args.top_k}!")


if __name__ == "__main__":
    main()
