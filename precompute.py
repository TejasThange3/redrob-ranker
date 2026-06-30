#!/usr/bin/env python3
"""
Pre-computation Script
======================
Generates semantic embeddings for all candidates OFFLINE.
This step has no time limit — it runs before the ranking step.

What it does:
  1. Load all candidates from candidates.jsonl
  2. Build a text representation for each candidate
  3. Encode using sentence-transformers (all-MiniLM-L6-v2)
  4. Encode the Job Description
  5. Save everything to disk

Output files (in data/ directory):
  - embeddings.npy — (N, 384) float32 array of candidate embeddings
  - candidate_ids.json — list of candidate_id strings (same order)
  - jd_embedding.npy — (384,) float32 array for the JD
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, 384-dim, good quality
BATCH_SIZE = 256
OUTPUT_DIR = Path(__file__).parent / "data"

# The Job Description text (extracted from job_description.md)
JD_TEXT = """
Senior AI Engineer - Founding Team at Redrob AI.
Series A AI-native talent intelligence platform. Location: Pune/Noida, India (Hybrid).
Experience Required: 5-9 years.

We need someone who is simultaneously comfortable with deep technical depth in modern
ML systems (embeddings, retrieval, ranking, LLMs, fine-tuning) and a scrappy
product-engineering attitude — willing to ship a working ranker quickly.

Own the intelligence layer: ranking, retrieval, and matching systems that decide what
recruiters see when they search for candidates.

Must-have skills:
- Production experience with embeddings-based retrieval systems (sentence-transformers,
  OpenAI embeddings, BGE, E5, or similar)
- Production experience with vector databases or hybrid search (Pinecone, Weaviate,
  Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS)
- Strong Python
- Experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP,
  offline-to-online correlation, A/B testing)

Nice-to-have:
- LLM fine-tuning (LoRA, QLoRA, PEFT)
- Learning-to-rank models (XGBoost-based or neural)
- HR-tech or marketplace product experience
- Distributed systems or large-scale inference optimization
- Open-source contributions

Ideal candidate: 6-8 years total experience, 4-5 in applied ML/AI at product companies.
Has shipped at least one end-to-end ranking, search, or recommendation system.
Located in or willing to relocate to Noida or Pune.
"""


def build_candidate_text(candidate: dict) -> str:
    """
    Build a rich text representation of a candidate for embedding.
    Combines headline, summary, career descriptions, and skill names.
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    parts = []

    # Headline and summary
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    title = profile.get("current_title", "")
    if headline:
        parts.append(headline)
    if title and title.lower() not in headline.lower():
        parts.append(f"Current role: {title}")
    if summary:
        # Truncate very long summaries
        parts.append(summary[:500])

    # Career history descriptions (most recent first, limited)
    for job in career_history[:4]:
        desc = job.get("description", "")
        job_title = job.get("title", "")
        if desc:
            parts.append(f"{job_title}: {desc[:300]}")

    # Skill names
    skill_names = [s.get("name", "") for s in skills if s.get("name")]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    return " ".join(parts)


def main():
    """Run the pre-computation pipeline."""
    candidates_path = None

    # Find candidates file
    for p in [
        Path(__file__).parent / "candidates.jsonl",
        Path(r"d:\Hackathon\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"),
    ]:
        if p.exists():
            candidates_path = p
            break

    if candidates_path is None:
        if len(sys.argv) > 1:
            candidates_path = Path(sys.argv[1])
        else:
            print("Usage: python precompute.py [path/to/candidates.jsonl]")
            print("Or place candidates.jsonl in the project directory.")
            sys.exit(1)

    if not candidates_path.exists():
        print(f"Error: {candidates_path} not found")
        sys.exit(1)

    print(f"Loading candidates from {candidates_path}...")
    start_time = time.time()

    candidates = []
    candidate_ids = []
    candidate_texts = []

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            candidates.append(cand)
            candidate_ids.append(cand["candidate_id"])
            candidate_texts.append(build_candidate_text(cand))

    print(f"Loaded {len(candidates)} candidates in {time.time() - start_time:.1f}s")

    # Load sentence-transformers model
    print(f"Loading embedding model: {MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    # Encode candidates
    print(f"Encoding {len(candidate_texts)} candidates (batch_size={BATCH_SIZE})...")
    start_time = time.time()
    embeddings = model.encode(
        candidate_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # For cosine similarity via dot product
    )
    print(f"Encoded in {time.time() - start_time:.1f}s")

    # Encode JD
    print("Encoding Job Description...")
    jd_embedding = model.encode(
        [JD_TEXT],
        normalize_embeddings=True,
    )[0]

    # Save to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings_path = OUTPUT_DIR / "embeddings.npy"
    ids_path = OUTPUT_DIR / "candidate_ids.json"
    jd_path = OUTPUT_DIR / "jd_embedding.npy"

    np.save(embeddings_path, embeddings.astype(np.float32))
    with open(ids_path, "w") as f:
        json.dump(candidate_ids, f)
    np.save(jd_path, jd_embedding.astype(np.float32))

    print(f"\nSaved to {OUTPUT_DIR}/:")
    print(f"  embeddings.npy    — {embeddings.shape} ({os.path.getsize(embeddings_path) / 1024 / 1024:.1f} MB)")
    print(f"  candidate_ids.json — {len(candidate_ids)} IDs")
    print(f"  jd_embedding.npy  — {jd_embedding.shape}")
    print("\nPre-computation complete!")


if __name__ == "__main__":
    main()
