# 🎯 Redrob AI Candidate Ranker

Intelligent Candidate Discovery & Ranking system for the Redrob Hackathon 2026.

## Architecture

```
Input: candidates.jsonl (100K profiles) + job_description.md
                        │
                        ▼
              ┌─────────────────────┐
              │   PRE-COMPUTATION   │
              │  (offline, no cap)  │
              │                     │
              │ sentence-transformers│
              │ → embeddings.npy    │
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │   RANKING PIPELINE  │
              │   (< 5 min, CPU)    │
              │                     │
              │ 1. Semantic scores  │
              │ 2. Title relevance  │
              │ 3. Skill matching   │
              │ 4. Experience fit   │
              │ 5. Location check   │
              │ 6. Education signal │
              │ 7. Behavioral mult  │
              │ 8. Consulting check │
              │ 9. Honeypot filter  │
              │                     │
              │ → Weighted combine  │
              │ → Top 100 + reason  │
              └────────┬────────────┘
                       │
                       ▼
              ┌─────────────────────┐
              │   submission.csv    │
              │  100 ranked cands   │
              │  + reasoning        │
              └─────────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pre-compute embeddings (offline, ~30-60 min)

```bash
python precompute.py path/to/candidates.jsonl
```

This generates `data/embeddings.npy`, `data/candidate_ids.json`, and `data/jd_embedding.npy`.

### 3. Run the ranker (< 5 min on CPU)

```bash
python rank.py --candidates path/to/candidates.jsonl --out submission.csv
```

### 4. Validate the submission

```bash
python validate_submission.py submission.csv
```

### 5. Run the web app (sandbox demo)

```bash
streamlit run app.py
```

## Project Structure

```
redrob-ranker/
├── rank.py                      ← Main ranking pipeline (produces CSV)
├── precompute.py                ← Offline embedding generation
├── app.py                       ← Streamlit web app (sandbox)
├── scorer/
│   ├── __init__.py
│   ├── title_scorer.py          ← Title & career relevance (most important)
│   ├── skill_scorer.py          ← Skill matching with trust scoring
│   ├── experience_scorer.py     ← Experience fit (5-9yr sweet spot)
│   ├── location_scorer.py       ← Location, notice period, work mode
│   ├── signal_scorer.py         ← 23 behavioral signal multiplier
│   ├── honeypot_detector.py     ← Impossible profile detection
│   ├── consulting_checker.py    ← Consulting-firm career penalty
│   └── reasoning_generator.py   ← Per-candidate reasoning text
├── data/                        ← Pre-computed artifacts
│   ├── embeddings.npy
│   ├── candidate_ids.json
│   └── jd_embedding.npy
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

## Scoring Formula

```
final_score = (
    0.20 × semantic_similarity +     # Embedding cosine similarity
    0.30 × title_career_relevance +  # Is this person actually in AI/ML?
    0.20 × skill_match +             # Do they have the right skills?
    0.12 × experience_fit +          # Right experience level?
    0.12 × location_logistics +      # Can we hire them?
    0.06 × education_signal          # Education relevance
) × behavioral_multiplier            # Are they active/responsive?
  × consulting_penalty               # All-consulting career?
  × honeypot_filter                  # Impossible profile?
```

## Key Design Decisions

1. **Title > Keywords**: The system weights title and career history much higher than listed skills, because the dataset contains keyword-stuffed profiles (HR Managers with AI skill lists).

2. **Trust scoring**: Skills are weighted by endorsements and duration — a skill with 20 endorsements and 36 months of use is treated differently from one with 0 endorsements and 0 months.

3. **Behavioral signals as multiplier**: Active, responsive candidates get boosted; inactive ghosts get penalized. This reflects real-world recruitability.

4. **Honeypot detection**: Six different impossibility checks (timeline contradictions, expert-with-zero-duration, etc.) catch the ~80 synthetic trap candidates.

5. **Semantic + Rule-based hybrid**: Embeddings capture nuanced meaning; rules enforce hard constraints from the JD.

## Compute Environment

- Platform: Windows (local development)
- CPU only — no GPU used during ranking
- No network calls during ranking
- Pre-computation (embeddings) runs offline with no time limit
- Ranking step completes in < 5 minutes on 16GB RAM

## AI Tools Used

- Claude (architecture discussion, code review)
- GitHub Copilot (autocomplete assistance)
