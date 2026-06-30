#!/usr/bin/env python3
"""
Redrob AI Candidate Ranker — Streamlit Web App (Sandbox)
=========================================================
A demo interface where organizers can:
  1. View the Job Description
  2. Upload a small JSON of candidates
  3. Run the ranker end-to-end
  4. See ranked results with scores and reasoning
  5. Download the output CSV

This is the "sandbox" required by submission_spec.md Section 10.5.
"""

import csv
import io
import json
import time

import streamlit as st
import pandas as pd

from scorer.title_scorer import score_title
from scorer.skill_scorer import score_skills
from scorer.experience_scorer import score_experience
from scorer.location_scorer import score_location
from scorer.signal_scorer import score_signals
from scorer.honeypot_detector import is_honeypot
from scorer.consulting_checker import check_consulting_career
from scorer.reasoning_generator import generate_reasoning

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Redrob AI Candidate Ranker",
    layout="wide",
)

# Custom CSS for a professional, premium dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Force Main Background */
    [data-testid="stAppViewContainer"], .stApp {
        background-color: #0B0F19 !important;
        background-image: radial-gradient(circle at 50% 0%, #1a2333 0%, transparent 70%);
        color: #E2E8F0 !important;
    }
    
    /* Force Sidebar Background (if any) */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    .stMarkdown p {
        color: #CBD5E1 !important;
    }
    
    /* Tabs styling */
    [data-testid="stTabs"] button {
        color: #94A3B8 !important;
        font-weight: 500 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #3B82F6 !important;
    }
    
    /* Metrics styling */
    [data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 1rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }
    
    /* Info boxes (st.info) */
    [data-testid="stAlert"] {
        background-color: rgba(59, 130, 246, 0.15) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        color: #F8FAFC !important;
    }
    [data-testid="stAlert"] * {
        color: #F8FAFC !important;
    }
    
    /* File Uploader Container & text */
    [data-testid="stFileUploadDropzone"] {
        background-color: #1E293B !important;
        border: 2px dashed #334155 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3B82F6 !important;
        background-color: #0F172A !important;
    }
    .stFileUploader label {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploadDropzone"] div {
        color: #E2E8F0 !important;
    }
    [data-testid="stFileUploadDropzone"] small {
        color: #94A3B8 !important;
    }
    /* Fix opacity on the upload button inside the dropzone */
    [data-testid="stFileUploadDropzone"] button {
        opacity: 1 !important;
        background-color: #334155 !important;
        color: #ffffff !important;
    }
    
    /* Styled Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%) !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {background: transparent !important;}
    
    /* Custom divider */
    hr {
        border-top: 1px solid #1E293B !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# JD text
# ---------------------------------------------------------------------------

JD_SUMMARY = """
**Role:** Senior AI Engineer — Founding Team  
**Company:** Redrob AI (Series A)  
**Location:** Pune/Noida, India (Hybrid)  
**Experience:** 5-9 years  

**Must-Have:** Embeddings-based retrieval, vector databases, strong Python,
ranking evaluation frameworks (NDCG, MRR, MAP)

**Nice-to-Have:** LLM fine-tuning (LoRA), learning-to-rank, HR-tech experience

**Disqualifiers:** Pure research (no production), entire career at consulting firms,
non-India without visa sponsorship, title-chasers
"""

# Score weights
WEIGHTS = {
    "title": 0.38,
    "skill": 0.28,
    "experience": 0.14,
    "location": 0.14,
    "education": 0.06,
}


def score_education(candidate: dict) -> float:
    """Score education relevance."""
    education = candidate.get("education", [])
    if not education:
        return 0.3

    best_score = 0.0
    for edu in education:
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()
        tier = edu.get("tier", "unknown")

        if any(f in field for f in [
            "computer science", "machine learning", "artificial intelligence",
            "data science", "information technology", "software", "statistics",
        ]):
            field_score = 1.0
        elif any(f in field for f in ["mathematics", "physics", "electronics", "electrical"]):
            field_score = 0.7
        elif "engineering" in field:
            field_score = 0.4
        else:
            field_score = 0.15

        if "ph.d" in degree or "phd" in degree:
            degree_score = 1.0
        elif any(d in degree for d in ["m.tech", "m.s", "m.sc", "m.e.", "master"]):
            degree_score = 0.8
        elif any(d in degree for d in ["b.tech", "b.e.", "b.sc", "b.s", "bachelor"]):
            degree_score = 0.6
        else:
            degree_score = 0.4

        tier_scores = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.45, "tier_4": 0.25, "unknown": 0.35}
        tier_score = tier_scores.get(tier, 0.35)

        edu_score = 0.45 * field_score + 0.25 * degree_score + 0.30 * tier_score
        best_score = max(best_score, edu_score)

    return round(best_score, 4)


def rank_candidates(candidates: list[dict]) -> list[dict]:
    """Run the full ranking pipeline on a list of candidates."""
    scored = []

    for candidate in candidates:
        cid = candidate.get("candidate_id", "")
        honeypot = is_honeypot(candidate)

        title_sc = score_title(candidate)
        skill_sc = score_skills(candidate)
        exp_sc = score_experience(candidate)
        loc_sc = score_location(candidate)
        edu_sc = score_education(candidate)
        signal_mult = score_signals(candidate)
        consulting_mult = check_consulting_career(candidate)

        base_score = (
            WEIGHTS["title"] * title_sc +
            WEIGHTS["skill"] * skill_sc +
            WEIGHTS["experience"] * exp_sc +
            WEIGHTS["location"] * loc_sc +
            WEIGHTS["education"] * edu_sc
        )

        final_score = base_score * signal_mult * consulting_mult
        if honeypot:
            final_score *= 0.01

        scored.append({
            "candidate": candidate,
            "candidate_id": cid,
            "final_score": final_score,
            "components": {
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

    # Sort by score descending, then by candidate_id for tie-breaking
    scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    # Take top 100 (or all if fewer)
    top_k = scored[:min(100, len(scored))]

    # Generate reasoning
    results = []
    for rank_idx, entry in enumerate(top_k):
        rank = rank_idx + 1
        reasoning = generate_reasoning(
            entry["candidate"], rank, entry["final_score"], entry["components"]
        )
        results.append({
            "candidate_id": entry["candidate_id"],
            "rank": rank,
            "score": round(entry["final_score"], 4),
            "reasoning": reasoning,
            "title": entry["candidate"]["profile"].get("current_title", ""),
            "company": entry["candidate"]["profile"].get("current_company", ""),
            "years_exp": entry["candidate"]["profile"].get("years_of_experience", 0),
            "location": entry["candidate"]["profile"].get("location", ""),
            "is_honeypot": entry["components"]["is_honeypot"],
            **{f"score_{k}": round(v, 3) for k, v in entry["components"].items()
               if k != "is_honeypot"},
        })

    return results


def results_to_csv(results: list[dict]) -> str:
    """Convert results to CSV string."""
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["candidate_id", "rank", "score", "reasoning"],
    )
    writer.writeheader()
    for r in results:
        writer.writerow({
            "candidate_id": r["candidate_id"],
            "rank": r["rank"],
            "score": r["score"],
            "reasoning": r["reasoning"],
        })
    return output.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.markdown("<h1 style='text-align: center; color: #58a6ff !important; margin-bottom: 0;'>Redrob AI Candidate Ranker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 2rem;'>Intelligent Candidate Discovery & Ranking — Enterprise Grade</p>", unsafe_allow_html=True)

tab_rank, tab_config = st.tabs(["Ranking Engine", "Job Configuration"])

with tab_config:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Job Description")
        st.info(JD_SUMMARY)
    with col2:
        st.markdown("### Scoring Weights")
        st.markdown("The system uses these weights to compute the base score.")
        w_cols = st.columns(2)
        idx = 0
        for key, weight in WEIGHTS.items():
            w_cols[idx % 2].metric(label=key.title(), value=f"{weight:.0%}")
            idx += 1

with tab_rank:
    st.markdown("### Upload Candidates")
    st.markdown(
        "Upload a JSON file containing candidate records "
        "(array of objects, or JSONL with one object per line)."
    )
    
    uploaded_file = st.file_uploader(
        "Choose a candidates file",
        type=["json", "jsonl"],
        help="JSON array or JSONL format. Max ~100 candidates for sandbox demo.",
    )

if uploaded_file is not None:
    # Parse the file
    try:
        content = uploaded_file.read().decode("utf-8")

        # Try JSON array first
        try:
            candidates = json.loads(content)
            if isinstance(candidates, dict):
                candidates = [candidates]
        except json.JSONDecodeError:
            # Try JSONL
            candidates = []
            for line in content.strip().split("\n"):
                if line.strip():
                    candidates.append(json.loads(line))

        st.success(f"Loaded **{len(candidates)}** candidates")

        if st.button("Run Ranker", type="primary", use_container_width=True):
            start = time.time()

            with st.spinner("Scoring and ranking candidates..."):
                results = rank_candidates(candidates)

            elapsed = time.time() - start
            st.success(f"Ranked {len(results)} candidates in {elapsed:.2f}s")

            # Results table
            st.markdown("### Ranked Results")

            df = pd.DataFrame(results)
            display_cols = [
                "rank", "candidate_id", "score", "title", "company",
                "years_exp", "location", "reasoning",
            ]
            display_cols = [c for c in display_cols if c in df.columns]

            st.dataframe(
                df[display_cols],
                use_container_width=True,
                height=500,
                hide_index=True,
            )

            # Score breakdown
            with st.expander("Score Breakdown"):
                score_cols = [c for c in df.columns if c.startswith("score_")]
                if score_cols:
                    breakdown_df = df[["rank", "candidate_id", "score"] + score_cols + ["is_honeypot"]]
                    st.dataframe(breakdown_df, use_container_width=True)

            # Honeypot warning
            honeypot_count = sum(1 for r in results if r.get("is_honeypot", False))
            if honeypot_count > 0:
                st.warning(f"{honeypot_count} potential honeypots detected and down-weighted")

            # Download CSV
            csv_data = results_to_csv(results)
            st.download_button(
                label="Download Submission CSV",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv",
                type="primary",
            )

    except Exception as e:
        st.error(f"Error parsing file: {e}")
        st.exception(e)

else:
    st.info("Upload a candidates JSON/JSONL file to get started")

    # Show a demo with embedded sample
    st.markdown("---")
    st.markdown("### Quick Demo")
    st.markdown(
        "No file? Click below to run on a built-in sample of 5 candidates."
    )

    if st.button("Run Demo with Sample Data"):
        sample_candidates = [
            {
                "candidate_id": "DEMO_001",
                "profile": {
                    "anonymized_name": "Demo AI Engineer",
                    "headline": "Senior ML Engineer | NLP, Embeddings, Search",
                    "summary": "7 years building production ML systems including ranking and retrieval. "
                               "Built a semantic search engine using FAISS and sentence-transformers.",
                    "location": "Pune, Maharashtra",
                    "country": "India",
                    "years_of_experience": 7.0,
                    "current_title": "Senior ML Engineer",
                    "current_company": "AI Startup",
                    "current_company_size": "51-200",
                    "current_industry": "Technology",
                },
                "career_history": [{
                    "company": "AI Startup",
                    "title": "Senior ML Engineer",
                    "start_date": "2022-01-01",
                    "end_date": None,
                    "duration_months": 54,
                    "is_current": True,
                    "industry": "Technology",
                    "company_size": "51-200",
                    "description": "Built production ranking system with hybrid search using "
                                   "embeddings and BM25. Deployed vector search with FAISS.",
                }],
                "education": [{
                    "institution": "IIT Bombay",
                    "degree": "B.Tech",
                    "field_of_study": "Computer Science",
                    "start_year": 2013, "end_year": 2017,
                    "grade": "8.5 CGPA", "tier": "tier_1",
                }],
                "skills": [
                    {"name": "Python", "proficiency": "expert", "endorsements": 30, "duration_months": 72},
                    {"name": "PyTorch", "proficiency": "advanced", "endorsements": 20, "duration_months": 48},
                    {"name": "NLP", "proficiency": "advanced", "endorsements": 15, "duration_months": 36},
                    {"name": "FAISS", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
                    {"name": "Elasticsearch", "proficiency": "intermediate", "endorsements": 8, "duration_months": 30},
                ],
                "certifications": [],
                "languages": [{"language": "English", "proficiency": "professional"}],
                "redrob_signals": {
                    "profile_completeness_score": 92,
                    "signup_date": "2025-01-15", "last_active_date": "2026-06-20",
                    "open_to_work_flag": True,
                    "profile_views_received_30d": 45, "applications_submitted_30d": 3,
                    "recruiter_response_rate": 0.82,
                    "avg_response_time_hours": 4.5,
                    "skill_assessment_scores": {"NLP": 78, "Python": 91},
                    "connection_count": 420, "endorsements_received": 83,
                    "notice_period_days": 30,
                    "expected_salary_range_inr_lpa": {"min": 25, "max": 40},
                    "preferred_work_mode": "hybrid",
                    "willing_to_relocate": True,
                    "github_activity_score": 72,
                    "search_appearance_30d": 180, "saved_by_recruiters_30d": 22,
                    "interview_completion_rate": 0.95,
                    "offer_acceptance_rate": 0.8,
                    "verified_email": True, "verified_phone": True,
                    "linkedin_connected": True,
                },
            },
            {
                "candidate_id": "DEMO_002",
                "profile": {
                    "anonymized_name": "Demo HR Person",
                    "headline": "HR Manager | People Operations",
                    "summary": "HR professional with experience in talent management. "
                               "Interested in AI tools for HR automation.",
                    "location": "Mumbai, Maharashtra",
                    "country": "India",
                    "years_of_experience": 8.0,
                    "current_title": "HR Manager",
                    "current_company": "TCS",
                    "current_company_size": "10001+",
                    "current_industry": "IT Services",
                },
                "career_history": [{
                    "company": "TCS",
                    "title": "HR Manager",
                    "start_date": "2018-06-01",
                    "end_date": None,
                    "duration_months": 96,
                    "is_current": True,
                    "industry": "IT Services",
                    "company_size": "10001+",
                    "description": "Managing HR operations and talent acquisition.",
                }],
                "education": [{
                    "institution": "Local College",
                    "degree": "MBA",
                    "field_of_study": "Human Resources",
                    "start_year": 2014, "end_year": 2016,
                    "grade": "7.5 CGPA", "tier": "tier_3",
                }],
                "skills": [
                    {"name": "Python", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                    {"name": "Machine Learning", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                    {"name": "NLP", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
                    {"name": "PyTorch", "proficiency": "advanced", "endorsements": 0, "duration_months": 0},
                    {"name": "TensorFlow", "proficiency": "advanced", "endorsements": 0, "duration_months": 0},
                ],
                "certifications": [],
                "languages": [{"language": "English", "proficiency": "professional"}],
                "redrob_signals": {
                    "profile_completeness_score": 75,
                    "signup_date": "2025-05-01", "last_active_date": "2026-06-15",
                    "open_to_work_flag": True,
                    "profile_views_received_30d": 10, "applications_submitted_30d": 5,
                    "recruiter_response_rate": 0.60,
                    "avg_response_time_hours": 24,
                    "skill_assessment_scores": {},
                    "connection_count": 200, "endorsements_received": 15,
                    "notice_period_days": 60,
                    "expected_salary_range_inr_lpa": {"min": 12, "max": 18},
                    "preferred_work_mode": "hybrid",
                    "willing_to_relocate": True,
                    "github_activity_score": -1,
                    "search_appearance_30d": 50, "saved_by_recruiters_30d": 2,
                    "interview_completion_rate": 0.70,
                    "offer_acceptance_rate": 0.5,
                    "verified_email": True, "verified_phone": True,
                    "linkedin_connected": False,
                },
            },
        ]

        with st.spinner("Running demo..."):
            results = rank_candidates(sample_candidates)

        st.markdown("### Demo Results")
        st.markdown(
            "Notice how the **ML Engineer** ranks far above the "
            "**HR Manager** — even though the HR Manager listed many AI skills. "
            "The system detects the keyword-stuffing (zero endorsements and zero duration)."
        )

        df = pd.DataFrame(results)
        display_cols = ["rank", "candidate_id", "score", "title", "company", "reasoning"]
        st.dataframe(df[display_cols], use_container_width=True)

# Footer
st.markdown("---")
st.caption("Built for the Redrob Intelligent Candidate Discovery & Ranking Challenge 2026")
