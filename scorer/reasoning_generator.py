"""
Reasoning Generator
====================
Generates specific, honest, 1-2 sentence reasoning for each top-100 candidate.

Requirements (from submission_spec.md Stage 4):
  - Reference specific facts from the candidate's profile
  - Connect to specific JD requirements
  - Acknowledge gaps/concerns honestly
  - No hallucination (only reference data that exists)
  - Each reasoning must be unique (not templated)
  - Tone matches rank (top = confident, bottom = cautious)
"""


# Map candidate skills to specific JD requirements for explicit connection
JD_SKILL_MAP = {
    # Must-Have: Embeddings-based retrieval
    "faiss": "JD must-have: embeddings-based retrieval",
    "pinecone": "JD must-have: vector databases",
    "milvus": "JD must-have: vector databases",
    "weaviate": "JD must-have: vector databases",
    "embeddings": "JD must-have: embeddings-based retrieval",
    "vector": "JD must-have: vector databases",
    "semantic search": "JD must-have: embeddings-based retrieval",
    # Must-Have: Strong Python
    "python": "JD must-have: strong Python",
    # Must-Have: Ranking evaluation
    "information retrieval": "JD must-have: ranking evaluation",
    "ranking": "JD must-have: ranking evaluation",
    "recommendation": "JD must-have: ranking/recommendation systems",
    "ndcg": "JD must-have: ranking evaluation frameworks",
    # Nice-to-Have
    "lora": "JD nice-to-have: LLM fine-tuning",
    "fine-tuning": "JD nice-to-have: LLM fine-tuning",
    "llm": "JD nice-to-have: LLM experience",
    "learning-to-rank": "JD nice-to-have: learning-to-rank",
    "learning to rank": "JD nice-to-have: learning-to-rank",
    # Core AI/ML
    "nlp": "directly relevant NLP expertise",
    "natural language processing": "directly relevant NLP expertise",
    "pytorch": "production ML framework experience",
    "tensorflow": "production ML framework experience",
    "deep learning": "core deep learning background",
    "machine learning": "core ML background",
    "transformers": "transformer architecture experience",
    "huggingface": "modern NLP tooling experience",
    "bert": "transformer model experience",
    "gpt": "LLM experience",
    "scikit-learn": "classical ML toolkit proficiency",
    "xgboost": "ML modeling experience",
    "rag": "JD-relevant RAG pipeline experience",
    "neural network": "deep learning fundamentals",
    "feature engineering": "applied ML engineering",
}

CONCERN_TITLES = {
    "hr manager", "accountant", "marketing manager", "content writer",
    "graphic designer", "sales executive", "civil engineer",
    "mechanical engineer", "customer support", "operations manager",
}


def _get_jd_connections(candidate: dict) -> list[str]:
    """Map candidate skills to specific JD requirements."""
    skills = candidate.get("skills", [])
    connections = []
    seen_connections = set()

    for skill in skills:
        name = skill.get("name", "").strip().lower()
        for keyword, jd_reason in JD_SKILL_MAP.items():
            if keyword in name or name in keyword:
                if jd_reason not in seen_connections:
                    connections.append((skill.get("name", ""), jd_reason))
                    seen_connections.add(jd_reason)
                break

    return connections


def generate_reasoning(candidate: dict, rank: int, score: float,
                       component_scores: dict) -> str:
    """
    Generate a specific, honest reasoning string for a ranked candidate.

    Args:
        candidate: Full candidate record
        rank: 1-100
        component_scores: dict with keys like 'title', 'skill', 'experience',
                          'location', 'signal', 'semantic'
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    years = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")

    # Get JD-connected skills
    jd_connections = _get_jd_connections(candidate)

    # Key signals
    response_rate = signals.get("recruiter_response_rate", 0)
    notice_days = signals.get("notice_period_days", 90)
    github_score = signals.get("github_activity_score", -1)

    # Build reasoning parts
    parts = []
    concerns = []

    # --- Strength statement with JD connection ---
    if rank <= 10:
        # Top 10 — confident, JD-specific
        if jd_connections:
            skill_refs = [f"{s} ({r})" for s, r in jd_connections[:2]]
            parts.append(f"{title} at {company} with {years:.1f} years; "
                         f"directly relevant: {', '.join(skill_refs)}")
        else:
            parts.append(f"{title} at {company} with {years:.1f} years; "
                         f"strong career alignment with the Senior AI Engineer role")
    elif rank <= 30:
        # Top 30 — solid, connect to JD
        if jd_connections:
            skill_refs = [f"{s} ({r})" for s, r in jd_connections[:2]]
            parts.append(f"{title} with {years:.1f} yrs; "
                         f"relevant to JD: {', '.join(skill_refs)}")
        else:
            parts.append(f"{title} with {years:.1f} yrs; "
                         f"career history shows adjacent AI/ML systems work")
    elif rank <= 60:
        # Mid-range — balanced
        if jd_connections:
            skill_refs = [f"{s}" for s, r in jd_connections[:2]]
            parts.append(f"{years:.1f} yrs as {title}; "
                         f"partial JD alignment via {', '.join(skill_refs)}")
        else:
            parts.append(f"{years:.1f} yrs as {title}; "
                         f"some adjacent experience but limited direct JD skill match")
    else:
        # Bottom 40 — cautious, note gaps
        if jd_connections:
            skill_refs = [f"{s}" for s, r in jd_connections[:2]]
            parts.append(f"{title} ({years:.1f} yrs); "
                         f"limited overlap with JD ({', '.join(skill_refs)} only)")
        else:
            parts.append(f"{title} ({years:.1f} yrs); "
                         f"weak match against core JD requirements; "
                         f"included based on secondary signals")

    # --- Location/logistics note ---
    if country.lower() == "india":
        if any(c in location.lower() for c in ["pune", "noida"]):
            parts.append("preferred location (Pune/Noida)")
        elif any(c in location.lower() for c in ["bangalore", "bengaluru", "hyderabad",
                                                    "mumbai", "delhi", "gurgaon", "chennai"]):
            parts.append(f"India-based ({location})")
    else:
        concerns.append(f"based outside India ({country})")

    # --- Behavioral strengths/concerns ---
    if response_rate >= 0.7:
        parts.append(f"high recruiter engagement ({response_rate:.0%} response rate)")
    elif response_rate < 0.2 and response_rate > 0:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")

    if notice_days <= 30:
        parts.append("available with short notice")
    elif notice_days >= 120:
        concerns.append(f"long notice period ({notice_days} days)")

    if github_score >= 50:
        parts.append(f"active GitHub contributor (score: {github_score:.0f})")

    # --- Experience concerns ---
    if years < 4:
        concerns.append("below the JD's 5-9 year experience requirement")
    elif years > 12:
        concerns.append("above the JD's target experience range; may be overqualified")

    # --- Title concern ---
    title_lower = title.lower()
    if any(kw in title_lower for kw in CONCERN_TITLES):
        concerns.append(f"title ({title}) not directly AI/ML-related per JD")

    # --- Assemble ---
    reasoning = "; ".join(parts)

    if concerns and rank > 20:
        reasoning += ". Concerns: " + "; ".join(concerns)
    elif concerns and rank <= 20:
        reasoning += ". Note: " + "; ".join(concerns[:1])

    # Ensure it's not too long (keep to ~2 sentences)
    if len(reasoning) > 350:
        reasoning = reasoning[:347] + "..."

    return reasoning

