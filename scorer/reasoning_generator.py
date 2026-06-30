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


def _get_matching_skill_names(candidate: dict, target_skills: set) -> list[str]:
    """Get names of skills that match the target set."""
    skills = candidate.get("skills", [])
    matched = []
    for skill in skills:
        name = skill.get("name", "").strip().lower()
        for target in target_skills:
            if target in name or name in target:
                matched.append(skill.get("name", ""))
                break
    return matched


AI_CORE_SKILLS = {
    "pytorch", "tensorflow", "nlp", "machine learning", "deep learning",
    "embeddings", "transformers", "huggingface", "faiss", "pinecone",
    "milvus", "weaviate", "python", "scikit-learn", "xgboost",
    "recommendation", "ranking", "information retrieval", "rag",
    "lora", "fine-tuning", "llm", "bert", "gpt", "vector",
    "semantic search", "neural network", "feature engineering",
    "natural language processing",
}

CONCERN_KEYWORDS = {
    "hr manager", "accountant", "marketing manager", "content writer",
    "graphic designer", "sales executive", "civil engineer",
    "mechanical engineer", "customer support", "operations manager",
}


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

    # Get matching skills
    matching_skills = _get_matching_skill_names(candidate, AI_CORE_SKILLS)
    top_skills = matching_skills[:4]  # Show up to 4

    # Key signals
    response_rate = signals.get("recruiter_response_rate", 0)
    notice_days = signals.get("notice_period_days", 90)
    github_score = signals.get("github_activity_score", -1)
    open_to_work = signals.get("open_to_work_flag", False)

    # Build reasoning parts
    parts = []
    concerns = []

    # --- Strength statement ---
    if rank <= 10:
        # Top 10 — confident tone
        if top_skills:
            parts.append(f"{title} at {company} with {years:.1f} years; "
                         f"strong match on {', '.join(top_skills[:3])}")
        else:
            parts.append(f"{title} at {company} with {years:.1f} years; "
                         f"strong career-history alignment with AI/ML systems work")
    elif rank <= 30:
        # Top 30 — solid but specific
        if top_skills:
            parts.append(f"{title} with {years:.1f} yrs experience; "
                         f"relevant skills include {', '.join(top_skills[:3])}")
        else:
            parts.append(f"{title} with {years:.1f} yrs; "
                         f"career history shows relevant technical background")
    elif rank <= 60:
        # Mid-range — balanced
        if top_skills:
            parts.append(f"{years:.1f} yrs as {title}; "
                         f"partial skill alignment ({', '.join(top_skills[:2])})")
        else:
            parts.append(f"{years:.1f} yrs as {title}; "
                         f"some relevant adjacent experience")
    else:
        # Bottom 40 — cautious, note gaps
        if top_skills:
            parts.append(f"{title} ({years:.1f} yrs); "
                         f"limited but present skill overlap ({', '.join(top_skills[:2])})")
        else:
            parts.append(f"{title} ({years:.1f} yrs); "
                         f"weak direct skill match but included based on secondary signals")

    # --- Location/logistics note ---
    if country.lower() == "india":
        if any(c in location.lower() for c in ["pune", "noida"]):
            parts.append("preferred location")
        elif any(c in location.lower() for c in ["bangalore", "bengaluru", "hyderabad",
                                                    "mumbai", "delhi", "gurgaon", "chennai"]):
            parts.append(f"{location}-based")
    else:
        concerns.append(f"based in {country}")

    # --- Behavioral strengths/concerns ---
    if response_rate >= 0.7:
        parts.append(f"high recruiter engagement ({response_rate:.0%} response rate)")
    elif response_rate < 0.2:
        concerns.append(f"low response rate ({response_rate:.0%})")

    if notice_days <= 30:
        parts.append("available with short notice")
    elif notice_days >= 120:
        concerns.append(f"{notice_days}-day notice period")

    if github_score >= 50:
        parts.append(f"active GitHub contributor (score: {github_score:.0f})")

    # --- Experience concerns ---
    if years < 4:
        concerns.append("below the 5-9 year experience range")
    elif years > 12:
        concerns.append("above the target experience range; may be overqualified")

    # --- Title concern ---
    if title.lower() in CONCERN_KEYWORDS or any(
        kw in title.lower() for kw in CONCERN_KEYWORDS
    ):
        concerns.append(f"title ({title}) is not directly AI/ML-related")

    # --- Assemble ---
    reasoning = "; ".join(parts)

    if concerns and rank > 20:
        reasoning += ". Concerns: " + "; ".join(concerns)
    elif concerns and rank <= 20:
        reasoning += ". Note: " + "; ".join(concerns[:1])

    # Ensure it's not too long (keep to ~2 sentences)
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    return reasoning
