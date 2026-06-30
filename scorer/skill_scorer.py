"""
Skill Match Scorer
==================
Scores candidates based on how well their listed skills match the JD requirements.

Key design decisions:
  - Don't just count keyword matches — weight by proficiency, endorsements, duration
  - Trust multiplier: skills with endorsements AND real duration are more believable
  - Cross-check: candidates with many AI skills but irrelevant titles get penalized
    (handled by the title_scorer, but we also detect "keyword stuffing" here)
"""


# ---------------------------------------------------------------------------
# Skill categories for the Senior AI Engineer JD
# ---------------------------------------------------------------------------

# Must-have skills (from JD "things you absolutely need")
MUST_HAVE_SKILLS = {
    # Embeddings & Retrieval
    "sentence-transformers", "embeddings", "faiss", "pinecone",
    "milvus", "weaviate", "qdrant", "opensearch", "elasticsearch",
    "vector search", "vector database", "semantic search",
    "information retrieval", "search systems",

    # Core ML/AI
    "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "nlp", "natural language processing", "transformers",
    "huggingface", "hugging face", "deep learning",
    "machine learning", "neural networks",

    # Python
    "python",

    # Ranking & Evaluation
    "ranking", "learning to rank", "recommendation systems",
    "ndcg", "mrr", "map", "a/b testing", "evaluation",
    "bm25", "hybrid search",
}

# Nice-to-have skills (from JD)
NICE_TO_HAVE_SKILLS = {
    # LLM fine-tuning
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
    "llm", "large language models", "gpt", "bert",
    "rag", "retrieval augmented generation",

    # Learning to rank
    "xgboost", "lightgbm", "catboost", "gradient boosting",

    # Infrastructure
    "docker", "kubernetes", "aws", "gcp", "azure",
    "mlflow", "weights & biases", "wandb", "mlops",
    "kubeflow", "sagemaker", "bentoml", "ray",

    # Data engineering (adjacent)
    "spark", "airflow", "kafka", "sql", "databricks",
    "data pipelines", "etl",

    # Other ML
    "feature engineering", "statistical modeling",
    "computer vision", "image classification",
    "object detection", "gans",
}

# Proficiency weights
PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2,
}


def _normalize_skill_name(name: str) -> str:
    """Lowercase and strip whitespace for matching."""
    return name.strip().lower()


def _skill_trust_score(skill: dict) -> float:
    """
    How much should we trust this skill claim?
    High endorsements + long duration = very trustworthy.
    Zero of both = possibly keyword-stuffed.
    """
    endorsements = skill.get("endorsements", 0)
    duration_months = skill.get("duration_months", 0)

    # Base trust
    trust = 0.3

    # Duration adds trust
    if duration_months >= 24:
        trust += 0.35
    elif duration_months >= 12:
        trust += 0.25
    elif duration_months >= 6:
        trust += 0.15
    elif duration_months > 0:
        trust += 0.05

    # Endorsements add trust
    if endorsements >= 20:
        trust += 0.35
    elif endorsements >= 10:
        trust += 0.25
    elif endorsements >= 3:
        trust += 0.15
    elif endorsements > 0:
        trust += 0.05

    return min(1.0, trust)


def score_skills(candidate: dict) -> float:
    """
    Score how well the candidate's skills match the JD.
    Returns a float between 0.0 and 1.0.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0

    must_have_matched = 0
    must_have_weighted_score = 0.0
    nice_to_have_matched = 0
    nice_to_have_weighted_score = 0.0
    total_trust = 0.0

    for skill in skills:
        name = _normalize_skill_name(skill.get("name", ""))
        proficiency = skill.get("proficiency", "beginner")
        prof_weight = PROFICIENCY_WEIGHTS.get(proficiency, 0.2)
        trust = _skill_trust_score(skill)
        total_trust += trust

        # Check must-have
        is_must_have = any(
            mh in name or name in mh
            for mh in MUST_HAVE_SKILLS
        )

        # Check nice-to-have
        is_nice_to_have = any(
            nth in name or name in nth
            for nth in NICE_TO_HAVE_SKILLS
        )

        if is_must_have:
            must_have_matched += 1
            must_have_weighted_score += prof_weight * trust
        elif is_nice_to_have:
            nice_to_have_matched += 1
            nice_to_have_weighted_score += prof_weight * trust

    # --- Score calculation ---

    # Must-have coverage (how many of the key skill areas are covered)
    # We don't expect all 30+ synonyms, but at least 4-5 key areas
    must_have_score = min(1.0, must_have_weighted_score / 3.0)

    # Nice-to-have bonus
    nice_to_have_score = min(1.0, nice_to_have_weighted_score / 2.0)

    # Average trust (keyword-stuffing detection)
    avg_trust = total_trust / len(skills) if skills else 0.0

    # Keyword stuffing penalty: too many skills with low trust
    keyword_stuff_penalty = 1.0
    if len(skills) > 12 and avg_trust < 0.4:
        keyword_stuff_penalty = 0.5  # Suspicious — many skills, low trust

    # Check skill assessment scores from redrob_signals
    assessment_bonus = 0.0
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        relevant_assessment_scores = []
        for skill_name, score_val in assessments.items():
            norm_name = _normalize_skill_name(skill_name)
            is_relevant = any(
                mh in norm_name or norm_name in mh
                for mh in MUST_HAVE_SKILLS | NICE_TO_HAVE_SKILLS
            )
            if is_relevant:
                relevant_assessment_scores.append(score_val)

        if relevant_assessment_scores:
            avg_assessment = sum(relevant_assessment_scores) / len(relevant_assessment_scores)
            assessment_bonus = min(0.15, avg_assessment / 100.0 * 0.15)

    # Combine
    combined = (
        0.65 * must_have_score +
        0.20 * nice_to_have_score +
        0.15 * avg_trust +
        assessment_bonus
    ) * keyword_stuff_penalty

    return round(min(1.0, combined), 4)
