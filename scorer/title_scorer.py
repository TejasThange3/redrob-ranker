"""
Title & Career Relevance Scorer
================================
The MOST important scorer. Determines whether a candidate is actually
an AI/ML professional vs a keyword-stuffer (e.g., HR Manager listing AI skills).

Two sub-scores:
  1. Title relevance — current title mapped to tiers
  2. Career evidence — scan career_history descriptions for real AI/ML work
"""

import re

# ---------------------------------------------------------------------------
# Title tier mappings (case-insensitive matching)
# ---------------------------------------------------------------------------

# Direct AI/ML roles → highest relevance
TIER_1_PATTERNS = [
    r"\bai\s+engineer\b", r"\bml\s+engineer\b", r"\bmachine\s*learning\s+engineer\b",
    r"\bdata\s+scientist\b", r"\bnlp\s+engineer\b", r"\bresearch\s+engineer\b",
    r"\bapplied\s+scientist\b", r"\bdeep\s+learning\b", r"\bml\s+scientist\b",
    r"\bai\s+architect\b", r"\bmlops\s+engineer\b", r"\bai\s+developer\b",
    r"\bcomputer\s+vision\s+engineer\b", r"\bsearch\s+engineer\b",
    r"\brecommendation\b.*\bengineer\b", r"\branking\s+engineer\b",
    r"\bai\s+researcher\b", r"\bmachine\s*learning\b.*\bdeveloper\b",
]

# Adjacent technical roles → good, transferable
TIER_2_PATTERNS = [
    r"\bsoftware\s+engineer\b", r"\bbackend\s+engineer\b",
    r"\bdata\s+engineer\b", r"\bplatform\s+engineer\b",
    r"\bfull\s*stack\s+(developer|engineer)\b", r"\bdevops\s+engineer\b",
    r"\binfrastructure\s+engineer\b", r"\bcloud\s+engineer\b",
    r"\banalytics\s+engineer\b", r"\bdata\s+analyst\b",
    r"\bsoftware\s+developer\b", r"\bsde\b",
]

# Semi-technical roles → weak fit
TIER_3_PATTERNS = [
    r"\bproduct\s+manager\b", r"\bproject\s+manager\b",
    r"\bbusiness\s+analyst\b", r"\btechnical\s+lead\b",
    r"\btech\s+lead\b", r"\barchitect\b", r"\bcto\b",
    r"\bquality\s+analyst\b", r"\btest\s+engineer\b",
]

# Completely irrelevant roles → near-zero score
IRRELEVANT_PATTERNS = [
    r"\bhr\s+manager\b", r"\bhuman\s+resources\b",
    r"\baccountant\b", r"\bmarketing\s+manager\b",
    r"\bcontent\s+writer\b", r"\bgraphic\s+designer\b",
    r"\bsales\s+executive\b", r"\bcivil\s+engineer\b",
    r"\bmechanical\s+engineer\b", r"\bcustomer\s+support\b",
    r"\boperations\s+manager\b", r"\bteacher\b", r"\blecturer\b",
    r"\bnurse\b", r"\bchef\b", r"\bdriver\b",
    r"\belectrical\s+engineer\b", r"\bchemical\s+engineer\b",
    r"\bfinance\s+manager\b", r"\blegal\b", r"\blawyer\b",
    r"\bdesigner\b(?!.*\bml\b)(?!.*\bai\b)(?!.*\bsystem\b)",
]

# ---------------------------------------------------------------------------
# Career evidence — keywords that indicate real AI/ML production work
# ---------------------------------------------------------------------------

STRONG_AI_EVIDENCE = [
    r"\bembedding[s]?\b", r"\bvector\s+(database|db|search|store)\b",
    r"\bretrieval\b", r"\branking\s+system\b", r"\brecommendation\s+system\b",
    r"\bsearch\s+(engine|system|infrastructure)\b",
    r"\bfine[- ]?tun(e|ed|ing)\b", r"\bmodel\s+(training|deployment|serving)\b",
    r"\bml\s+pipeline\b", r"\bmlops\b", r"\bmodel\s+inference\b",
    r"\btransformer[s]?\b", r"\bbert\b", r"\bgpt\b", r"\bllm\b",
    r"\brag\b", r"\bneural\s+network\b", r"\bdeep\s+learning\b",
    r"\bndcg\b", r"\bprecision\b.*\brecall\b", r"\ba/b\s+test\b",
    r"\bfeature\s+(engineering|store|extraction)\b",
    r"\bpinecone\b", r"\bweaviate\b", r"\bqdrant\b", r"\bmilvus\b",
    r"\bfaiss\b", r"\belasticsearch\b", r"\bopensearch\b",
    r"\bpytorch\b", r"\btensorflow\b", r"\bhugging\s*face\b",
    r"\bsentence[- ]?transformer\b", r"\bsklearn\b", r"\bscikit\b",
    r"\bxgboost\b", r"\blightgbm\b", r"\bcatboost\b",
    r"\bproduction\s+ml\b", r"\breal[- ]?time\s+(inference|prediction)\b",
    r"\bmodel\s+evaluation\b", r"\bonline\s+learning\b",
    r"\bnatural\s+language\s+processing\b", r"\bnlp\b",
]

MODERATE_AI_EVIDENCE = [
    r"\bpython\b", r"\bdata\s+pipeline\b", r"\betl\b",
    r"\bspark\b", r"\bairflow\b", r"\bkafka\b",
    r"\baws\s+sagemaker\b", r"\bkubeflow\b",
    r"\bapi\b.*\b(deploy|serve|endpoint)\b",
    r"\bmicroservice\b", r"\bdocker\b", r"\bkubernetes\b",
    r"\bcloud\b.*\b(gcp|aws|azure)\b",
    r"\bdata\s+(warehouse|lake)\b", r"\bsnowflake\b", r"\bdatabricks\b",
    r"\banalytics\b", r"\bdashboard\b",
]


def _match_tier(text: str, patterns: list[str]) -> bool:
    """Check if any pattern matches the text (case-insensitive)."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _count_evidence(text: str, patterns: list[str]) -> int:
    """Count how many distinct evidence patterns match."""
    count = 0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            count += 1
    return count


def score_title(candidate: dict) -> float:
    """
    Score a candidate's title & career relevance.
    Returns a float between 0.0 and 1.0.
    """
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "").strip()
    headline = profile.get("headline", "").strip()
    title_text = f"{current_title} {headline}"

    # --- Step 1: Title tier ---
    if _match_tier(title_text, TIER_1_PATTERNS):
        title_score = 1.0
    elif _match_tier(title_text, TIER_2_PATTERNS):
        title_score = 0.65
    elif _match_tier(title_text, TIER_3_PATTERNS):
        title_score = 0.25
    elif _match_tier(title_text, IRRELEVANT_PATTERNS):
        title_score = 0.02
    else:
        # Unknown title — default to low-moderate
        title_score = 0.15

    # --- Step 2: Career evidence from descriptions ---
    career_history = candidate.get("career_history", [])
    all_descriptions = " ".join(
        job.get("description", "") for job in career_history
    )
    # Also include summary
    summary = profile.get("summary", "")
    all_text = f"{all_descriptions} {summary}"

    strong_count = _count_evidence(all_text, STRONG_AI_EVIDENCE)
    moderate_count = _count_evidence(all_text, MODERATE_AI_EVIDENCE)

    # Evidence score: 0 to 1
    # 8+ strong evidence signals → max evidence score
    evidence_score = min(1.0, (strong_count * 0.10) + (moderate_count * 0.03))

    # --- Step 3: Check for "product company" experience in career ---
    # Look at career titles for AI/ML roles (not just current title)
    career_titles = " ".join(job.get("title", "") for job in career_history)
    has_ai_career_history = _match_tier(career_titles, TIER_1_PATTERNS)

    career_bonus = 0.15 if has_ai_career_history else 0.0

    # --- Step 4: Combine ---
    # Title is the primary signal, evidence is secondary
    combined = (0.55 * title_score) + (0.35 * evidence_score) + (0.10 * min(1.0, career_bonus + evidence_score))

    return round(min(1.0, combined), 4)
