"""
Behavioral Signal Scorer
=========================
Computes a MULTIPLIER (0.1 to 1.3) based on the 23 redrob behavioral signals.
This is applied ON TOP of the base score to reward active, responsive candidates
and penalize ghosts / unreliable ones.

Key signals (in order of importance for this JD):
  1. last_active_date — are they even around?
  2. recruiter_response_rate — will they reply?
  3. open_to_work_flag — are they looking?
  4. interview_completion_rate — do they show up?
  5. profile_completeness_score — are they serious?
  6. github_activity_score — do they actually code?
  7. avg_response_time_hours — how fast do they respond?
  8. offer_acceptance_rate — do they accept offers?
"""

from datetime import datetime, date


def _parse_date(date_str: str) -> date | None:
    """Parse a date string in YYYY-MM-DD format."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def score_signals(candidate: dict, reference_date: date | None = None) -> float:
    """
    Compute a behavioral multiplier based on redrob_signals.
    Returns a float between 0.1 and 1.3.

    Values > 1.0 = candidate gets a boost (very active & engaged).
    Values < 1.0 = candidate gets a penalty (inactive / unreliable).
    """
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return 0.5  # No signals → significant penalty

    if reference_date is None:
        reference_date = date(2026, 6, 26)  # Competition reference date

    multiplier = 1.0

    # --- 1. Recency (last_active_date) — MOST IMPORTANT ---
    last_active = _parse_date(signals.get("last_active_date", ""))
    if last_active:
        days_since_active = (reference_date - last_active).days
        if days_since_active < 0:
            days_since_active = 0  # Future date — treat as active

        if days_since_active <= 14:
            multiplier *= 1.10   # Very active — bonus
        elif days_since_active <= 30:
            multiplier *= 1.05
        elif days_since_active <= 60:
            multiplier *= 1.0    # Neutral
        elif days_since_active <= 90:
            multiplier *= 0.85
        elif days_since_active <= 180:
            multiplier *= 0.60   # Going stale
        else:
            multiplier *= 0.30   # Ghost — 6+ months inactive
    else:
        multiplier *= 0.50  # No date → probably inactive

    # --- 2. Recruiter response rate ---
    response_rate = signals.get("recruiter_response_rate", 0.0)
    if response_rate >= 0.7:
        multiplier *= 1.10   # Very responsive
    elif response_rate >= 0.5:
        multiplier *= 1.05
    elif response_rate >= 0.3:
        multiplier *= 0.95
    elif response_rate >= 0.1:
        multiplier *= 0.75
    else:
        multiplier *= 0.50   # Almost never replies

    # --- 3. Open to work flag ---
    if signals.get("open_to_work_flag", False):
        multiplier *= 1.08
    else:
        multiplier *= 0.90   # Not actively looking

    # --- 4. Interview completion rate ---
    interview_rate = signals.get("interview_completion_rate", 0.5)
    if interview_rate >= 0.8:
        multiplier *= 1.05   # Reliable
    elif interview_rate >= 0.6:
        multiplier *= 1.0    # OK
    elif interview_rate >= 0.4:
        multiplier *= 0.90
    else:
        multiplier *= 0.70   # Unreliable — drops out

    # --- 5. Profile completeness ---
    completeness = signals.get("profile_completeness_score", 50)
    if completeness >= 85:
        multiplier *= 1.05
    elif completeness >= 60:
        multiplier *= 1.0
    elif completeness >= 40:
        multiplier *= 0.90
    else:
        multiplier *= 0.75   # Barely filled out profile

    # --- 6. GitHub activity (important for AI Engineer role) ---
    github_score = signals.get("github_activity_score", -1)
    if github_score == -1:
        pass  # No GitHub linked — neutral, don't penalize
    elif github_score >= 60:
        multiplier *= 1.08   # Very active coder
    elif github_score >= 30:
        multiplier *= 1.03
    elif github_score >= 10:
        multiplier *= 1.0
    else:
        multiplier *= 0.95   # Has GitHub but barely active

    # --- 7. Response time ---
    avg_response_hours = signals.get("avg_response_time_hours", 48)
    if avg_response_hours <= 6:
        multiplier *= 1.05   # Lightning fast
    elif avg_response_hours <= 24:
        multiplier *= 1.02
    elif avg_response_hours <= 72:
        multiplier *= 1.0
    elif avg_response_hours <= 168:
        multiplier *= 0.95
    else:
        multiplier *= 0.85   # Takes over a week to respond

    # --- 8. Offer acceptance rate ---
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate == -1:
        pass  # No offer history — neutral
    elif offer_rate >= 0.7:
        multiplier *= 1.03   # Accepts offers
    elif offer_rate >= 0.4:
        multiplier *= 1.0
    else:
        multiplier *= 0.90   # Tends to reject offers

    # --- 9. Verification signals (small bonus) ---
    if signals.get("verified_email", False):
        multiplier *= 1.01
    if signals.get("verified_phone", False):
        multiplier *= 1.01
    if signals.get("linkedin_connected", False):
        multiplier *= 1.02

    # --- 10. Social proof signals (small bonus) ---
    saved_by = signals.get("saved_by_recruiters_30d", 0)
    if saved_by >= 15:
        multiplier *= 1.05
    elif saved_by >= 5:
        multiplier *= 1.02

    search_appearances = signals.get("search_appearance_30d", 0)
    if search_appearances >= 200:
        multiplier *= 1.03
    elif search_appearances >= 50:
        multiplier *= 1.01

    # Clamp to range
    return round(max(0.1, min(1.3, multiplier)), 4)
