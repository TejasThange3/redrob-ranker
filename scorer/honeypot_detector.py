"""
Honeypot Detector
==================
Identifies the ~80 honeypot candidates with subtly impossible profiles.
These are forced to relevance tier 0 — ranking them = disqualification risk.

Detection strategies:
  1. Career timeline contradictions (duration_months vs start/end dates)
  2. Expert proficiency in many skills with 0 duration
  3. Years of experience wildly mismatched with career history total
  4. Education timeline impossibilities
  5. Impossible company tenures
"""

from datetime import datetime, date


def _parse_date(date_str: str | None) -> date | None:
    """Parse a YYYY-MM-DD date string."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _months_between(d1: date, d2: date) -> int:
    """Approximate months between two dates."""
    return abs((d2.year - d1.year) * 12 + (d2.month - d1.month))


def is_honeypot(candidate: dict) -> bool:
    """
    Detect whether a candidate has an impossible/contradictory profile.
    Returns True if the candidate is likely a honeypot.
    """
    flags = 0  # Count red flags; threshold = 2 to be safe

    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    years_of_exp = profile.get("years_of_experience", 0)

    # --- Check 1: Career timeline contradictions ---
    for job in career_history:
        start = _parse_date(job.get("start_date"))
        end = _parse_date(job.get("end_date"))
        claimed_duration = job.get("duration_months", 0)

        if start and end:
            actual_months = _months_between(start, end)
            # Allow some tolerance (±6 months for rounding)
            if claimed_duration > actual_months + 12:
                flags += 1  # Duration way longer than dates suggest

            # End before start
            if end < start:
                flags += 2  # Obvious impossibility

        # Future start dates (but allow "current" jobs)
        if start and not job.get("is_current", False):
            if end and end > date(2027, 1, 1):
                flags += 1  # Ends in the far future

    # --- Check 2: Expert proficiency with 0 duration months ---
    expert_zero_count = 0
    for skill in skills:
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 0) == 0:
            expert_zero_count += 1
    if expert_zero_count >= 3:
        flags += 2  # Expert in 3+ skills with 0 months of use — very suspicious
    elif expert_zero_count >= 2:
        flags += 1

    # --- Check 3: Total career vs claimed experience ---
    total_career_months = sum(job.get("duration_months", 0) for job in career_history)
    claimed_months = years_of_exp * 12

    if total_career_months > 0 and claimed_months > 0:
        # Allow overlapping jobs (parallel roles), so we use a generous threshold
        if claimed_months > total_career_months * 2.5:
            flags += 1  # Claims way more experience than career history shows
        if total_career_months > claimed_months * 2.5:
            flags += 1  # Career history adds up to way more than claimed

    # --- Check 4: Education timeline impossibilities ---
    for edu in education:
        start_year = edu.get("start_year", 2000)
        end_year = edu.get("end_year", 2000)

        if end_year < start_year:
            flags += 2  # Graduated before enrolling

        # Very long degrees (>8 years for undergrad is suspicious)
        degree = edu.get("degree", "").lower()
        if "ph.d" not in degree and "phd" not in degree:
            if end_year - start_year > 8:
                flags += 1

        # Very young to have graduated + years of experience
        if end_year > 0 and years_of_exp > 0:
            # If they graduated in 2023 and claim 15 years experience...
            implied_start_work = end_year
            implied_years = 2026 - implied_start_work
            if years_of_exp > implied_years + 5:
                flags += 1  # More experience than mathematically possible

    # --- Check 5: Overlapping current jobs ---
    current_jobs = [j for j in career_history if j.get("is_current", False)]
    if len(current_jobs) > 2:
        flags += 1  # 3+ simultaneous current jobs is suspicious

    # --- Check 6: Impossibly high skill count with zero backing ---
    zero_endorsement_skills = sum(
        1 for s in skills
        if s.get("endorsements", 0) == 0 and s.get("duration_months", 0) == 0
    )
    if zero_endorsement_skills >= 8:
        flags += 2  # Many skills with absolutely no validation

    # --- Decision ---
    return flags >= 2
