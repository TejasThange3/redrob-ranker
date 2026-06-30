"""
Experience Fit Scorer
=====================
Scores candidates based on years of experience relative to the JD requirement.
JD says 5-9 years, ideal is 6-8. Uses a smooth curve.
"""

import math


def score_experience(candidate: dict) -> float:
    """
    Score experience fit. Returns 0.0 to 1.0.
    Sweet spot: 6-8 years (score = 1.0).
    Falls off smoothly in both directions.
    """
    profile = candidate.get("profile", {})
    years = profile.get("years_of_experience", 0)

    # Ideal center = 7 years, sigma = 3
    # Gaussian-like decay from the ideal
    ideal = 7.0
    sigma = 3.0

    if 5.5 <= years <= 9.0:
        # In the sweet spot
        return 1.0
    elif 4.0 <= years < 5.5:
        # Slightly below range — still reasonable
        return 0.80
    elif 9.0 < years <= 12.0:
        # Slightly above — JD says 5-9 but experienced is OK
        return 0.75
    elif 3.0 <= years < 4.0:
        # Below range
        return 0.50
    elif 12.0 < years <= 15.0:
        # Quite senior — may not be a fit (JD warns about non-coding seniors)
        return 0.45
    else:
        # Well outside range — use gaussian decay
        score = math.exp(-0.5 * ((years - ideal) / sigma) ** 2)
        return round(max(0.05, score), 4)
