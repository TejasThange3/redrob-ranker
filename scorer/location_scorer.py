"""
Location & Logistics Scorer
============================
Scores candidates on:
  1. Country (India required, no visa sponsorship)
  2. City preference (Pune/Noida > other Tier-1 > elsewhere)
  3. Notice period (shorter = better)
  4. Work mode compatibility (hybrid/flexible preferred)
  5. Willingness to relocate
"""


# Indian Tier-1 cities and metro regions
PREFERRED_CITIES = ["pune", "noida"]
TIER_1_CITIES = [
    "bangalore", "bengaluru", "hyderabad", "mumbai",
    "delhi", "gurgaon", "gurugram", "chennai",
    "new delhi", "ncr", "ghaziabad", "faridabad",
]


def score_location(candidate: dict) -> float:
    """
    Score location & logistics fit. Returns 0.0 to 1.0.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    country = profile.get("country", "").strip().lower()
    location = profile.get("location", "").strip().lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    notice_period = signals.get("notice_period_days", 90)
    work_mode = signals.get("preferred_work_mode", "onsite")

    # --- 1. Country check ---
    if country == "india":
        country_score = 1.0
    elif willing_to_relocate:
        # Outside India but willing to relocate — still risky (no visa sponsorship)
        country_score = 0.25
    else:
        # Outside India, not willing to relocate
        country_score = 0.05

    # --- 2. City preference ---
    city_score = 0.5  # Default for unknown Indian city

    if any(city in location for city in PREFERRED_CITIES):
        city_score = 1.0
    elif any(city in location for city in TIER_1_CITIES):
        city_score = 0.8
    elif country != "india":
        city_score = 0.2

    # Relocation willingness boosts city score for non-preferred locations
    if city_score < 0.8 and willing_to_relocate:
        city_score = min(1.0, city_score + 0.2)

    # --- 3. Notice period ---
    if notice_period <= 30:
        notice_score = 1.0  # JD: "We'd love sub-30-day notice"
    elif notice_period <= 60:
        notice_score = 0.80  # Standard
    elif notice_period <= 90:
        notice_score = 0.55  # "Bar gets higher"
    elif notice_period <= 120:
        notice_score = 0.35
    else:
        notice_score = 0.15  # 120+ days is very long

    # --- 4. Work mode ---
    # JD says "Hybrid - flexible cadence"
    work_mode_map = {
        "hybrid": 1.0,
        "flexible": 1.0,
        "onsite": 0.8,   # They have offices, onsite is fine
        "remote": 0.5,   # They expect some office presence
    }
    work_mode_score = work_mode_map.get(work_mode, 0.5)

    # --- Combine ---
    combined = (
        0.35 * country_score +
        0.25 * city_score +
        0.25 * notice_score +
        0.15 * work_mode_score
    )

    return round(combined, 4)
