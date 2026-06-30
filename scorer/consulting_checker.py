"""
Consulting Firm Career Checker
===============================
The JD explicitly says:
  "People who have only worked at consulting firms (TCS, Infosys, Wipro,
   Accenture, Cognizant, Capgemini, etc.) in their entire career"
  → NOT a fit.

If they have MIXED experience (consulting + product company), that's fine.
"""

# Major IT consulting/services companies
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant", "cognizant technology solutions",
    "capgemini",
    "hcl", "hcl technologies",
    "tech mahindra",
    "mindtree",  # Now part of LTIMindtree
    "ltimindtree", "lti",
    "mphasis",
    "persistent systems",
    "l&t infotech", "larsen & toubro infotech",
    "hexaware",
    "cyient",
    "zensar",
    "niit technologies",
    "css corp",
    "sonata software",
    "birlasoft",
    "coforge",
    "deloitte",  # Consulting arm
    "kpmg",
    "ey", "ernst & young",
    "pwc", "pricewaterhousecoopers",
    "mckinsey",
    "bain",
    "bcg", "boston consulting group",
}


def _normalize_company(name: str) -> str:
    """Normalize company name for matching."""
    return name.strip().lower()


def check_consulting_career(candidate: dict) -> float:
    """
    Returns a multiplier:
      1.0 = no consulting concern
      0.85 = mostly consulting but some product experience
      0.2 = entire career at consulting firms (JD disqualifier)
    """
    career_history = candidate.get("career_history", [])
    if not career_history:
        return 1.0

    total_jobs = len(career_history)
    consulting_jobs = 0

    for job in career_history:
        company = _normalize_company(job.get("company", ""))
        if company in CONSULTING_FIRMS:
            consulting_jobs += 1

    if consulting_jobs == 0:
        return 1.0  # No consulting — great

    consulting_ratio = consulting_jobs / total_jobs

    if consulting_ratio >= 1.0:
        # ENTIRE career at consulting firms — JD explicitly says no
        return 0.20
    elif consulting_ratio >= 0.75:
        # Mostly consulting
        return 0.60
    elif consulting_ratio >= 0.5:
        # Mixed — some product experience
        return 0.85
    else:
        # Mostly product companies — fine
        return 0.95
