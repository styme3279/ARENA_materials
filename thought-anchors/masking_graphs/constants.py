"""
Constants for subject categorization and visualization styling.
"""

from collections import defaultdict

# Domain name constants - change these to update domain names throughout the codebase
MATH_PHYSICS_LOGIC_NAME = "Math, Physics, & Logic"
LIFE_SCIENCES_NAME = "Life Sciences"
HUMANITIES_SOCIAL_NAME = "Humanities/Social"


def make_subject_to_domain_dict():
    subject2domain = {}
    domain2subjects = defaultdict(list)
    for subject in STEM_LOGIC_SUBJECTS:
        subject2domain[subject] = MATH_PHYSICS_LOGIC_NAME
        domain2subjects[MATH_PHYSICS_LOGIC_NAME].append(subject)
    for subject in LIFE_SCIENCES_SUBJECTS:
        subject2domain[subject] = LIFE_SCIENCES_NAME
        domain2subjects[LIFE_SCIENCES_NAME].append(subject)
    for subject in HUMANITIES_SOCIAL_SUBJECTS:
        subject2domain[subject] = HUMANITIES_SOCIAL_NAME
        domain2subjects[HUMANITIES_SOCIAL_NAME].append(subject)
    return subject2domain, dict(domain2subjects)


# Mathematics, Physics, and Logic - includes formal logic, mathematics, physics, CS, engineering
STEM_LOGIC_SUBJECTS = [
    "formal_logic",
    "logical_fallacies",
    "college_computer_science",
    "high_school_computer_science",
    "computer_security",
    "machine_learning",
    "college_mathematics",
    "high_school_mathematics",
    "elementary_mathematics",
    "abstract_algebra",
    "college_physics",
    "high_school_physics",
    "conceptual_physics",
    "high_school_statistics",
    "econometrics",
    "electrical_engineering",
    "astronomy",
]

# Life Sciences - includes biology, medicine, chemistry
LIFE_SCIENCES_SUBJECTS = [
    "college_chemistry",
    "high_school_chemistry",
    "college_biology",
    "high_school_biology",
    "college_medicine",
    "professional_medicine",
    "medical_genetics",
    "clinical_knowledge",
    "anatomy",
    "virology",
    "human_aging",
    "human_sexuality",
    "nutrition",
    "professional_psychology",
]

# Humanities/Social Sciences
HUMANITIES_SOCIAL_SUBJECTS = [
    "high_school_government_and_politics",
    "high_school_psychology",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "business_ethics",
    "high_school_european_history",
    "high_school_geography",
    "high_school_world_history",
    "high_school_us_history",
    "prehistory",
    "international_law",
    "jurisprudence",
    "professional_law",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "management",
    "marketing",
    "public_relations",
    "professional_accounting",
    "us_foreign_policy",
    "security_studies",
    "sociology",
    "world_religions",
    "global_facts",
    "miscellaneous",
]

# Color mappings for category visualizations
CATEGORY_COLORS = {
    MATH_PHYSICS_LOGIC_NAME: "#FF6B6B",
    LIFE_SCIENCES_NAME: "#95E77E",
    HUMANITIES_SOCIAL_NAME: "#4ECDC4",
}

# Extended color mapping including "All" category for certain plots
CATEGORY_COLORS_WITH_ALL = {
    MATH_PHYSICS_LOGIC_NAME: "#FF6B6B",
    LIFE_SCIENCES_NAME: "#95E77E",
    HUMANITIES_SOCIAL_NAME: "#4ECDC4",
    "All": "#666666",
}

SUBJECT2DOMAIN, DOMAIN2SUBJECTS = make_subject_to_domain_dict()
