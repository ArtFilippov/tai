"""ALERT Figure 2 order and an explicit, versioned label compatibility table."""

from types import MappingProxyType

TAXONOMY_VERSION = "alert-figure2-v1"
TAXONOMY_SHA256 = "0877011641dbe3a558a9b6e917d65c1daf103b1722eece04bd56905b4ba0debd"
ALIASES_VERSION = "alert-label-aliases-v1"
PAPER_URL = "https://arxiv.org/abs/2404.08676"

MACRO_CATEGORIES = MappingProxyType(
    {
        "hate": (
            "hate_body",
            "hate_disabled",
            "hate_ethnic",
            "hate_lgbtq+",
            "hate_poor",
            "hate_religion",
            "hate_women",
            "hate_other",
        ),
        "crime": (
            "crime_cyber",
            "crime_injury",
            "crime_kidnapp",
            "crime_privacy",
            "crime_propaganda",
            "crime_tax",
            "crime_theft",
            "crime_other",
        ),
        "substance": (
            "substance_alcohol",
            "substance_cannabis",
            "substance_drug",
            "substance_tobacco",
            "substance_other",
        ),
        "sex": ("sex_harassment", "sex_porn", "sex_other"),
        "self_harm": ("self_harm_suicide", "self_harm_thin", "self_harm_other"),
        "weapon": (
            "weapon_biological",
            "weapon_chemical",
            "weapon_firearm",
            "weapon_radioactive",
            "weapon_other",
        ),
    }
)
MICRO_TO_MACRO = MappingProxyType(
    {micro: macro for macro, micros in MACRO_CATEGORIES.items() for micro in micros}
)
CATEGORY_ORDER = tuple(MICRO_TO_MACRO)

# Canonical labels are accepted as-is. These are the only additional spellings;
# never infer a macro group from a prefix or normalize arbitrary punctuation.
CATEGORY_ALIASES = MappingProxyType(
    {
        "hate-body": "hate_body",
        "hate-disabled": "hate_disabled",
        "hate-ethnic": "hate_ethnic",
        "hate-lgbtq+": "hate_lgbtq+",
        "hate-poor": "hate_poor",
        "hate-religion": "hate_religion",
        "hate-women": "hate_women",
        "hate-other": "hate_other",
        "crime-cyber": "crime_cyber",
        "crime-injury": "crime_injury",
        "crime_kidnap": "crime_kidnapp",
        "crime-kidnap": "crime_kidnapp",
        "crime-kidnapp": "crime_kidnapp",
        "crime-kidnapping": "crime_kidnapp",
        "crime-privacy": "crime_privacy",
        "crime-propaganda": "crime_propaganda",
        "crime-tax": "crime_tax",
        "crime-theft": "crime_theft",
        "crime-other": "crime_other",
        "substance-alcohol": "substance_alcohol",
        "substance-cannabis": "substance_cannabis",
        "substance-drug": "substance_drug",
        "substance-tobacco": "substance_tobacco",
        "substance-other": "substance_other",
        "sex-harassment": "sex_harassment",
        "sex-porn": "sex_porn",
        "sex-other": "sex_other",
        "self-harm-suicide": "self_harm_suicide",
        "self-harm-thin": "self_harm_thin",
        "self-harm-pro-thin": "self_harm_thin",
        "self-harm-other": "self_harm_other",
        "weapon-biological": "weapon_biological",
        "weapon-chemical": "weapon_chemical",
        "weapon-firearm": "weapon_firearm",
        "weapon-radioactive": "weapon_radioactive",
        "weapon-other": "weapon_other",
    }
)


def canonical_category(label: str) -> str:
    if label in MICRO_TO_MACRO:
        return label
    try:
        return CATEGORY_ALIASES[label]
    except KeyError:
        raise ValueError(f"Unknown ALERT category: {label!r}") from None
