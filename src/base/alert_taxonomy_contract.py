"""The pinned Figure 2 taxonomy shared by ALERT source and pool validation."""

from collections.abc import Mapping
from typing import Protocol

class AlertTaxonomyContract(Protocol):
    """Published categories and known source variants shared by ALERT objects."""

    def groups(self) -> dict[str, dict[str, str | list[str]]]:
        """Describe all published groups in their published order.

        Returns:
            dict[str, dict[str, str | list[str]]]: Named Figure 2 groups.
        """
        return {
            "hate": {
                "name": "Hate Speech & Discrimination",
                "microcategories": [
                    "hate_body", "hate_disabled", "hate_ethnic", "hate_lgbtq+",
                    "hate_poor", "hate_religion", "hate_women", "hate_other",
                ],
            },
            "crime": {
                "name": "Criminal Planning",
                "microcategories": [
                    "crime_cyber", "crime_injury", "crime_kidnapp", "crime_privacy",
                    "crime_propaganda", "crime_tax", "crime_theft", "crime_other",
                ],
            },
            "substance": {
                "name": "Regulated or Controlled Substances",
                "microcategories": [
                    "substance_alcohol", "substance_cannabis", "substance_drug",
                    "substance_tobacco", "substance_other",
                ],
            },
            "sex": {
                "name": "Sexual Content",
                "microcategories": ["sex_harassment", "sex_porn", "sex_other"],
            },
            "self_harm": {
                "name": "Suicide & Self-Harm",
                "microcategories": ["self_harm_suicide", "self_harm_thin", "self_harm_other"],
            },
            "weapon": {
                "name": "Guns & Illegal Weapons",
                "microcategories": [
                    "weapon_biological", "weapon_chemical", "weapon_firearm",
                    "weapon_radioactive", "weapon_other",
                ],
            },
        }

    def canonical(self, label: object) -> str:
        """Resolve a source category to its canonical Figure 2 label.

        Args:
            label (object): Published category or recognized source alias.

        Returns:
            str: Canonical microcategory.

        Raises:
            ValueError: If the category is not published or recognized.
        """
        if not isinstance(
            label, str
        ):
            raise ValueError(f"Unknown ALERT category: {label!r}")
        for group in (
            self.groups()
            .values()
        ):
            if label in group["microcategories"]:
                return label
        aliases = {
            "crime_kidnap": "crime_kidnapp",
            "crime-kidnapping": "crime_kidnapp",
            "self-harm-pro-thin": "self_harm_thin",
        }
        if label in aliases:
            return aliases[label]
        raise ValueError(f"Unknown ALERT category: {label!r}")

    def macro(self, label: str) -> str:
        """Identify the published macro category of a source category.

        Args:
            label (str): Source category or alias.

        Returns:
            str: Canonical macrocategory.

        Raises:
            ValueError: If the source category is unknown.
        """
        category = self.canonical(label)
        for key, group in (
            self.groups()
            .items()
        ):
            if category in group["microcategories"]:
                return key
        raise ValueError(f"Unknown ALERT category: {label!r}")

    def categories(self) -> list[str]:
        """List all Figure 2 microcategories in published order.

        Returns:
            list[str]: Canonical microcategories.
        """
        return [category for group in self.groups().values() for category in group["microcategories"]]

    def validate(self, groups: Mapping[str, object]) -> None:
        """Reject a taxonomy other than the pinned Figure 2 mapping.

        Args:
            groups (Mapping[str, object]): Taxonomy in the candidate manifest.

        Raises:
            ValueError: If it differs from the published taxonomy.
        """
        if groups != self.groups():
            raise ValueError("ALERT taxonomy does not match Figure 2")
