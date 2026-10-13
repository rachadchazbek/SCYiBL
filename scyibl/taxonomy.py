"""The eight categories, defined once (previously duplicated in three legacy files).

Definitions are PROVISIONAL until ``docs/labels.md`` is written from the original annotation
guide (issue #3). The expansions of the acronyms CP and ED are not confirmed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    CP = "cp"
    CONSTRUCTIVISM = "constructivism"
    COPING = "coping"
    ED = "ed"
    ER_SR = "er_sr"
    EMPOWERMENT = "empowerment"
    EXPOSURE_HABIT = "exposure_habit"
    PERMA = "perma"


@dataclass(frozen=True)
class CategoryInfo:
    label: str
    """Display label used in the original dataset and in API responses."""
    definition: str
    confirmed: bool
    """False while the definition is a working hypothesis (see issue #3)."""


CATEGORIES: dict[Category, CategoryInfo] = {
    Category.CP: CategoryInfo(
        label="CP",
        definition=(
            "Cognitive processing of experience: recalling, narrating or making sense of past "
            "events, situations and decisions, and their consequences."
        ),
        confirmed=False,
    ),
    Category.CONSTRUCTIVISM: CategoryInfo(
        label="Constructivism",
        definition=(
            "Constructing personal meaning: reinterpreting experiences, building a new narrative "
            "or perspective about oneself, others or events."
        ),
        confirmed=False,
    ),
    Category.COPING: CategoryInfo(
        label="Coping",
        definition=(
            "Concrete strategies used to deal with stress or difficulties, such as "
            "problem-solving, planning, seeking support or taking a break."
        ),
        confirmed=False,
    ),
    Category.ED: CategoryInfo(
        label="ED",
        definition=(
            "Working hypothesis, acronym unconfirmed: expressions of relief, acceptance, "
            "gratitude or growth in one's outlook on life."
        ),
        confirmed=False,
    ),
    Category.ER_SR: CategoryInfo(
        label="ER/SR",
        definition=(
            "Emotion regulation / self-regulation: noticing, expressing or managing emotions and "
            "moods, including negative states such as anxiety, sadness or irritability."
        ),
        confirmed=False,
    ),
    Category.EMPOWERMENT: CategoryInfo(
        label="Empowerment",
        definition=(
            "A sense of agency, confidence or control: believing one can change things or "
            "wanting to become a stronger, more capable person."
        ),
        confirmed=False,
    ),
    Category.EXPOSURE_HABIT: CategoryInfo(
        label="Exposure/Habit",
        definition=(
            "Gradual exposure to feared situations, or building repeated habits and routines."
        ),
        confirmed=False,
    ),
    Category.PERMA: CategoryInfo(
        label="PERMA",
        definition=(
            "Well-being in Seligman's PERMA model: positive emotion, engagement, relationships, "
            "meaning or accomplishment."
        ),
        confirmed=False,
    ),
}

assert set(CATEGORIES) == set(Category), "every category needs a definition"  # noqa: S101
