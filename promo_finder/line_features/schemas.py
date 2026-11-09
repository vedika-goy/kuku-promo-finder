from enum import StrEnum

from pydantic import BaseModel


class SoundType(StrEnum):
    IMPACT = "IMPACT"
    STING = "STING"
    GASP = "GASP"
    ENTRANCE = "ENTRANCE"
    AMBIENT = "AMBIENT"
    OTHER = "OTHER"


class LineFeatures(BaseModel):
    """Model-free signals for one content block (camera shots get none)."""

    block_id: str
    episode_number: int
    intensity: float
    delta: float  # intensity minus the mean of the previous 3 content blocks
    z_in_episode: float
    has_tension_tag: bool
    sound_type: SoundType | None
    is_cliffhanger_cue: bool  # the sound cue literally says "cliffhanger"
    is_episode_head: bool
    is_episode_tail: bool
    is_scene_end: bool
    recap_of_block_id: str | None  # this block restages a line from the previous episode's ending
    is_tag_known: bool  # False when no emotion tag of this line is in the word list


class LexiconCoverage(BaseModel):
    tagged_dialogue_lines: int
    lines_with_known_tag: int
    unknown_tag_words: list[str]

    @property
    def coverage_ratio(self) -> float:
        return self.lines_with_known_tag / self.tagged_dialogue_lines if self.tagged_dialogue_lines else 1.0
