from enum import StrEnum

from pydantic import BaseModel, Field


class BlockKind(StrEnum):
    CAMERA = "camera"
    ACTION = "action"
    SOUND = "sound"
    DIALOGUE = "dialogue"


class Block(BaseModel):
    """One timed entry of the screenplay: a shot, an action, a sound cue or a line of dialogue."""

    block_id: str  # e.g. E07-B023, numbered in file order
    episode_number: int
    scene_number: int
    kind: BlockKind
    speaker: str | None = None
    speaker_qualifiers: list[str] = Field(default_factory=list)  # e.g. ["O.S."], ["Rupert"]
    emotion_tags: list[str] = Field(default_factory=list)
    start_s: float
    end_s: float
    text: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def is_content(self) -> bool:
        return self.kind != BlockKind.CAMERA


class Scene(BaseModel):
    scene_number: int
    heading: str
    start_s: float
    end_s: float


class Episode(BaseModel):
    episode_number: int
    source_name: str
    logline: str
    continuity_notes: str
    duration_s: float
    scenes: list[Scene]
    blocks: list[Block]  # all kinds, sorted by start_s then file order
    parse_warnings: list[str] = Field(default_factory=list)

    @property
    def shots(self) -> list[Block]:
        return [block for block in self.blocks if block.kind == BlockKind.CAMERA]

    @property
    def content_blocks(self) -> list[Block]:
        return [block for block in self.blocks if block.is_content]


class Show(BaseModel):
    title: str
    episodes: list[Episode]

    def build_block_index(self) -> dict[str, Block]:
        return {block.block_id: block for episode in self.episodes for block in episode.blocks}

    def find_episode(self, episode_number: int) -> Episode | None:
        return next((episode for episode in self.episodes if episode.episode_number == episode_number), None)
