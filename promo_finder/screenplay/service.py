"""Parse Kuku screenplay .txt files (from a folder or a .zip) into timed blocks."""

import logging
import re
import zipfile
from pathlib import Path

from promo_finder.screenplay.schemas import Block, BlockKind, Episode, Scene, Show

logger = logging.getLogger(__name__)

# Labels and field names vary in case between episodes (episode 6 is all uppercase), so every
# pattern is case-insensitive.
_HEADER_FIELD = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES):\s*(.*)$", re.I)
_SCENE_NUMBER = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
_SCENE_FIELD = re.compile(r"^SCENE (HEADING|START|END):\s*(.*)$", re.I)
_BLOCK_LABEL = re.compile(r"^\*\*(.+?):\*\*\s*$")
# matches "Start: (00:11.30) | End: (00:13.10)", also h:mm:ss.cc
_BLOCK_TIMING = re.compile(r"^START:\s*\(([\d:.]+)\)\s*\|\s*END:\s*\(([\d:.]+)\)\s*$", re.I)
_SCRIPT_END = re.compile(r"---\s*SCRIPT END\s*---", re.I)
_EPISODE_NUMBER_IN_NAME = re.compile(r"episode[-_ ]?0*(\d+)", re.I)
_PARENTHESIZED = re.compile(r"\(([^()]*)\)")
_DELIVERY_QUALIFIERS = {"O.S.", "V.O.", "O.C.", "OFF-SCREEN", "CONT'D", "CONT’D"}
_NON_DIALOGUE_LABELS = {"CAMERA": BlockKind.CAMERA, "ACTION": BlockKind.ACTION, "SOUND FX": BlockKind.SOUND}


def load_show(input_path: Path) -> Show:
    """Read every episode file from a folder or a .zip. Episodes are sorted by number."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    named_texts = _read_zip_texts(input_path) if input_path.suffix.lower() == ".zip" else _read_folder_texts(input_path)
    if not named_texts:
        raise ValueError(f"No .txt screenplay files found in {input_path}")
    episodes = [parse_episode_text(text, name) for name, text in named_texts]
    episodes.sort(key=lambda episode: episode.episode_number)
    _check_unique_episode_numbers(episodes)
    # Path.stem would cut a folder name like "He Was a Duke. Part 2" at the dot, so folders keep their full name.
    raw_title = input_path.stem if input_path.suffix.lower() == ".zip" else input_path.name
    title = raw_title.split("-scripts")[0].strip()
    return Show(title=title, episodes=episodes)


def parse_episode_text(text: str, source_name: str) -> Episode:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header = _parse_header(lines)
    episode_number = _resolve_episode_number(header.get("EPISODE NUMBER", ""), source_name)
    scenes, raw_blocks, warnings = _parse_body(lines, episode_number)
    blocks = _number_and_sort_blocks(raw_blocks, episode_number)
    duration_s = max([scene.end_s for scene in scenes] + [block.end_s for block in blocks] + [0.0])
    if not blocks:
        warnings.append("no timed blocks found")
    for warning in warnings:
        logger.warning("Episode %s (%s): %s", episode_number, source_name, warning)
    return Episode(
        episode_number=episode_number,
        source_name=source_name,
        logline=header.get("EPISODE LOGLINE", ""),
        continuity_notes=header.get("CONTINUITY NOTES", ""),
        duration_s=duration_s,
        scenes=scenes,
        blocks=blocks,
        parse_warnings=warnings,
    )


def parse_timestamp_s(value: str) -> float:
    """'01:12.50' -> 72.5; also accepts '1:01:12.50' and plain seconds."""
    parts = [float(part) for part in value.strip().split(":")]
    total_s = 0.0
    for part in parts:
        total_s = total_s * 60 + part
    return round(total_s, 3)


def _read_zip_texts(zip_path: Path) -> list[tuple[str, str]]:
    named_texts = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in sorted(archive.namelist()):
            if member.lower().endswith(".txt") and not Path(member).name.startswith("."):
                named_texts.append((Path(member).name, archive.read(member).decode("utf-8", errors="replace")))
    return named_texts


def _read_folder_texts(folder_path: Path) -> list[tuple[str, str]]:
    paths = sorted(path for path in folder_path.rglob("*") if path.suffix.lower() == ".txt" and not path.name.startswith("."))
    return [(path.name, path.read_text(encoding="utf-8", errors="replace")) for path in paths]


def _check_unique_episode_numbers(episodes: list[Episode]) -> None:
    seen_numbers = set()
    for episode in episodes:
        if episode.episode_number in seen_numbers:
            raise ValueError(f"Episode number {episode.episode_number} appears twice ({episode.source_name})")
        seen_numbers.add(episode.episode_number)


def _parse_header(lines: list[str]) -> dict[str, str]:
    header = {}
    for line in lines[:20]:
        match = _HEADER_FIELD.match(line.strip())
        if match:
            header[match.group(1).upper()] = match.group(2).strip()
    return header


def _resolve_episode_number(header_value: str, source_name: str) -> int:
    digits = re.search(r"(\d+)", header_value)
    if digits:
        return int(digits.group(1))
    name_match = _EPISODE_NUMBER_IN_NAME.search(source_name)
    if name_match:
        return int(name_match.group(1))
    raise ValueError(f"Cannot tell the episode number of {source_name}")


def _parse_body(lines: list[str], episode_number: int) -> tuple[list[Scene], list[dict], list[str]]:
    scenes: list[Scene] = []
    raw_blocks: list[dict] = []
    warnings: list[str] = []
    scene_fields: dict[str, str] = {}
    current_scene_number = 0
    line_index = 0
    while line_index < len(lines):
        line = _SCRIPT_END.split(lines[line_index])[0].strip()
        scene_match = _SCENE_NUMBER.match(line)
        if scene_match:
            _close_scene(scenes, current_scene_number, scene_fields)
            current_scene_number, scene_fields = int(scene_match.group(1)), {}
        elif _SCENE_FIELD.match(line):
            field_match = _SCENE_FIELD.match(line)
            scene_fields[field_match.group(1).upper()] = field_match.group(2).strip()
        elif _BLOCK_LABEL.match(line):
            block, line_index = _parse_block(lines, line_index, current_scene_number, warnings)
            if block:
                raw_blocks.append(block)
            continue
        line_index += 1
    _close_scene(scenes, current_scene_number, scene_fields)
    return scenes, raw_blocks, warnings


def _close_scene(scenes: list[Scene], scene_number: int, scene_fields: dict[str, str]) -> None:
    if scene_number == 0:
        return
    scenes.append(
        Scene(
            scene_number=scene_number,
            heading=scene_fields.get("HEADING", ""),
            start_s=parse_timestamp_s(scene_fields.get("START", "0")),
            end_s=parse_timestamp_s(scene_fields.get("END", "0")),
        )
    )


def _parse_block(lines: list[str], label_index: int, scene_number: int, warnings: list[str]) -> tuple[dict | None, int]:
    """Read one '**LABEL:**' block. Returns the raw block and the index of the next unread line."""
    label = _BLOCK_LABEL.match(lines[label_index].strip()).group(1).strip()
    timing_index = label_index + 1
    timing_match = _BLOCK_TIMING.match(lines[timing_index].strip()) if timing_index < len(lines) else None
    if not timing_match:
        warnings.append(f"block '{label}' at line {label_index + 1} has no Start/End line; skipped")
        return None, label_index + 1
    text_lines = []
    next_index = timing_index + 1
    while next_index < len(lines):
        candidate_line = _SCRIPT_END.split(lines[next_index])[0].strip()
        if not candidate_line or _BLOCK_LABEL.match(candidate_line) or _SCENE_NUMBER.match(candidate_line):
            break
        text_lines.append(candidate_line)
        next_index += 1
    start_s = parse_timestamp_s(timing_match.group(1))
    end_s = parse_timestamp_s(timing_match.group(2))
    if end_s < start_s:
        warnings.append(f"block '{label}' at line {label_index + 1} ends before it starts; end set to start")
        end_s = start_s
    block = {"label": label, "scene_number": scene_number, "start_s": start_s, "end_s": end_s, "text": " ".join(text_lines)}
    return block, next_index


def _number_and_sort_blocks(raw_blocks: list[dict], episode_number: int) -> list[Block]:
    blocks = []
    for file_order, raw_block in enumerate(raw_blocks, start=1):
        kind, speaker, qualifiers, emotion_tags = _classify_label(raw_block["label"])
        text = raw_block["text"]
        if kind == BlockKind.DIALOGUE:
            text = text.strip().strip('"“”').strip()
        blocks.append(
            Block(
                block_id=f"E{episode_number:02d}-B{file_order:03d}",
                episode_number=episode_number,
                scene_number=raw_block["scene_number"],
                kind=kind,
                speaker=speaker,
                speaker_qualifiers=qualifiers,
                emotion_tags=emotion_tags,
                start_s=raw_block["start_s"],
                end_s=raw_block["end_s"],
                text=text,
            )
        )
    # Python's sort is stable, so blocks with the same start keep file order.
    return sorted(blocks, key=lambda block: block.start_s)


def _classify_label(label: str) -> tuple[BlockKind, str | None, list[str], list[str]]:
    """'ALEXANDER (O.S.) (Pained, Selfless)' -> dialogue, 'ALEXANDER', ['O.S.'], ['Pained', 'Selfless']."""
    upper_label = label.upper()
    if upper_label in _NON_DIALOGUE_LABELS:
        return _NON_DIALOGUE_LABELS[upper_label], None, [], []
    groups = _PARENTHESIZED.findall(label)
    speaker = _PARENTHESIZED.sub("", label).strip()
    emotion_tags: list[str] = []
    qualifiers: list[str] = []
    for group_index, group in enumerate(groups):
        is_last_group = group_index == len(groups) - 1
        if is_last_group and group.strip().upper() not in _DELIVERY_QUALIFIERS:
            emotion_tags = [tag.strip() for tag in group.split(",") if tag.strip()]
        else:
            qualifiers.append(group.strip())
    return BlockKind.DIALOGUE, speaker, qualifiers, emotion_tags
