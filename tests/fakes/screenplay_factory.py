"""Build screenplay text in the Kuku input-pack format for tests."""

from promo_finder.ranking.service import format_timestamp


def build_episode_text(
    episode_number: int,
    scenes: list[list[tuple[str, float, float, str]]],
    is_uppercase_labels: bool = False,
    line_ending: str = "\n",
    continuity_notes: str = "Continues the story.",
) -> str:
    """scenes: one list per scene of (label, start_s, end_s, text); a label like 'JULIANA (Furious)' is dialogue."""
    field_case = str.upper if is_uppercase_labels else (lambda text: text)
    lines = [
        f"EPISODE NUMBER: Episode {episode_number}",
        f"EPISODE LOGLINE: Logline of episode {episode_number}.",
        f"CONTINUITY NOTES: {continuity_notes}",
        "",
        "--- SCRIPT START ---",
        "",
    ]
    for scene_index, scene_blocks in enumerate(scenes, start=1):
        scene_start_s = min(start_s for _, start_s, _, _ in scene_blocks)
        scene_end_s = max(end_s for _, _, end_s, _ in scene_blocks)
        lines += [
            f"SCENE {scene_index:02d}",
            field_case("Scene heading:") + " INT. HALL – DAY",
            field_case("Scene start:") + f" {format_timestamp(scene_start_s)}",
            field_case("Scene end:") + f" {format_timestamp(scene_end_s)}",
            "",
            field_case("Actions and dialogue:"),
            "",
        ]
        for label, start_s, end_s, text in scene_blocks:
            is_dialogue = "(" in label
            lines += [
                f"**{label}:**",
                f"Start: ({format_timestamp(start_s)}) | End: ({format_timestamp(end_s)})",
                f'"{text}"' if is_dialogue else text,
                "",
            ]
    return line_ending.join(lines) + "--- SCRIPT END ---"


def build_standard_episode_scenes(episode_number: int) -> list[list[tuple[str, float, float, str]]]:
    """A 70 s episode: calm opening, a slap at 20 s, rising argument, a cliffhanger line at the end."""
    blocks = [("CAMERA", 0.0, 4.0, "Wide shot of the grand hall."), ("ACTION", 0.5, 4.0, "Guests mingle quietly.")]
    blocks += [("MAID (Calm)", 4.5, 6.5, f"The tea is ready, episode {episode_number}.")]
    blocks += [("CAMERA", 4.0, 8.0, "Medium shot of the maid."), ("CAMERA", 8.0, 20.0, "Slow pan across the guests.")]
    blocks += [("LORD (Gentle)", 9.0, 11.0, "Thank you, that will be all.")]
    blocks += [("CAMERA", 20.0, 23.0, "Close-up as Clarissa raises her hand.")]
    blocks += [("CLARISSA (Furious, Screaming)", 20.2, 22.0, "How dare you!")]
    blocks += [("ACTION", 22.0, 23.0, "Clarissa slaps Juliana across the face.")]
    blocks += [("SOUND FX", 22.0, 22.8, "A sharp slap echoes through the hall.")]
    for index in range(8):
        start_s = 23.0 + index * 5.0
        blocks += [("CAMERA", start_s, start_s + 5.0, f"Close-up {index} on the quarrel.")]
        speaker = "JULIANA (Defiant)" if index % 2 == 0 else "CLARISSA (Cold)"
        blocks += [(speaker, start_s + 0.5, start_s + 3.5, f"Argument line {index}.")]
    blocks += [("CAMERA", 63.0, 67.0, "Close-up on Juliana, eyes wide.")]
    blocks += [("JULIANA (Shocked, Trembling)", 63.5, 66.5, "Then who is my real father?")]
    blocks += [("SOUND FX", 66.0, 68.0, "Ominous drone swells to a cliffhanger peak.")]
    blocks += [("CAMERA", 67.0, 69.0, "Close-up on Clarissa, frozen."), ("CAMERA", 69.0, 70.0, "Cut to black.")]
    return [blocks]
