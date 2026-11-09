# Promo Finder: pipeline design

Problem 1 of the Kuku AI Generalist assignment: read a microdrama's screenplays and find the best 20–60 s promos.
Each promo has a hook in the first 3 s, ends on a cliffhanger, and makes sense to a stranger. For each one the tool
gives the segments, hook, cliffhanger and why it works, then ranks them.

Start reading the code at `promo_finder/pipeline/service.py` (`run_pipeline`); it calls each stage in order.

## Core idea

**The LLM judges and code measures.** The model never writes a timestamp. It points at block ids (`E07-B023`), and
code does all the timing, the rules, assembling the clips and removing duplicates. Every hard rule is therefore
checked against the parsed script, never against model output (`promo_rules/service.py`).

## What the input pack looks like, and what that changed

| Fact in the data | Design consequence |
|---|---|
| Every block (shot, action, sound, line) has an exact start and end time | No duration guessing. Clips snap to **CAMERA shot boundaries**, which are the real cut points |
| Dialogue labels carry the writer's emotion tags: `JULIANA (Outraged, Accusing)` | Emotion intensity comes from these tags through a word list (97.8% of tagged lines covered), not a downloaded classifier |
| 114 `SOUND FX` cues, some saying "cliffhanger peak" | Sound cues are typed (IMPACT, STING, GASP, ENTRANCE, AMBIENT) and act as an independent hook/cliffhanger signal |
| CAMERA blocks describe what is on screen | The tagger sees them, which partly covers visual-only hooks |
| 11 episodes open by restaging the previous ending in new words | Recaps are found by word-stem overlap (exact matching fails) and skipped when stitching across episodes |
| Episode 6 uses uppercase labels; files end without a newline | The parser ignores letter case and line endings; a test covers a synthetic show in a different layout |
| One label can mean different people (`BLOND NOBLEMAN` is Gerald or Rupert) | The story bible maps speaker labels to characters per episode range |

## Stages

| # | Stage | Who | Output file |
|---|---|---|---|
| 1 | Parse every episode into timed blocks | code | `01_lines.json` |
| 2 | Line features: intensity, delta, z-score in the episode, tension tags, sound type, episode head/tail, scene end, recap | code | `02_line_features.json` |
| 3 | Story bible: characters, label aliases per episode range, arcs, twists with spoiler level, episode summaries | LLM, once per show | `03_story_bible.json` |
| 4 | Tagging per episode: up to 5 hooks and 3 cliffhangers by block id, with strength 1–5, type, needs_context, spoiler level, open question, `answered_at`, and arc lines (ESSENTIAL/SUPPORTING) | LLM, validated in code | `04_llm_tags/E##.json` |
| 5 | Merge: per-line hook and cliffhanger scores from the LLM tags plus emotion/sound signals; evidence labels; one LLM review of lines the signal flags but the tagger missed | code + 1 LLM call | `05_line_map.json` |
| 6 | Candidates: hook × cliffhanger pairs, snapped to shots, stitched across episodes, compressed, checked against the hard rules, spoilers held back, duplicates removed | code | `06_candidates.json` |
| 7 | Cold viewer: an opening test on the first 3 s only; a stranger who sees the whole clip (names hidden unless spoken); a grader that compares their understanding with the bible; one repair attempt | LLM | `07_judged.json` |
| 7b | Pairwise: the top 15 meet their 3 nearest neighbours head to head, in both orders; Bradley–Terry strengths | LLM + code | `08_pairwise.json` |
| 8 | Rank: final score blended with pairwise strength, early-episode bonus, a diversity-aware posting order, a why-it-works line | code + 1 LLM call | `promos.json`, `promos.csv`, `report.md` |
| 9 | Editor reviews the top 10–15 | human | — |

### Two signals, kept independent
The tagging prompt never sees the emotion scores, so the two can check each other.

```
hook_score        = 0.6 · LLM strength/5 + 0.4 · max(emotion spike, sound cue) around the line
cliffhanger_score = 0.6 · LLM strength/5 + 0.2 · boundary (episode tail 1, scene end 0.5) + 0.2 · tension
```

Each hook also gets an evidence label:

| Evidence | Meaning | What happens |
|---|---|---|
| `CORROBORATED` | LLM tag plus signal | Normal path |
| `CONTEXT_HOOK` | LLM tag only, typical of quiet reveals | Must convince the cold viewer |
| `SIGNAL_ONLY` | The signal fired, the tagger missed it, and the review confirmed it | Normal path |

If the LLM fails on an episode, it is marked `UNTAGGED` and runs on the signals alone.

### Hard rules (code, every candidate)
- Duration is 20–60 s.
- The hook lands within 3 s.
- The clip ends before `answered_at`, and no dialogue starts after the cliffhanger line.
- No cut goes through a line.
- At most 3 segments, each at least 3 s.
- No recap lines inside the clip.
- A clip never opens on the tail of a sentence. A hook line that opens with "..." starts where its sentence starts;
  in this pack, 38 of 39 such lines finish the same speaker's previous line.
- Major spoilers are held back only after the free episodes (1–10).

Every rejected candidate is kept in `06_candidates.json` with its reasons.

### Cold viewer (the "makes sense on its own" check)
- **Opening test.** A separate call sees only the first 3 seconds and rates the stop strength 1–5. A viewer who has
  read the whole clip rates the opening too kindly.
- **Viewer.** The viewer sees the clip as a transcript with relative times. A word counts as a name only if it
  never appears in lowercase in the scripts; each character gets one "Person X" label, in a fixed order so reruns
  match the cache. It answers:
  - what is happening, and who is who?
  - what question is left?
  - what was confusing?
  - how much does it want the next episode (1–5)?
- **Grader.** A grader with the bible scores:
  - comprehension (0–2)
  - cliffhanger match (0–2)
  - whether the clip resolves its own question
- **Outcome.**
  - Comprehension 0 triggers one repair: add back the most important line that compression dropped.
  - Comprehension still 0, or a question answered inside the clip, means the candidate is rejected. This is a semantic cliffhanger check, not just a positional one.

### Ranking
```
appeal = 0.5 · want_next/5 + 0.5 · opening stop strength/5
final  = 0.30 · appeal + 0.25 · comprehension/2 + 0.15 · cliffhanger match/2 + 0.15 · hook + 0.15 · emotion shape
final *= 0.8 + 0.2 · early_episode   (1.0 for episodes 1–10, falling to 0.3 by episode 40)
relevance = 0.6 · pairwise strength + 0.4 · final   (promos outside the pairwise pool: 0.4 · final)
order: each pick = argmax 0.7 · relevance − 0.3 · similarity to earlier picks (same arc, hook type, episode)
```

**Why pairwise.** Absolute LLM scores bunch together. On 5 episodes, opening strength was 2–3, want-next 3–4 and
comprehension 1–2. Head-to-head judgments spread the same promos over 0–1.
- **Position bias.** Each pair is asked in both orders, and a win counts only if it survives the swap; a split is
  a tie.
- **Cost.** Pairs are limited to nearest neighbours in score order: 39 pairs, 78 calls for a pool of 15.

**Assumption:** the first ~10 episodes are free, so a promo from an early episode lets a new viewer reach the moment
before the paywall. Every weight and threshold lives in `promo_finder/config.py`.

## Swapping models
- All LLM calls go through `llm/service.py`. It handles the cache, schema validation, one repair, and logging of tokens and cost.
- Providers: `anthropic` (default, structured outputs plus server-side refusal fallback), `openai_compatible` (any compatible API) and `offline` (deterministic rules, for smoke tests).
- Model names are set in `.env` (`BIBLE_MODEL`, `TAGGING_MODEL`, `JUDGE_MODEL`, `VIEWER_MODEL`).

## Known limits
- **Text only.** Acting, faces and music are judged from the script's descriptions.
- **Timing.** Timestamps are the script's. The real edit may drift by about a second.
- **The cold viewer is a model.** It may infer more than a real scroller would, so the editor still reviews the top picks.
- **Recap detection is a word-overlap heuristic.**
