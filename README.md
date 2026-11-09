# Promo Finder

Finds 20–60 s promos (hook in the first 3 s, cliffhanger at the cut, makes sense to a stranger) in microdrama
screenplays, and ranks them. The design is in [docs/pipeline_design.md](docs/pipeline_design.md).

## Run

Needs Python 3.12+. Put the input pack's episode `.txt` files (or the `.zip`) in `data/`.

**Mac / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env
python run.py --input "data/I Married a Broken Soldier. He was a Duke" --out outputs/
```

**Windows (PowerShell)**
```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then put your ANTHROPIC_API_KEY in .env
python run.py --input "data\I Married a Broken Soldier. He was a Duke" --out outputs
```

No key? Add `--provider offline --max-episodes 3` to check that everything runs (stand-in rules, no API calls).

- `--input` takes a folder or a `.zip` of episode `.txt` files.
- Each show gets a folder in `outputs/`, with one file per stage and `report.md`, `promos.csv` and `promos.json`.
- LLM replies are cached in `.cache/llm`, so a rerun is free and gives the same output.

Useful flags:
- `--provider offline` runs every stage without a key, using simple stand-in rules. Use it for smoke tests only.
- `--max-episodes 3` gives a quick check on the first episodes.
- `--no-emotion-signal` and `--no-cold-viewer` are ablation runs; they write to their own folders.

To swap models, edit `.env`: `TAGGING_MODEL`, `JUDGE_MODEL` and so on, or set
`LLM_PROVIDER=openai_compatible` with a base URL and key.

## Tests

```bash
python -m pytest
```

## Layout

| Path | Role |
|---|---|
| `promo_finder/pipeline/service.py` | Runs the stages in order (start reading here) |
| `promo_finder/screenplay/` | Parser |
| `promo_finder/line_features/` | Emotion, sound, position and recap signals (word lists in `config.py`) |
| `promo_finder/story_bible/`, `tagging/`, `signal_merge/`, `cold_viewer/`, `ranking/` | One stage each; LLM prompts in `prompts.py` |
| `promo_finder/candidates/`, `promo_rules/` | Building clips, and the hard rules |
| `promo_finder/llm/` | Swappable LLM clients, cache, validation |
| `promo_finder/config.py` | Every threshold and weight |

The input pack is Kuku's confidential content. `data/` is gitignored, so never publish it.
