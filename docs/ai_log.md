# AI tools used, and where they got it wrong

Tools:
- **Claude (Claude Code).** Design discussion, code, tests and this log.
- **Claude API, inside the tool.** Story bible, tagging, cold viewer, grader, why-lines.

Where AI got it wrong (kept as I build; most recent last):

1. **Sound typing.** The first word list matched "crack" inside "crackle", so a warm fireplace counted as an IMPACT
   sound and showed up among the most intense lines. I caught it by reading the top z-scores. Matching is now at word
   starts, with `crack(?!l)`.
2. **Recap detection.** The plan assumed episode openings repeat the previous ending word for word. They don't: Ep 2
   restages "I cannot marry a woman of such plainness…" as "I cannot marry you, Juliana." Matching is now by
   word-stem overlap.
3. **Show title.** `Path.stem` cut the folder name at its first dot ("I Married a Broken Soldier. He was a Duke"
   became "I Married a Broken Soldier"). The offline smoke run caught it.
4. **A wrong test.** A test I wrote assumed a strength-4 hook with no emotion spike passes the 0.5 threshold. It
   scores 0.48, so the code was right and the test was wrong.
5. **Spoilers (Claude inside the tool, first real run on 5 episodes).** The story bible and the tagger marked
   premise events as MAJOR twists: the jilting in Ep 1, the stranger's rescue in Ep 3, the marriage in Ep 4. The
   show's title already promises them. This held back the best hook in the run, the guard takedown (score 1.0).
   - Fix in code: never hold a spoiler from the free episodes 1–10.
   - Fix in the prompts: the title and premise are known to every viewer.
   - Result: MAJOR tags went from 5 to 0, and the takedown now ranks #3.
6. **My own threshold design.** The merge weights meant an LLM-rated 4 with no emotion spike scored 0.48, under
   the 0.5 cut, so 8 of 15 such hooks never became clips. LLM-rated 4s now go to the cold viewer. On the same 5
   episodes, candidates went from 14 to 22 and ranked promos from 4 to 7.
7. **A cold viewer too kind (Claude inside the tool).** Having read the whole clip, the viewer said "would stop
   scrolling" for 7 of 7 promos, so the score separated nothing. A separate call now sees only the first 3 seconds
   and rates them 1–5, calibrated with "most clips in a feed get 1 or 2". The same 7 promos now score 2–3.
   - Remaining limit: absolute LLM scores still bunch together, so pairwise comparison is the next step.
8. **Hooks and setup lines that start mid-sentence.**
   - The tagger picked "...than spend another moment begging for yours." as a hook, and "...and it isn't you."
     as a setup line. Both are tails of the same speaker's previous line; 38 of the 39 "..." lines in the pack
     are like that.
   - Fix in code: a hook starts where its sentence starts, and a clip never opens on a sentence tail.
   - Result: the Ep 4 hook now opens on the full sentence, "I would rather choose my own life... ...than spend
     another moment begging for yours." The "Guards! Remove her!" clip dropped out, because its sentence starts 5 s
     before the hook.
9. **A name-hiding bug I wrote, caught by comparing two runs.** One promo's final score fell from 0.910 to 0.680
   with nothing upstream changed. Two causes:
   - The anonymizer picked a character's "Person X" label from a Python set, whose order changes between runs. The
     clip text changed, the cache missed, and the viewer gave a different verdict.
   - It split descriptive names into single words, so "Mysterious Bearded Man" became "Person E Person F
     Person G", and it hid ordinary words like "Man".

   Fix:
   - A word counts as a name only if it never appears in lowercase in the scripts.
   - Each character gets one label, in order of first appearance.
   - Words are processed in sorted order.

   Check: renders are byte-identical across hash seeds, and a rerun reproduces `promos.json` exactly.
