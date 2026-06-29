# GemmaBot — 60-second voiceover

**Total: 60 s. ~150 words. Read at a normal pace — do NOT rush.**

Numbers in bold are real measurements from this session — see source file
in parens.

---

## [0:00 – 0:08] OPEN — show `nocheat_Lift.mp4`

> A Franka Panda arm. A red cube.
> Nobody told the robot where the cube is.
> Gemma 4 looked at one frame, identified the cube, the arm picked it up.

---

## [0:08 – 0:20] ARCHITECTURE — show `nocheat_Stack.mp4` OR a slide

> The loop: camera → perception finds objects in 3-D to **one centimeter**
> → Gemma reads the image and chooses the target → executor grasps.
> Ground truth is used only by the judge — never by Gemma.

---

## [0:20 – 0:45] CEREBRAS vs OPENROUTER — hold on `comparison_chart.png`

> Same model. Same prompts. Same images. Only the silicon changes.
> Twenty calls per provider, measured live.
> Text: Cerebras **three hundred** milliseconds, OpenRouter **four** seconds.
> Thirteen times faster.
> Vision: **two hundred** milliseconds vs **eleven hundred**. Five times
> faster at p50, **thirty-nine** times at the tail.
> For a closed-loop robot, that's the difference between reactive and useless.

---

## [0:45 – 0:55] HONEST CAVEATS — slide or hold on chart

> On an un-gameable spatial test, Gemma scores **eighty-eight percent**.
> Where it slips: stringing four ordered actions together, **thirty-six percent**.
> We name the weak spot.

---

## [0:55 – 1:00] CLOSE

> Honest pipeline, honest numbers, no ground truth in the loop.
> Code's in the repo.

---

# CHEAT-SHEET — say one of these per shot, that's it

| Shot | One-line |
|---|---|
| Lift video | "Gemma identified the cube, the arm picked it up — 600 ms." |
| Chart left  | "**13× faster** on text." |
| Chart mid   | "**5× faster** on vision — **39× at the tail**." |
| Chart right | "Same model, only the silicon changes." |
| Fail clip   | "At low resolution it confuses items — we name the limit." |
| Close       | "Honest pipeline. Code's in the repo." |
