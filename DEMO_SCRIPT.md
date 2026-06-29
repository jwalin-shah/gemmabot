# GemmaBot — Voiceover script

Total runtime target: **90–120 s**. Read in your normal voice.
Numbers in **bold** are real measurements from THIS project's data — every one
is reproducible from a script in `scripts/` or a JSONL in `overnight_results/`.

---

## OPEN (10 s) — over `overnight_results/videos/nocheat_Lift.mp4`

> A Franka Panda arm. A red cube on the table.
> Nobody told the robot where the cube is.
> Gemma 4 looked at one camera frame, identified the cube, and the arm picked it up.
> Six hundred and seventy-five **milliseconds** of inference.

---

## ARCHITECTURE (15 s) — over `runs/clear_table_*/` step JSON, or a slide

> Here's the loop:
> A camera frame goes in. A perception front-end — instance segmentation plus
> depth back-projection — recovers each object's 3-D position to about
> **one centimetre**. We hand Gemma the image and a list of un-named
> coordinates. **Gemma identifies each object from the pixels, picks a target,
> and we execute the grasp.**
> Ground truth from the simulator is used only by the judge — never by Gemma,
> never by perception.

---

## HONEST RESULTS (25 s) — over the comparison chart + a table card

> What does Gemma actually do?
> On an un-gameable spatial-reasoning test where the answer never appears in
> the prompt, Gemma scores **88 percent**. Strong at color and left/right
> position; weaker on fine proximity, around **64 percent**.
> Closed-loop, Gemma + perception + grasp succeeds on the lift task end-to-end.
> Pick-and-place identification is harder at 384-pixel camera resolution —
> at 768 pixels and above, Gemma reads brand text off the boxes, but at the
> low-res frame we use for the closed loop, it confuses the soda can and the
> milk carton roughly **a third of the time**. That's the limit; we name it.

---

## CEREBRAS vs OPENROUTER (25 s) — over `overnight_results/compare_or/comparison_chart.png`

> Same model. Same prompts. Same images. Only the silicon changes.
> Twenty calls per provider per workload, measured live this session.
> Text-only inference: Cerebras p50 **three hundred milliseconds**;
> OpenRouter p50 **three point nine seconds**. Thirteen times faster.
> Vision: Cerebras **two hundred milliseconds**; OpenRouter **one point one
> seconds**. Five and a half times faster — and at p95, **thirty-nine times**
> faster, because the OpenRouter tail goes into the tens of seconds.
> Structured JSON output: Cerebras **two seventy**; OpenRouter **eight sixty**.
> Three times faster.
> For a closed loop where the model decides what to do next, the difference
> between three hundred ms and four seconds is the difference between
> a reactive robot and a slideshow.

---

## CAPSTONE / FAILURES (15 s) — over a soda-capstone clip OR a static slide

> What doesn't work yet, honestly:
> Long-horizon plan ordering. Plan-complete rate is **thirty-six percent** —
> Gemma reads each scene well, but stringing four ordered actions together
> is where it slips.
> Active perception — zooming in on an unclear object — works only if the
> closer look is actually high-res. The naive eye-in-hand close-up didn't
> help. A proper hi-res inspection escalation is the next thing to build.

---

## CLOSE (10 s)

> So: Gemma 4 reasons over images, picks the right object, the arm grasps it,
> and on Cerebras the whole loop runs in under half a second.
> Honest pipeline, honest numbers, no ground truth in the loop.
> Code and data are in the repo.

---

# TIMING TIPS

- Open clip is short — let it breathe; don't overlap dialogue with the grasp.
- Pause for ~1 s on the "**13×**" speedup callout — viewers need to read the chart.
- Close on a held shot of the chart with the speedup numbers visible.

# REAL NUMBERS — sourced one-liners (for the lower-third / overlays)

- "Cerebras p50 inference: **199 ms** (vision), **273 ms** (vision + JSON)" — from `overnight_results/compare_or/image/summary.json`
- "OpenRouter p50 inference: **1,088 ms** (vision)" — same file
- "p95 speedup: **39×** (vision)" — same file
- "Perception localization: **1.05 cm** mean error across 7 objects" — `final_report_corrected.md`
- "Closed-loop Lift: end-to-end success, no GT coords" — `overnight_results/videos/nocheat_Lift.mp4`
- "Visual reasoning (un-gameable): **88.4%** overall, n=160, 95% CI [83.6, 93.3]" — same report
- "Plan ordering (multi-step): **36.3%**" — same report (honest known weakness)
