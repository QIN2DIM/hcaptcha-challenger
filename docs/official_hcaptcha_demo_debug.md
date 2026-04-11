# Official hCaptcha demo debug notes

Date: 2026-04-11

## Goal
Answer the narrow question: does the new `hcaptcha-challenger -> openai_compat -> codex-slot-relay -> Codex` solver stack actually work on a reachable real hCaptcha page.

## Target
- Official demo: `https://accounts.hcaptcha.com/demo?sitekey=c86d730b-300a-444c-a8c5-5312e7a93628`
- Browser lane: real Chrome under `xvfb-run`

## Baseline bounded run
Command shape used:
```bash
xvfb-run -a python3 tmp/root-scratch/test_hcaptcha_demo_solver.py
```

Observed behavior:
- page loaded normally
- checkbox iframe was present
- solver entered real challenge rounds
- the run repeatedly hit `image_drag_single`
- later rounds also included `image_label_multi_select`
- multi-select produced real coordinates through the new OpenAI-compatible/Codex lane, but those rounds were still rejected
- after the bounded budget, there was no accepted captcha token and no final pass oracle

## Isolation run: skip drag-single lane
Command shape used:
```bash
xvfb-run -a env \
  HCAPTCHA_IGNORE_TYPES=image_drag_single,image_drag_drop,image_label_single_select \
  python3 tmp/root-scratch/test_hcaptcha_demo_solver.py
```

Observed behavior:
- the solver still saw some initial `image_drag_single` rounds, but those were ignored and refreshed
- later the challenge rotated into `image_label_multi_select`
- those multi-select rounds were still somewhat noisy and not immediately accepted
- the challenge then rotated into `image_label_binary`
- binary classification produced a valid click set and the demo returned `Challenge success`

Key runtime evidence:
- `wait_for_challenge:1046 | Challenge success`
- success arrived around `87s` into the bounded run

## Practical conclusion
The new multimodal backend and relay pipeline are alive and can pass a real official hCaptcha demo.

The strongest current blocker is **not** the whole pipeline. It is the **drag-single lane**:
- `image_drag_single` is the dominant failure source in the official demo runs
- `image_label_multi_select` is imperfect but not a total blocker
- `image_label_binary` is good enough to complete the demo after challenge rotation when drag-single is skipped

## Honest verdict
- `pipeline works`: **yes**
- `official demo pass with all challenge variants enabled`: **not yet reliable**
- `official demo pass after excluding drag-single`: **yes**

## Highest-leverage next debugging targets
1. instrument `image_drag_single` so each predicted drag path is recorded before actuation
2. compare predicted path against actual accepted/rejected rounds
3. decide whether drag-single should be improved or temporarily treated as unsupported in practical test harnesses
4. add a first-class bounded demo runner with explicit oracles so future regressions are easy to check
