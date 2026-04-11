# Cryptojunkie signup POC

## Current state
- The local `hcaptcha-challenger -> openai_compat -> codex-slot-relay -> Codex` image lane is live-proven.
- `cryptojunkie.net` itself was not reachable from this runtime during recon, so live DOM extraction and end-to-end signup proof are still blocked by reachability.
- A reusable Playwright harness now exists at `examples/demo_cryptojunkie_signup_poc.py` so the target can be tested immediately once the page is reachable.

## Archived selector map currently baked into the POC
- signup page: `https://cryptojunkie.net/register`
- form: `form.form-horizontal[action="https://cryptojunkie.net/auth/register"][method="POST"]`
- email: `input#email[name="email"][type="email"]`
- username: `input#username[name="username"][type="text"]`
- password: `input#password[name="password"][type="password"]`
- confirm password: `input#confirm_password[name="confirm_password"][type="password"]`
- csrf: `input[name="csrf_token_name"]`
- captcha select: `select#selectCaptcha[name="captcha"]`
- tos checkbox: `input[name="tos_agree"][type="checkbox"][value="1"]`
- submit: `button.btn.btn-success.btn-block[type="submit"]`
- hCaptcha widget: `div#hcaptcha.captcha > div.h-captcha[data-sitekey="4f75f835-7c6c-4030-98d1-539aa73ce089"]`

## What the POC does
- opens the signup page
- auto-discovers likely email, username, password, confirm-password, and submit selectors
- records HTML, screenshots, and network traces under `tmp/.challenge/cryptojunkie/`
- detects hCaptcha markup and iframe URLs
- triggers the challenge either through the visible checkbox or the submit button
- hands control to `AgentV` for solving
- optionally retries submit after a successful challenge
- prints a JSON summary with the observed selectors, hCaptcha state, and final page state

## Default run
```bash
cd /root/.openclaw/workspace/projects/hcaptcha-challenger-codex

export AGENT_BACKEND=openai_compat
export OPENAI_COMPAT_BASE_URL=${OPENAI_COMPAT_BASE_URL:-${CODEX_BASE_URL:-http://127.0.0.1:8787}}
export OPENAI_COMPAT_API_KEY=${OPENAI_COMPAT_API_KEY:-${CODEX_API_KEY:-relay-dev-token}}
export OPENAI_COMPAT_MODEL=${OPENAI_COMPAT_MODEL:-${CODEX_MODEL:-gpt-5.4}}

export CRYPTOJUNKIE_EMAIL='you@example.com'
export CRYPTOJUNKIE_USERNAME='yourname'
export CRYPTOJUNKIE_PASSWORD='strong-password-here'

python3 examples/demo_cryptojunkie_signup_poc.py
```

## Optional selector overrides
If the page uses unusual field names, override them directly:
```bash
export CRYPTOJUNKIE_EMAIL_SELECTOR="input[name='email_address']"
export CRYPTOJUNKIE_USERNAME_SELECTOR="input[name='user_name']"
export CRYPTOJUNKIE_PASSWORD_SELECTOR="input[name='passwd']"
export CRYPTOJUNKIE_CONFIRM_PASSWORD_SELECTOR="input[name='passwd_confirm']"
export CRYPTOJUNKIE_SUBMIT_SELECTOR="button[type='submit']"
```

## Artifacts
The script writes these artifacts under `tmp/.challenge/cryptojunkie/`:
- `01_initial.*`
- `02_filled.*`
- `03_after_trigger.*`
- `04_after_solve.*`
- `05_final.*`

Each stage keeps:
- full-page screenshot
- page HTML dump
- JSON summary snapshot

## Live verification checklist
1. confirm the page is reachable from the runtime that executes the script
2. run the script with real signup credentials
3. inspect `03_after_trigger.json` for discovered hCaptcha iframe URLs and selector map
4. inspect `04_after_solve.json` for `challenge_signal` and any stored captcha response
5. inspect `05_final.json` and network traces for the real signup success oracle
