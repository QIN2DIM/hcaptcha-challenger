# Cryptojunkie signup POC

## Current state
- The local `hcaptcha-challenger -> openai_compat -> codex-slot-relay -> Codex` image lane is live-proven.
- `cryptojunkie.net` is still intermittently unreachable from this runtime, so live end-to-end signup proof depends on target reachability.
- The Playwright harness at `examples/demo_cryptojunkie_signup_poc.py` now supports live mode, fixture mode, and off-domain redirect recovery attempts.

## Archived selector map baked into the POC
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

## Redirect recovery behavior
The harness now assumes a hostile but realistic case where the first visit to the target gets thrown to an off-domain landing page.

Current recovery order:
1. navigate to `CRYPTOJUNKIE_SIGNUP_URL`
2. if the browser lands off-domain, record `offdomain_redirect_detected`
3. try `page.go_back()` / `history.back()`
4. if the page is still off-domain or falls to `about:blank`, reopen the signup URL again so any cookie/session side effect from the first redirect can take effect
5. only continue selector discovery and hCaptcha work after the page is back on an allowed target host

Useful knobs:
- `CRYPTOJUNKIE_TARGET_HOSTS` default: `cryptojunkie.net,www.cryptojunkie.net`
- `CRYPTOJUNKIE_REDIRECT_BACK_ATTEMPTS` default: `3`
- `CRYPTOJUNKIE_REDIRECT_BACK_WAIT_MS` default: `1500`

## What the POC now reports
- selector discovery for the signup form
- hCaptcha widget and iframe presence before and after trigger
- request, response, request-failed, and console summaries
- detected `POST /auth/register` attempts with status, headers, and short response snippet
- redirect recovery attempts under `redirect_recovery_attempts`
- a final state classifier in `final_state`, for example:
  - `target_unreachable`
  - `offdomain_redirect_not_recovered`
  - `offdomain_redirect_active`
  - `fixture_ready_no_submit`
  - `register_post_completed_navigated`
  - `register_post_completed_success_hint`
  - `register_validation_or_captcha_error`
  - `captcha_challenge_not_cleared`
  - `captcha_token_present_but_no_register_post`
  - `submit_attempt_unknown`

## Default live run
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

## Fixture mode for offline validation
Use this when the live site is unreachable but you still want to verify selector discovery, field filling, and artifact generation.

```bash
cd /root/.openclaw/workspace/projects/hcaptcha-challenger-codex

export CRYPTOJUNKIE_FIXTURE_HTML=examples/fixtures/cryptojunkie_register_archived_fixture.html
export CRYPTOJUNKIE_EMAIL='you@example.com'
export CRYPTOJUNKIE_USERNAME='yourname'
export CRYPTOJUNKIE_PASSWORD='strong-password-here'

python3 examples/demo_cryptojunkie_signup_poc.py
```

Notes:
- default fixture behavior skips the real submit and returns `final_state=fixture_ready_no_submit`
- if you want to observe a submit attempt from the fixture anyway, set `CRYPTOJUNKIE_FIXTURE_ALLOW_SUBMIT=1`
- fixture mode is only for local development, not a success oracle for the real site

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
2. if the site throws an off-domain landing page, inspect `redirect_recovery_attempts`
3. run the script with real signup credentials
4. inspect `03_after_trigger.json` for discovered hCaptcha iframe URLs and selector map
5. inspect `04_after_solve.json` for `challenge_signal` and any stored captcha response
6. inspect `05_final.json` for `auth_register_attempts`, `final_page_state`, and `final_state`
7. only call it successful if the register POST and the downstream page effect both match the intended oracle
