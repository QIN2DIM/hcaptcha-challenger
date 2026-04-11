# hcaptcha-challenger x codex-utils ROADMAP

- [done] Fork upstream `QIN2DIM/hcaptcha-challenger` into `IndraLawliet13/hcaptcha-challenger`
- [done] Verify local codex-utils / codex-slot-relay topology and current runtime state
- [done] Add custom provider config surface in hcaptcha-challenger for generic OpenAI-compatible / codex-utils-backed usage
- [done] Patch relay path so image inputs are preserved instead of dropped during OpenAI-compatible translation
- [done] Add docs/examples for Codex relay mode and custom provider mode
- [done] Repair at least one working Codex slot/relay auth path for live multimodal tests
- [done] Smoke-test `hcaptcha-challenger -> codex-slot-relay -> Codex` with real image input and strict JSON output
- [done] Build signup automation POC against `cryptojunkie.net` with hCaptcha handling (`examples/demo_cryptojunkie_signup_poc.py`)
- [done] Harden the Cryptojunkie harness with `POST /auth/register` observation, final-state classification, richer network logging, and fixture-mode offline validation
- [in progress] Run end-to-end verification against live `https://cryptojunkie.net/register` once the target is reachable from the execution runtime
- [pending] Record final success oracle / blockers from a live submit attempt
