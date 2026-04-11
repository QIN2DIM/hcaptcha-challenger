import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from playwright.async_api import Error, Page, TimeoutError, async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hcaptcha_challenger import AgentConfig, AgentV  # noqa: E402


SIGNUP_URL = os.getenv("CRYPTOJUNKIE_SIGNUP_URL", "https://cryptojunkie.net/register")
ARTIFACT_DIR = Path(os.getenv("CRYPTOJUNKIE_ARTIFACT_DIR", "tmp/.challenge/cryptojunkie"))
USER_DATA_DIR = Path(
    os.getenv("CRYPTOJUNKIE_USER_DATA_DIR", "tmp/.cache/cryptojunkie_user_data")
)
TIMEOUT_MS = int(os.getenv("CRYPTOJUNKIE_TIMEOUT_MS", "30000"))
_headless_env = os.getenv("CRYPTOJUNKIE_HEADLESS")
if _headless_env is None:
    HEADLESS = not bool(os.getenv("DISPLAY"))
else:
    HEADLESS = _headless_env.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
BROWSER_EXECUTABLE = os.getenv("CRYPTOJUNKIE_BROWSER_EXECUTABLE") or shutil.which(
    "google-chrome-stable"
) or shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

EMAIL_VALUE = os.getenv("CRYPTOJUNKIE_EMAIL", "")
USERNAME_VALUE = os.getenv("CRYPTOJUNKIE_USERNAME", "")
PASSWORD_VALUE = os.getenv("CRYPTOJUNKIE_PASSWORD", "")

EMAIL_SELECTOR_OVERRIDE = os.getenv("CRYPTOJUNKIE_EMAIL_SELECTOR")
USERNAME_SELECTOR_OVERRIDE = os.getenv("CRYPTOJUNKIE_USERNAME_SELECTOR")
PASSWORD_SELECTOR_OVERRIDE = os.getenv("CRYPTOJUNKIE_PASSWORD_SELECTOR")
CONFIRM_SELECTOR_OVERRIDE = os.getenv("CRYPTOJUNKIE_CONFIRM_PASSWORD_SELECTOR")
SUBMIT_SELECTOR_OVERRIDE = os.getenv("CRYPTOJUNKIE_SUBMIT_SELECTOR")

EMAIL_SELECTORS = [
    "input#email[name='email'][type='email']",
    "input[type='email']",
    "input[name='email']",
    "input[name*='email' i]",
    "input[id='email']",
    "input[id*='email' i]",
    "input[placeholder*='email' i]",
]
USERNAME_SELECTORS = [
    "input#username[name='username'][type='text']",
    "input[name='username']",
    "input[name*='user' i]",
    "input[id='username']",
    "input[id*='user' i]",
    "input[placeholder*='username' i]",
    "input[placeholder*='user name' i]",
    "input[autocomplete='username']",
    "input[type='text']",
]
PASSWORD_SELECTORS = [
    "input#password[name='password'][type='password']",
    "input[type='password'][name='password']",
    "input[type='password'][id='password']",
    "input[type='password'][name*='password' i]",
    "input[type='password'][id*='password' i]",
    "input[type='password']",
]
CONFIRM_PASSWORD_SELECTORS = [
    "input#confirm_password[name='confirm_password'][type='password']",
    "input[type='password'][name*='confirm' i]",
    "input[type='password'][id*='confirm' i]",
    "input[type='password'][name*='repeat' i]",
    "input[type='password'][id*='repeat' i]",
]
CSRF_SELECTORS = [
    "input[name='csrf_token_name']",
]
CAPTCHA_SELECTORS = [
    "select#selectCaptcha[name='captcha']",
    "select[name='captcha']",
]
TOS_SELECTORS = [
    "input[name='tos_agree'][type='checkbox'][value='1']",
    "input[name='tos_agree'][type='checkbox']",
    "input[name='tos_agree']",
]
SUBMIT_SELECTORS = [
    "button.btn.btn-success.btn-block[type='submit']",
    "button[type='submit']",
    "input[type='submit']",
    "button:has-text('Sign Up')",
    "button:has-text('Register')",
    "button:has-text('Create Account')",
    "button:has-text('Submit')",
    "form button",
]


async def first_selector(page: Page, selectors: list[str], *, visible: bool = True) -> str | None:
    for selector in selectors:
        locator = page.locator(selector)
        try:
            count = await locator.count()
        except Error:
            continue
        if count == 0:
            continue
        if not visible:
            return selector
        try:
            await locator.first.wait_for(state="visible", timeout=1200)
            return selector
        except TimeoutError:
            continue
        except Error:
            continue
    return None


async def fill_if_present(page: Page, selector: str | None, value: str) -> bool:
    if not selector or not value:
        return False
    locator = page.locator(selector).first
    await locator.click(timeout=TIMEOUT_MS)
    await locator.fill(value, timeout=TIMEOUT_MS)
    return True


async def check_if_present(page: Page, selector: str | None) -> bool:
    if not selector:
        return False
    locator = page.locator(selector).first
    await locator.check(timeout=TIMEOUT_MS)
    return True


async def select_if_present(page: Page, selector: str | None, value: str) -> bool:
    if not selector:
        return False
    locator = page.locator(selector).first
    await locator.select_option(value=value, timeout=TIMEOUT_MS)
    return True


async def collect_frame_urls(page: Page) -> list[str]:
    urls: list[str] = []
    for frame in page.frames:
        if frame.url:
            urls.append(frame.url)
    return urls


async def detect_hcaptcha(page: Page) -> dict[str, Any]:
    iframe_selector = "iframe[src*='hcaptcha.com'], iframe[src*='newassets.hcaptcha.com']"
    textarea_selector = "textarea[name='h-captcha-response']"
    widget_selector = ".h-captcha, [data-sitekey]"

    iframe_count = await page.locator(iframe_selector).count()
    textarea_count = await page.locator(textarea_selector).count()
    widget_count = await page.locator(widget_selector).count()

    textarea_value = ""
    if textarea_count:
        try:
            textarea_value = await page.locator(textarea_selector).first.evaluate("el => el.value || ''")
        except Error:
            textarea_value = ""

    frame_urls = [url for url in await collect_frame_urls(page) if "hcaptcha" in url]

    return {
        "iframe_count": iframe_count,
        "textarea_count": textarea_count,
        "widget_count": widget_count,
        "textarea_value_len": len(textarea_value),
        "frame_urls": frame_urls,
        "detected": bool(iframe_count or textarea_count or widget_count or frame_urls),
    }


async def wait_for_hcaptcha_markup(page: Page, timeout_ms: int = 10000) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + (timeout_ms / 1000)
    last_state = await detect_hcaptcha(page)
    while asyncio.get_running_loop().time() < deadline:
        if last_state["detected"]:
            return last_state
        await page.wait_for_timeout(500)
        last_state = await detect_hcaptcha(page)
    return last_state


async def dump_artifacts(page: Page, prefix: str, summary: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = ARTIFACT_DIR / f"{prefix}.html"
    png_path = ARTIFACT_DIR / f"{prefix}.png"
    json_path = ARTIFACT_DIR / f"{prefix}.json"

    try:
        html_path.write_text(await page.content(), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        summary.setdefault("artifact_errors", []).append(f"html:{exc}")

    try:
        await page.screenshot(path=str(png_path), full_page=True)
    except Exception as exc:  # noqa: BLE001
        summary.setdefault("artifact_errors", []).append(f"screenshot:{exc}")

    try:
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        summary.setdefault("artifact_errors", []).append(f"json:{exc}")


async def build_selector_map(page: Page) -> dict[str, str | None]:
    return {
        "email": EMAIL_SELECTOR_OVERRIDE or await first_selector(page, EMAIL_SELECTORS),
        "username": USERNAME_SELECTOR_OVERRIDE or await first_selector(page, USERNAME_SELECTORS),
        "password": PASSWORD_SELECTOR_OVERRIDE or await first_selector(page, PASSWORD_SELECTORS),
        "confirm_password": CONFIRM_SELECTOR_OVERRIDE
        or await first_selector(page, CONFIRM_PASSWORD_SELECTORS),
        "csrf": await first_selector(page, CSRF_SELECTORS, visible=False),
        "captcha_select": await first_selector(page, CAPTCHA_SELECTORS),
        "tos_agree": await first_selector(page, TOS_SELECTORS),
        "submit": SUBMIT_SELECTOR_OVERRIDE or await first_selector(page, SUBMIT_SELECTORS),
    }


async def main() -> None:
    summary: dict[str, Any] = {
        "signup_url": SIGNUP_URL,
        "artifact_dir": str(ARTIFACT_DIR),
        "headless": HEADLESS,
        "timeout_ms": TIMEOUT_MS,
        "mode": "recon",
        "status": "starting",
        "browser_executable": BROWSER_EXECUTABLE,
        "network_requests": [],
        "network_responses": [],
    }

    has_submit_payload = bool(EMAIL_VALUE and USERNAME_VALUE and PASSWORD_VALUE)
    if has_submit_payload:
        summary["mode"] = "submit"

    try:
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(USER_DATA_DIR),
                headless=HEADLESS,
                locale="en-US",
                executable_path=BROWSER_EXECUTABLE,
            )
            page = context.pages[0] if context.pages else await context.new_page()
            page.set_default_timeout(TIMEOUT_MS)

            def on_request(request) -> None:
                summary["network_requests"].append(
                    {
                        "method": request.method,
                        "resource_type": request.resource_type,
                        "url": request.url,
                    }
                )
                del summary["network_requests"][:-80]

            def on_response(response) -> None:
                summary["network_responses"].append(
                    {
                        "status": response.status,
                        "url": response.url,
                    }
                )
                del summary["network_responses"][:-80]

            page.on("request", on_request)
            page.on("response", on_response)

            try:
                await page.goto(SIGNUP_URL, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
                summary["status"] = "page_loaded"
            except Exception as exc:  # noqa: BLE001
                summary["status"] = "goto_failed"
                summary["error"] = f"{type(exc).__name__}: {exc}"
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                await context.close()
                return

            summary["page_url"] = page.url
            summary["page_title"] = await page.title()
            summary["selector_map"] = await build_selector_map(page)
            summary["hcaptcha_before"] = await detect_hcaptcha(page)
            await dump_artifacts(page, "01_initial", summary)

            if has_submit_payload:
                filled = {
                    "email": await fill_if_present(page, summary["selector_map"]["email"], EMAIL_VALUE),
                    "username": await fill_if_present(
                        page, summary["selector_map"]["username"], USERNAME_VALUE
                    ),
                    "password": await fill_if_present(
                        page, summary["selector_map"]["password"], PASSWORD_VALUE
                    ),
                    "confirm_password": False,
                    "captcha_select": False,
                    "tos_agree": False,
                }
                if summary["selector_map"]["confirm_password"]:
                    filled["confirm_password"] = await fill_if_present(
                        page,
                        summary["selector_map"]["confirm_password"],
                        PASSWORD_VALUE,
                    )
                if summary["selector_map"]["captcha_select"]:
                    try:
                        filled["captcha_select"] = await select_if_present(
                            page,
                            summary["selector_map"]["captcha_select"],
                            "hcaptcha",
                        )
                    except Exception as exc:  # noqa: BLE001
                        summary["captcha_select_error"] = f"{type(exc).__name__}: {exc}"
                if summary["selector_map"]["tos_agree"]:
                    try:
                        filled["tos_agree"] = await check_if_present(
                            page,
                            summary["selector_map"]["tos_agree"],
                        )
                    except Exception as exc:  # noqa: BLE001
                        summary["tos_agree_error"] = f"{type(exc).__name__}: {exc}"
                summary["filled_fields"] = filled
                await dump_artifacts(page, "02_filled", summary)

                agent_config = AgentConfig(
                    AGENT_BACKEND=os.getenv("AGENT_BACKEND", "openai_compat"),
                    OPENAI_COMPAT_BASE_URL=os.getenv(
                        "OPENAI_COMPAT_BASE_URL",
                        os.getenv("CODEX_BASE_URL", "http://127.0.0.1:8787"),
                    ),
                    OPENAI_COMPAT_API_KEY=os.getenv(
                        "OPENAI_COMPAT_API_KEY",
                        os.getenv("CODEX_API_KEY", "relay-dev-token"),
                    ),
                    OPENAI_COMPAT_MODEL=os.getenv(
                        "OPENAI_COMPAT_MODEL",
                        os.getenv("CODEX_MODEL", "gpt-5.4"),
                    ),
                    OPENAI_COMPAT_TIMEOUT=float(
                        os.getenv("OPENAI_COMPAT_TIMEOUT", os.getenv("CODEX_TIMEOUT", "180"))
                    ),
                )
                agent = AgentV(page=page, agent_config=agent_config)

                submit_selector = summary["selector_map"]["submit"]
                initial_hcaptcha = summary["hcaptcha_before"]

                if initial_hcaptcha["iframe_count"] > 0:
                    summary["trigger"] = "checkbox"
                    await agent.robotic_arm.click_checkbox()
                elif submit_selector:
                    summary["trigger"] = "submit"
                    await page.locator(submit_selector).first.click(delay=120)
                else:
                    summary["status"] = "submit_selector_missing"
                    print(json.dumps(summary, indent=2, ensure_ascii=False))
                    await context.close()
                    return

                post_trigger_hcaptcha = await wait_for_hcaptcha_markup(page)
                summary["hcaptcha_after_trigger"] = post_trigger_hcaptcha
                await dump_artifacts(page, "03_after_trigger", summary)

                if post_trigger_hcaptcha["detected"]:
                    signal = await agent.wait_for_challenge()
                    summary["challenge_signal"] = getattr(signal, "value", str(signal))
                    if agent.cr_list:
                        summary["captcha_response"] = agent.cr_list[-1].model_dump(
                            mode="json", by_alias=True
                        )
                    await dump_artifacts(page, "04_after_solve", summary)
                else:
                    summary["challenge_signal"] = "no_hcaptcha_markup_detected"

                if submit_selector:
                    try:
                        await page.locator(submit_selector).first.click(delay=120)
                        summary["submit_retried"] = True
                    except Exception as exc:  # noqa: BLE001
                        summary["submit_retried"] = False
                        summary["submit_retry_error"] = f"{type(exc).__name__}: {exc}"

                await page.wait_for_timeout(3000)
                summary["page_url_after_submit"] = page.url
                summary["page_title_after_submit"] = await page.title()
                summary["hcaptcha_after_submit"] = await detect_hcaptcha(page)
                summary["status"] = "submit_attempt_finished"
                await dump_artifacts(page, "05_final", summary)

            print(json.dumps(summary, indent=2, ensure_ascii=False))
            await context.close()
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "fatal_error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
