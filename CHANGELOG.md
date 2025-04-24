## v0.16.1

- feat(challenge): compatible with [Rebrowser API](https://rebrowser.net/) ([#1028](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1028))

  ```python
  # Example
  
  import asyncio
  import json
  
  import dotenv
  from hcaptcha_challenger.agent import AgentV, AgentConfig
  from hcaptcha_challenger.models import CaptchaResponse
  from rebrowser_playwright.async_api import async_playwright, Page
  
  dotenv.load_dotenv()
  
  
  async def challenge(page: Page) -> AgentV:
      agent_config = AgentConfig()
      agent = AgentV(page=page, agent_config=agent_config)
      await agent.robotic_arm.click_checkbox()
      await agent.wait_for_challenge()
      return agent
  
  
  async def main():
      async with async_playwright() as p:
          browser = await p.chromium.launch(headless=False)
          page = await browser.new_page()
          await page.goto("https://accounts.hcaptcha.com/demo")
  
          # --- When you encounter hCaptcha in your workflow ---
          agent: AgentV = await challenge(page)
          if agent.cr_list:
              cr: CaptchaResponse = agent.cr_list[-1]
              print(json.dumps(cr.model_dump(by_alias=True), indent=2, ensure_ascii=False))
  
  
  if __name__ == "__main__":
      asyncio.run(main())
  ```

  

## v0.16.0

- feat(challenge): support image_drag_multi ([#1026](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1026))

## v0.15.7

- feat(cli): cost_calculator ([#1024](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1024))

- feat(helper): add cost calculator ([#1024](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1024))

- feat(challenge): save model answer ([#1023](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1023))

## v0.15.6

- feat(challenge): replace bad code ([#1022](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1022))

## v0.15.5

- feat(challenge): support gemini-2.5-flash-preview-04-17 ([#1021](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1021))

## v0.15.4

- feat(agent-config): add params `ignore_request_questions`([#1020](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1020))

  ```python
  # Example
  agent_config = AgentConfig(
      ignore_request_questions=["Select objects that fit the theme of the shown image"]
  )
  ```

## v0.15.3

- feat(challenge): handle nested frames ([#1019](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1019))
- fix(cli): add dataset check command ([#1015](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1015))

## v0.15.0

- feat(cli): create dataset management tool ([#1013](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1013))

## v0.14.7

- feat(challenge): improve the model's attention on object detection tasks

## v0.14.6

- fix(tools): Gemini model's response_schema ([#1012](https://github.com/QIN2DIM/hcaptcha-challenger/issues/1012))