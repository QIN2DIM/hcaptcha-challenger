from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from routers import challenge_router, datalake_router

app = FastAPI()

app.include_router(challenge_router, prefix="/challenge")
app.include_router(datalake_router, prefix="/datalake")


@app.get("/")
async def home():
    return RedirectResponse(url="https://github.com/QIN2DIM/hcaptcha-challenger")


@app.get("/ping", response_model=str)
async def ping():
    return "pong"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=33777)
