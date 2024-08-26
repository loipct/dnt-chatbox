from fastapi import FastAPI
import argparse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from web import search

app = FastAPI()

app.mount("/static", StaticFiles(directory=r"view", html=True), name="static")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(search.router)

@app.get("/")
def get() -> str:
    return "running"


if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Run FastAPI server with specified port")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server (default: 8000)")
    args = parser.parse_args()

    uvicorn.run("main:app", port=args.port, reload=True)