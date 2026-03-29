"""
Vercel Serverless Entry Point - Simplified for Vercel deployment
"""
import os
import sys
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# Initialize app
app = FastAPI(
    title="Clintelligence",
    description="AI-Powered Protocol Intelligence Platform",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
TEMPLATES_DIR = ROOT_DIR / "web_app" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_db_stats():
    return {"total_trials": 566622, "status": "demo"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    stats = get_db_stats()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "home",
        "stats": stats
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "page": "dashboard"
    })


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {
        "request": request,
        "page": "about"
    })


@app.get("/find-trials", response_class=HTMLResponse)
async def find_trials(request: Request):
    return templates.TemplateResponse("find_trials.html", {
        "request": request,
        "page": "find_trials",
        "recruiting_count": 95000
    })


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "clintelligence"}


# Vercel handler
handler = Mangum(app, lifespan="off")
