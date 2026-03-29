"""
Clintelligence - Modern FastAPI Backend
AI-Powered Protocol Intelligence Platform
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Initialize app
app = FastAPI(
    title="Clintelligence",
    description="AI-Powered Protocol Intelligence Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ============== MODELS ==============
class ProtocolInput(BaseModel):
    protocol_text: str
    max_similar: int = 500
    min_similarity: int = 40
    rank_top_n: int = 100


class AnalysisResponse(BaseModel):
    success: bool
    extracted_protocol: Optional[Dict] = None
    risk_assessment: Optional[Dict] = None
    similar_trials: Optional[List] = None
    metrics: Optional[Dict] = None
    site_recommendations: Optional[List] = None
    error: Optional[str] = None


# ============== HELPER FUNCTIONS ==============
def get_db_stats():
    """Get database statistics."""
    return {"total_trials": 566622, "status": "demo"}


# ============== PAGES ==============
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - Protocol Entry."""
    stats = get_db_stats()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "home",
        "stats": stats
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Analysis Dashboard."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "page": "dashboard"
    })


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About page."""
    return templates.TemplateResponse("about.html", {
        "request": request,
        "page": "about"
    })


@app.get("/find-trials", response_class=HTMLResponse)
async def find_trials(request: Request):
    """Patient trial finder page."""
    return templates.TemplateResponse("find_trials.html", {
        "request": request,
        "page": "find_trials",
        "recruiting_count": 95000
    })


# ============== API ENDPOINTS ==============
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_protocol(input_data: ProtocolInput):
    """Analyze a protocol - placeholder for demo."""
    if not input_data.protocol_text or len(input_data.protocol_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Protocol text must be at least 100 characters")

    # Demo response
    return AnalysisResponse(
        success=True,
        extracted_protocol={
            "condition": "Demo Condition",
            "phase": "Phase 3",
            "target_enrollment": 500,
            "primary_endpoint": "Demo Endpoint"
        },
        risk_assessment={
            "overall_risk": "Medium",
            "amendment_probability": 0.35,
            "delay_probability": 0.40
        },
        similar_trials=[],
        metrics={"trials_analyzed": 566622},
        site_recommendations=[]
    )


@app.get("/api/stats")
async def get_stats():
    """Get platform statistics."""
    return get_db_stats()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "clintelligence"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
