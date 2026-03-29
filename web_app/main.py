"""
ClinAnalytica - Modern FastAPI Backend
AI-Powered Protocol Intelligence Platform
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import tempfile
from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent.parent / '.env')

app = FastAPI(
    title="ClinAnalytica",
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
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


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


# Patient Trial Matching Models
class TrialSearchInput(BaseModel):
    condition: str
    age: Optional[int] = None
    location: Optional[str] = None
    phase: Optional[str] = None


class TrialMatchInput(BaseModel):
    condition: str
    answers: Dict[str, Any]


# ============== DATABASE ==============
def get_database():
    """Get database connection."""
    try:
        from src.database import DatabaseManager
        return DatabaseManager.get_instance()
    except Exception as e:
        print(f"Database error: {e}")
        return None


def get_db_stats():
    """Get database statistics."""
    db = get_database()
    if db:
        return db.get_stats()
    return {"total_trials": 566622, "status": "cached"}


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
    db = get_database()
    recruiting_count = 0
    if db:
        try:
            from sqlalchemy import text
            with db.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM trials WHERE status IN ('RECRUITING', 'NOT_YET_RECRUITING')"
                ))
                recruiting_count = result.scalar()
        except:
            recruiting_count = 95000  # Fallback estimate

    return templates.TemplateResponse("find_trials.html", {
        "request": request,
        "page": "find_trials",
        "recruiting_count": recruiting_count
    })


# ============== API ENDPOINTS ==============
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_protocol(input_data: ProtocolInput):
    """Analyze a protocol and return comprehensive intelligence."""

    if not input_data.protocol_text or len(input_data.protocol_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Protocol text must be at least 100 characters")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    db = get_database()
    if not db:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        from src.analysis.protocol_analyzer import ProtocolAnalyzer

        analyzer = ProtocolAnalyzer()
        results = analyzer.analyze_and_match(
            protocol_text=input_data.protocol_text,
            db_manager=db,
            include_site_recommendations=True,
            min_similarity=input_data.min_similarity,
            max_candidates=input_data.max_similar,
            semantic_rank_top_n=input_data.rank_top_n
        )

        # Convert dataclass to dict for JSON serialization
        extracted = results["extracted_protocol"]
        extracted_dict = {
            "condition": extracted.condition,
            "sponsor": getattr(extracted, 'sponsor', ''),  # Include sponsor information
            "phase": extracted.phase,
            "target_enrollment": extracted.target_enrollment,
            "primary_endpoint": extracted.primary_endpoint,
            "secondary_endpoints": extracted.secondary_endpoints,
            "inclusion_criteria": extracted.key_inclusion,
            "exclusion_criteria": extracted.key_exclusion,
            "intervention_type": extracted.intervention_type,
            "intervention_name": extracted.intervention_name,
            "comparator": extracted.comparator,
            "study_duration_months": extracted.study_duration_months,
            "study_type": extracted.study_type,
        }

        return AnalysisResponse(
            success=True,
            extracted_protocol=extracted_dict,
            risk_assessment=results.get("risk_assessment", {}),
            similar_trials=results.get("similar_trials", [])[:20],  # Limit for response size
            metrics=results.get("metrics", {}),
            site_recommendations=results.get("site_recommendations", [])[:10]
        )

    except Exception as e:
        return AnalysisResponse(
            success=False,
            error=str(e)
        )


# ============== IMPROVED ANALYSIS ENDPOINT ==============
class ProtocolInputV2(BaseModel):
    protocol_text: str
    min_score: int = 50
    max_results: int = 50
    use_reranking: bool = True


@app.post("/api/analyze-v2")
async def analyze_protocol_v2(input_data: ProtocolInputV2):
    """
    Analyze protocol using improved multi-dimensional matching.

    This endpoint uses:
    - Enhanced structured extraction
    - Multi-query vector search
    - Multi-dimensional scoring (condition, intervention, endpoint, population, design)
    - Hard filters for incompatible trials
    - Claude-powered intelligent reranking
    """
    if not input_data.protocol_text or len(input_data.protocol_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Protocol text must be at least 100 characters")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    db = get_database()
    if not db:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        from src.analysis.improved_matcher import ImprovedTrialMatcher

        matcher = ImprovedTrialMatcher(db)
        protocol, matches = matcher.find_similar_trials(
            protocol_text=input_data.protocol_text,
            min_score=input_data.min_score,
            max_results=input_data.max_results,
            use_reranking=input_data.use_reranking
        )

        # Format extracted protocol
        extracted_dict = {
            "condition": protocol.condition,
            "condition_category": protocol.condition_category,
            "therapeutic_area": protocol.therapeutic_area,
            "sponsor": protocol.sponsor,  # Include sponsor information
            "phase": protocol.design.phase,
            "target_enrollment": protocol.design.target_enrollment,
            "duration_weeks": protocol.design.duration_weeks,
            "primary_endpoint": protocol.endpoints.primary_endpoint,
            "intervention": {
                "type": protocol.intervention.intervention_type,
                "drug_name": protocol.intervention.drug_name,
                "drug_class": protocol.intervention.drug_class,
                "mechanism": protocol.intervention.mechanism_of_action,
                "route": protocol.intervention.route,
                "frequency": protocol.intervention.frequency,
                "similar_drugs": protocol.intervention.similar_known_drugs,
            },
            "population": {
                "min_age": protocol.population.min_age,
                "max_age": protocol.population.max_age,
                "excluded_conditions": protocol.population.excluded_conditions,
                "required_conditions": protocol.population.required_conditions,
            },
        }

        # Format matched trials
        similar_trials = [m.to_dict() for m in matches]

        # Get summary
        summary = matcher.get_summary(matches)

        # Generate comprehensive dashboard data
        from src.analysis.dashboard_analyzer import DashboardAnalyzer
        dashboard_analyzer = DashboardAnalyzer(db)
        dashboard_data = dashboard_analyzer.analyze_for_dashboard(
            protocol=protocol,
            similar_trials=matches,
            protocol_text=input_data.protocol_text
        )

        return {
            "success": True,
            "extracted_protocol": extracted_dict,
            "similar_trials": similar_trials,
            "summary": summary,
            "matching_version": "v2-multidimensional",
            "dashboard": dashboard_data
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and extract text from a PDF protocol document."""

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Only PDF files are supported"}
        )

    # Validate file size (10MB limit)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "File size must be less than 10MB"}
        )

    try:
        # Try pdfplumber first
        try:
            import pdfplumber

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name

            text_parts = []
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            os.unlink(tmp_path)
            extracted_text = "\n\n".join(text_parts)

        except ImportError:
            # Fall back to PyMuPDF
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(stream=contents, filetype="pdf")
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                extracted_text = "\n\n".join(text_parts)

            except ImportError:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": "PDF processing library not available. Please install pdfplumber or PyMuPDF."}
                )

        if not extracted_text or len(extracted_text.strip()) < 50:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Could not extract sufficient text from PDF. The file may be scanned or image-based."}
            )

        return {
            "success": True,
            "text": extracted_text,
            "characters": len(extracted_text),
            "filename": file.filename
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error processing PDF: {str(e)}"}
        )


@app.post("/api/upload-and-analyze")
async def upload_and_analyze_pdf(
    file: UploadFile = File(...),
    min_score: int = 50,
    max_results: int = 50,
    use_reranking: bool = True
):
    """
    P2 IMPROVEMENT: Upload a PDF protocol and run full analysis in one step.
    Extracts text, runs protocol matching, and returns dashboard data.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Only PDF files are supported"}
        )

    # Validate file size (10MB limit)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "File size must be less than 10MB"}
        )

    try:
        # Extract text from PDF
        extracted_text = ""
        try:
            import pdfplumber
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name

            text_parts = []
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            os.unlink(tmp_path)
            extracted_text = "\n\n".join(text_parts)
        except ImportError:
            try:
                import fitz
                doc = fitz.open(stream=contents, filetype="pdf")
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                extracted_text = "\n\n".join(text_parts)
            except ImportError:
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": "PDF processing library not available"}
                )

        if not extracted_text or len(extracted_text.strip()) < 100:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Could not extract sufficient text from PDF"}
            )

        # Run full analysis
        db = get_database()
        if not db:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Database connection failed"}
            )

        from src.analysis.improved_matcher import ImprovedTrialMatcher
        from src.analysis.dashboard_analyzer import DashboardAnalyzer

        matcher = ImprovedTrialMatcher(db)
        protocol, matches = matcher.find_similar_trials(
            protocol_text=extracted_text,
            min_score=min_score,
            max_results=max_results,
            use_reranking=use_reranking
        )

        # Format extracted protocol
        extracted_dict = {
            "condition": protocol.condition,
            "condition_category": protocol.condition_category,
            "therapeutic_area": protocol.therapeutic_area,
            "phase": protocol.design.phase,
            "target_enrollment": protocol.design.target_enrollment,
            "primary_endpoint": protocol.endpoints.primary_endpoint,
            "intervention_type": protocol.intervention.intervention_type,
            "drug_name": protocol.intervention.drug_name,
        }

        # Format similar trials
        similar_trials = [
            {
                "nct_id": m.nct_id,
                "title": m.title,
                "conditions": m.conditions,
                "interventions": m.interventions,
                "phase": m.phase,
                "status": m.status,
                "enrollment": m.enrollment,
                "overall_score": round(m.overall_score, 1),
            }
            for m in matches[:25]
        ]

        # Run dashboard analysis
        analyzer = DashboardAnalyzer(db)
        dashboard = analyzer.analyze_for_dashboard(protocol, matches, extracted_text)

        return {
            "success": True,
            "filename": file.filename,
            "characters_extracted": len(extracted_text),
            "extracted_protocol": extracted_dict,
            "similar_trials": similar_trials,
            "dashboard": dashboard
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error analyzing PDF: {str(e)}"}
        )


@app.post("/api/export-report")
async def export_report(data: Dict[str, Any]):
    """
    P3 IMPROVEMENT: Export dashboard data to a downloadable report.
    Generates a summary document from the analysis results.
    """
    try:
        from datetime import datetime

        protocol = data.get("extracted_protocol", {})
        dashboard = data.get("dashboard", {})
        similar_trials = data.get("similar_trials", [])

        # Build report content
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CLINANALYTICA PROTOCOL ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Protocol Summary
        report_lines.append("PROTOCOL SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Condition: {protocol.get('condition', 'N/A')}")
        report_lines.append(f"Therapeutic Area: {protocol.get('therapeutic_area', 'N/A')}")
        report_lines.append(f"Phase: {protocol.get('phase', 'N/A')}")
        report_lines.append(f"Target Enrollment: {protocol.get('target_enrollment', 'N/A')}")
        report_lines.append(f"Primary Endpoint: {protocol.get('primary_endpoint', 'N/A')}")
        report_lines.append(f"Intervention: {protocol.get('drug_name', 'N/A')} ({protocol.get('intervention_type', 'N/A')})")
        report_lines.append("")

        # Risk Analysis
        risk = dashboard.get("risk_analysis", {})
        report_lines.append("RISK ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Overall Risk Score: {risk.get('overall_score', 'N/A')}/100")

        predictions = risk.get("predictions", {})
        if predictions:
            term_risk = predictions.get("termination_risk", {})
            report_lines.append(f"Termination Risk: {term_risk.get('probability', 'N/A')}% ({term_risk.get('level', 'N/A')})")
            enroll_delay = predictions.get("enrollment_delay", {})
            report_lines.append(f"Enrollment Delay Risk: {enroll_delay.get('probability', 'N/A')}% ({enroll_delay.get('level', 'N/A')})")
        report_lines.append("")

        # Failure Analysis
        failure = risk.get("failure_analysis", {})
        if failure:
            report_lines.append("FAILURE PATTERN ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Success Rate in Similar Trials: {failure.get('success_rate_similar_trials', 'N/A')}%")
            top_failures = failure.get("top_failure_categories", [])
            for f in top_failures[:5]:
                report_lines.append(f"  - {f.get('category', 'N/A')}: {f.get('percentage', 0)}%")
            report_lines.append("")

        # Competitive Landscape
        competitive = risk.get("enhanced_competitive", {})
        if competitive:
            report_lines.append("COMPETITIVE LANDSCAPE")
            report_lines.append("-" * 40)
            pressure = competitive.get("competitive_pressure", {})
            report_lines.append(f"Competitive Pressure: {pressure.get('level', 'N/A').upper()} (Score: {pressure.get('score', 'N/A')})")
            report_lines.append(f"High-Risk Competitors: {pressure.get('high_risk_competitors', 0)}")
            report_lines.append(f"Total Competing Enrollment: {pressure.get('total_competing_enrollment', 0):,}")

            top_sponsors = competitive.get("top_sponsors", [])[:5]
            if top_sponsors:
                report_lines.append("Top Sponsors:")
                for s in top_sponsors:
                    report_lines.append(f"  - {s.get('sponsor', 'N/A')}: {s.get('trials', 0)} trials")
            report_lines.append("")

        # Site Intelligence
        sites = dashboard.get("site_intelligence", {})
        if sites:
            report_lines.append("SITE STRATEGY")
            report_lines.append("-" * 40)
            strategy = sites.get("strategy", {})
            report_lines.append(f"Recommended Sites: {strategy.get('recommended_sites', 'N/A')}")
            report_lines.append(f"Pts/Site Target: {strategy.get('pts_per_site_target', 'N/A')}")
            report_lines.append(f"Benchmark: {strategy.get('pts_per_site_benchmark', 'N/A')} pts/site")
            report_lines.append(f"Activation Timeline: {strategy.get('activation_timeline', 'N/A')}")
            report_lines.append("")

        # Enrollment Forecast
        enrollment = dashboard.get("enrollment_forecast", {})
        if enrollment:
            report_lines.append("ENROLLMENT FORECAST")
            report_lines.append("-" * 40)
            report_lines.append(f"Target Enrollment: {enrollment.get('target_enrollment', 'N/A')}")
            scenarios = enrollment.get("scenarios", [])
            for s in scenarios:
                report_lines.append(f"  {s.get('name', 'N/A').title()}: {s.get('months', 'N/A')} months ({s.get('probability', 'N/A')}% probability)")
            report_lines.append("")

        # Eligibility Analysis
        eligibility = dashboard.get("eligibility_analysis", {})
        if eligibility:
            report_lines.append("ELIGIBILITY ANALYSIS")
            report_lines.append("-" * 40)
            sf = eligibility.get("screen_failure_prediction", {})
            report_lines.append(f"Predicted Screen Failure Rate: {sf.get('predicted_rate', 'N/A')}%")
            report_lines.append(f"Estimated Patient Pool: {sf.get('estimated_pool', 'N/A'):,}")
            report_lines.append("")

        # Endpoint Intelligence
        endpoints = dashboard.get("endpoint_intelligence", {})
        if endpoints:
            report_lines.append("ENDPOINT INTELLIGENCE")
            report_lines.append("-" * 40)
            report_lines.append(f"Primary Endpoint: {endpoints.get('primary_endpoint', 'N/A')}")
            fda = endpoints.get("fda_alignment", {})
            report_lines.append(f"FDA Alignment: {fda.get('status', 'N/A')}")
            success_rates = endpoints.get("endpoint_success_rates", {})
            if success_rates:
                report_lines.append(f"Overall Success Rate: {success_rates.get('overall_results_rate', 'N/A')}%")
            report_lines.append("")

        # Similar Trials
        report_lines.append("TOP SIMILAR TRIALS")
        report_lines.append("-" * 40)
        for i, trial in enumerate(similar_trials[:10], 1):
            report_lines.append(f"{i}. {trial.get('nct_id', 'N/A')} - Score: {trial.get('overall_score', 0)}")
            report_lines.append(f"   {trial.get('title', 'N/A')[:70]}...")
            report_lines.append(f"   Phase: {trial.get('phase', 'N/A')} | Status: {trial.get('status', 'N/A')} | Enrollment: {trial.get('enrollment', 'N/A')}")
        report_lines.append("")

        # Recommendations
        report_lines.append("KEY RECOMMENDATIONS")
        report_lines.append("-" * 40)

        # From protocol optimization
        opt = dashboard.get("protocol_optimization", {})
        recs = opt.get("recommendations", [])
        for i, rec in enumerate(recs[:5], 1):
            if isinstance(rec, dict):
                report_lines.append(f"{i}. [{rec.get('priority', 'MEDIUM')}] {rec.get('recommendation', 'N/A')}")

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("Report generated by ClinAnalytica - AI-Powered Protocol Intelligence")
        report_lines.append("https://clinicaltrials.gov for trial details")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        return {
            "success": True,
            "report": report_text,
            "format": "text",
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error generating report: {str(e)}"}
        )


class ScenarioInput(BaseModel):
    """Input for interactive scenario simulation."""
    base_enrollment: int = 600
    base_sites: int = 50
    base_duration_months: int = 24
    # Adjustable parameters
    enrollment_change_pct: int = 0  # -50 to +100
    site_change_pct: int = 0  # -50 to +100
    eligibility_relaxation: int = 0  # 0-3 (none, mild, moderate, aggressive)
    competitive_pressure: str = "moderate"  # low, moderate, high
    therapeutic_area: str = "General"


@app.post("/api/simulate-scenario")
async def simulate_scenario(input_data: ScenarioInput):
    """
    P3 IMPROVEMENT: Interactive scenario simulator.
    Allows users to adjust parameters and see projected outcomes.
    """
    try:
        # Calculate adjusted values
        adjusted_enrollment = int(input_data.base_enrollment * (1 + input_data.enrollment_change_pct / 100))
        adjusted_sites = int(input_data.base_sites * (1 + input_data.site_change_pct / 100))

        # Eligibility impact on patient pool and screen failure
        eligibility_impacts = {
            0: {"pool_multiplier": 1.0, "sf_reduction": 0, "label": "Current criteria"},
            1: {"pool_multiplier": 1.15, "sf_reduction": 5, "label": "Mild relaxation"},
            2: {"pool_multiplier": 1.35, "sf_reduction": 10, "label": "Moderate relaxation"},
            3: {"pool_multiplier": 1.60, "sf_reduction": 15, "label": "Aggressive relaxation"}
        }
        elig_impact = eligibility_impacts.get(input_data.eligibility_relaxation, eligibility_impacts[0])

        # Competition impact on enrollment rate
        competition_impacts = {
            "low": {"rate_multiplier": 1.2, "timeline_factor": 0.85},
            "moderate": {"rate_multiplier": 1.0, "timeline_factor": 1.0},
            "high": {"rate_multiplier": 0.75, "timeline_factor": 1.3}
        }
        comp_impact = competition_impacts.get(input_data.competitive_pressure, competition_impacts["moderate"])

        # Base assumptions by therapeutic area
        area_assumptions = {
            "Rheumatology": {"base_rate_per_site": 0.8, "base_sf_rate": 35},
            "Oncology": {"base_rate_per_site": 1.2, "base_sf_rate": 40},
            "Cardiology": {"base_rate_per_site": 1.0, "base_sf_rate": 30},
            "Endocrinology": {"base_rate_per_site": 1.5, "base_sf_rate": 25},
            "Neurology": {"base_rate_per_site": 0.6, "base_sf_rate": 45},
            "Gastroenterology": {"base_rate_per_site": 0.9, "base_sf_rate": 35},
            "General": {"base_rate_per_site": 1.0, "base_sf_rate": 35}
        }
        area = area_assumptions.get(input_data.therapeutic_area, area_assumptions["General"])

        # Calculate projections
        effective_rate = area["base_rate_per_site"] * comp_impact["rate_multiplier"] * elig_impact["pool_multiplier"]
        monthly_enrollment = adjusted_sites * effective_rate
        projected_duration = adjusted_enrollment / monthly_enrollment if monthly_enrollment > 0 else 36

        # Adjust for realistic constraints
        projected_duration = max(6, min(48, projected_duration))  # 6-48 month bounds

        # Screen failure rate
        base_sf = area["base_sf_rate"]
        adjusted_sf = max(15, base_sf - elig_impact["sf_reduction"])

        # Screens needed
        screens_needed = int(adjusted_enrollment * (100 / (100 - adjusted_sf)))

        # Risk assessment
        risk_score = 50  # Base
        risk_factors = []

        if adjusted_enrollment > input_data.base_enrollment * 1.5:
            risk_score += 15
            risk_factors.append("Enrollment target significantly above baseline")

        if adjusted_sites < 30:
            risk_score += 10
            risk_factors.append("Low site count increases individual site dependency")

        if input_data.eligibility_relaxation >= 2:
            risk_score += 10
            risk_factors.append("Aggressive eligibility relaxation may affect population homogeneity")

        if input_data.competitive_pressure == "high":
            risk_score += 15
            risk_factors.append("High competitive pressure impacts site availability")

        if projected_duration > 30:
            risk_score += 10
            risk_factors.append("Extended timeline increases execution risk")

        risk_score = min(100, max(0, risk_score))

        # Timeline scenarios
        optimistic_months = round(projected_duration * 0.75, 1)
        base_months = round(projected_duration, 1)
        conservative_months = round(projected_duration * 1.4, 1)

        # Cost implications (rough estimates)
        cost_per_patient = 25000 if input_data.therapeutic_area in ["Oncology", "Neurology"] else 15000
        cost_per_site = 75000  # Setup + monitoring
        total_cost_estimate = (adjusted_enrollment * cost_per_patient) + (adjusted_sites * cost_per_site)

        return {
            "success": True,
            "scenario": {
                "enrollment_target": adjusted_enrollment,
                "site_count": adjusted_sites,
                "eligibility_setting": elig_impact["label"],
                "competition_level": input_data.competitive_pressure
            },
            "projections": {
                "monthly_enrollment_rate": round(monthly_enrollment, 1),
                "rate_per_site_month": round(effective_rate, 2),
                "screen_failure_rate": adjusted_sf,
                "screens_needed": screens_needed,
                "timeline": {
                    "optimistic_months": optimistic_months,
                    "base_months": base_months,
                    "conservative_months": conservative_months
                }
            },
            "risk_assessment": {
                "overall_score": risk_score,
                "level": "high" if risk_score >= 70 else "medium" if risk_score >= 40 else "low",
                "factors": risk_factors
            },
            "cost_estimate": {
                "total_usd": total_cost_estimate,
                "per_patient": cost_per_patient,
                "site_costs": adjusted_sites * cost_per_site,
                "note": "Rough estimate for planning purposes only"
            },
            "recommendations": _get_scenario_recommendations(
                adjusted_enrollment, adjusted_sites, projected_duration,
                risk_score, input_data.therapeutic_area
            )
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error simulating scenario: {str(e)}"}
        )


def _get_scenario_recommendations(enrollment: int, sites: int, duration: float,
                                   risk: int, area: str) -> List[Dict]:
    """Generate recommendations based on scenario parameters."""
    recommendations = []

    pts_per_site = enrollment / sites if sites > 0 else 0

    if pts_per_site > 20:
        recommendations.append({
            "type": "site_count",
            "recommendation": f"Consider adding {int((enrollment / 15) - sites)} more sites",
            "rationale": f"Current {pts_per_site:.1f} pts/site is high; 12-15 is typical",
            "priority": "high"
        })

    if duration > 24:
        recommendations.append({
            "type": "timeline",
            "recommendation": "Review eligibility criteria for relaxation opportunities",
            "rationale": f"Projected {duration:.1f} months is long; consider expanding criteria",
            "priority": "medium"
        })

    if risk >= 60:
        recommendations.append({
            "type": "risk",
            "recommendation": "Consider phased enrollment or adaptive design",
            "rationale": f"Risk score {risk} suggests significant execution challenges",
            "priority": "high"
        })

    if sites < 40 and enrollment > 400:
        recommendations.append({
            "type": "geography",
            "recommendation": "Expand geographic footprint",
            "rationale": "Limited site count for enrollment target; add regions",
            "priority": "medium"
        })

    if not recommendations:
        recommendations.append({
            "type": "general",
            "recommendation": "Scenario appears well-balanced",
            "rationale": "Key parameters within typical ranges",
            "priority": "low"
        })

    return recommendations


@app.get("/api/stats")
async def get_stats():
    """Get database statistics."""
    stats = get_db_stats()
    return {"success": True, "stats": stats}


# ============== PATIENT TRIAL MATCHING API ==============
@app.post("/api/trial-search")
async def trial_search(input_data: TrialSearchInput):
    """
    Search for recruiting trials and generate screening questions.

    Step 1 of patient matching flow.
    """
    if not input_data.condition or len(input_data.condition.strip()) < 2:
        raise HTTPException(status_code=400, detail="Please enter a condition to search for")

    db = get_database()
    if not db:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        from src.matching import TrialSearcher, QuestionGenerator
        from src.analysis.vector_store import get_vector_store

        # Get vector store for semantic search
        vector_store = get_vector_store()

        # Search for matching trials using semantic search
        searcher = TrialSearcher(db, vector_store=vector_store)
        trials = searcher.search_by_condition(
            condition=input_data.condition,
            max_results=100,
            min_age=input_data.age,
            use_semantic=True  # Use semantic search with embeddings
        )

        if not trials:
            return {
                "success": True,
                "trial_count": 0,
                "message": f"No recruiting trials found for '{input_data.condition}'. Try a different search term.",
                "questions": []
            }

        # Generate screening questions based on trial criteria
        generator = QuestionGenerator()
        question_set = generator.generate_questions(
            condition=input_data.condition,
            trials=trials,
            max_questions=8
        )

        return {
            "success": True,
            "trial_count": len(trials),
            "condition": input_data.condition,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "type": q.type,
                    "options": q.options,
                    "required": q.required,
                    "help_text": q.help_text
                }
                for q in question_set.questions
            ]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/trial-match")
async def trial_match(input_data: TrialMatchInput):
    """
    Match patient answers to trials and return ranked results.

    Step 2 of patient matching flow.
    """
    if not input_data.condition:
        raise HTTPException(status_code=400, detail="Condition is required")

    if not input_data.answers:
        raise HTTPException(status_code=400, detail="Patient answers are required")

    db = get_database()
    if not db:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        from src.matching import TrialSearcher, EligibilityMatcher
        from src.analysis.vector_store import get_vector_store

        # Get patient age and location from answers for filtering
        patient_age = input_data.answers.get('age')
        patient_location = input_data.answers.get('location')

        # Get vector store for semantic search
        vector_store = get_vector_store()

        # Search for trials with filters using semantic search
        searcher = TrialSearcher(db, vector_store=vector_store)
        trials = searcher.search_by_condition(
            condition=input_data.condition,
            max_results=50,
            min_age=int(patient_age) if patient_age else None,
            use_semantic=True  # Use semantic search with embeddings
        )

        if not trials:
            return {
                "success": True,
                "matches": [],
                "message": "No matching trials found with your criteria."
            }

        # Match patient to trials
        matcher = EligibilityMatcher()
        matches = matcher.match_patient_to_trials(
            patient_answers=input_data.answers,
            trials=trials,
            max_trials=15
        )

        # Format results
        results = []
        for match in matches:
            results.append({
                "nct_id": match.nct_id,
                "title": match.title,
                "phase": match.phase,
                "status": match.status,
                "match_score": match.match_score,
                "match_level": match.match_level,
                "summary": match.summary,
                "criteria_met": [
                    {"criterion": c.criterion, "explanation": c.explanation, "patient_value": c.patient_value}
                    for c in match.criteria_met
                ],
                "criteria_not_met": [
                    {"criterion": c.criterion, "explanation": c.explanation, "patient_value": c.patient_value}
                    for c in match.criteria_not_met
                ],
                "criteria_unknown": [
                    {"criterion": c.criterion, "explanation": c.explanation}
                    for c in match.criteria_unknown
                ],
                "nearest_site": match.nearest_site,
                "distance_miles": match.distance_miles
            })

        return {
            "success": True,
            "condition": input_data.condition,
            "total_evaluated": len(trials),
            "matches": results
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/popular-conditions")
async def get_popular_conditions():
    """
    Get popular conditions with recruiting trial counts.
    Used to dynamically populate the "Popular searches" section.
    """
    db = get_database()
    if not db:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        from sqlalchemy import text
        from collections import Counter

        # Get conditions from recruiting trials
        with db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT conditions, COUNT(*) as trial_count
                FROM trials
                WHERE status IN ('RECRUITING', 'NOT_YET_RECRUITING', 'ENROLLING_BY_INVITATION')
                AND conditions IS NOT NULL AND conditions != ''
                GROUP BY conditions
                HAVING COUNT(*) >= 10
                ORDER BY trial_count DESC
                LIMIT 200
            """))
            rows = result.fetchall()

        # Parse and aggregate individual conditions
        condition_counts = Counter()
        for row in rows:
            conditions_str = row[0] or ''
            count = row[1]
            # Split by common delimiters
            for cond in conditions_str.split('|'):
                cond = cond.strip()
                if cond and len(cond) > 3 and len(cond) < 60:
                    # Normalize common variations
                    cond_lower = cond.lower()
                    if 'cancer' in cond_lower or 'carcinoma' in cond_lower or 'tumor' in cond_lower:
                        condition_counts[cond] += count
                    elif 'diabetes' in cond_lower:
                        condition_counts[cond] += count
                    elif 'arthritis' in cond_lower or 'rheumat' in cond_lower:
                        condition_counts[cond] += count
                    elif 'heart' in cond_lower or 'cardio' in cond_lower or 'coronary' in cond_lower:
                        condition_counts[cond] += count
                    elif 'alzheimer' in cond_lower or 'dementia' in cond_lower:
                        condition_counts[cond] += count
                    elif 'depression' in cond_lower or 'anxiety' in cond_lower:
                        condition_counts[cond] += count
                    elif 'asthma' in cond_lower or 'copd' in cond_lower:
                        condition_counts[cond] += count
                    elif 'hiv' in cond_lower or 'hepatitis' in cond_lower:
                        condition_counts[cond] += count
                    else:
                        condition_counts[cond] += count

        # Get top conditions with meaningful counts
        top_conditions = condition_counts.most_common(20)

        return {
            "success": True,
            "conditions": [
                {"name": name, "trial_count": count}
                for name, count in top_conditions
            ]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "conditions": []}


@app.get("/api/condition-autocomplete")
async def condition_autocomplete(q: str = ""):
    """
    Autocomplete suggestions for condition search.
    """
    if not q or len(q) < 2:
        return {"suggestions": []}

    db = get_database()
    if not db:
        return {"suggestions": []}

    try:
        from sqlalchemy import text

        # Search conditions that match the query
        with db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT conditions, COUNT(*) as cnt
                FROM trials
                WHERE status IN ('RECRUITING', 'NOT_YET_RECRUITING')
                AND LOWER(conditions) LIKE LOWER(:pattern)
                GROUP BY conditions
                ORDER BY cnt DESC
                LIMIT 50
            """), {"pattern": f"%{q}%"})
            rows = result.fetchall()

        # Parse and deduplicate
        suggestions = set()
        for row in rows:
            conditions_str = row[0] or ''
            for cond in conditions_str.split('|'):
                cond = cond.strip()
                if cond and q.lower() in cond.lower() and len(cond) < 80:
                    suggestions.add(cond)

        # Sort by relevance (exact match first, then alphabetical)
        sorted_suggestions = sorted(
            suggestions,
            key=lambda x: (0 if x.lower().startswith(q.lower()) else 1, x.lower())
        )

        return {"suggestions": sorted_suggestions[:10]}

    except Exception as e:
        return {"suggestions": []}


@app.get("/api/trial-insights/{condition}")
async def get_trial_insights(condition: str):
    """
    Get insights about trials for a condition - therapeutic area,
    intervention types, phase distribution, etc.
    """
    db = get_database()
    if not db:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        from sqlalchemy import text
        from collections import Counter

        # Get trial data for this condition
        with db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    interventions,
                    phase,
                    therapeutic_area,
                    enrollment,
                    conditions
                FROM trials
                WHERE status IN ('RECRUITING', 'NOT_YET_RECRUITING')
                AND (
                    LOWER(conditions) LIKE LOWER(:pattern)
                    OR LOWER(title) LIKE LOWER(:pattern)
                )
                LIMIT 500
            """), {"pattern": f"%{condition}%"})
            rows = result.fetchall()

        if not rows:
            return {
                "success": True,
                "total_recruiting": 0,
                "insights": {}
            }

        # Analyze interventions
        intervention_types = Counter()
        drug_names = Counter()
        phases = Counter()
        therapeutic_areas = Counter()
        total_enrollment = 0

        for row in rows:
            interventions = row[0] or ''
            phase = row[1] or 'Not specified'
            ta = row[2] or ''
            enrollment = row[3] or 0

            phases[phase] += 1
            total_enrollment += enrollment

            if ta:
                therapeutic_areas[ta] += 1

            # Parse interventions
            for interv in interventions.split('|'):
                interv = interv.strip().lower()
                if not interv:
                    continue

                # Classify intervention type
                if 'drug:' in interv or 'biological:' in interv:
                    intervention_types['Drug/Biological'] += 1
                    # Extract drug name
                    drug = interv.replace('drug:', '').replace('biological:', '').strip()
                    if drug and len(drug) > 2:
                        drug_names[drug.title()] += 1
                elif 'device:' in interv:
                    intervention_types['Device'] += 1
                elif 'procedure:' in interv:
                    intervention_types['Procedure'] += 1
                elif 'behavioral:' in interv:
                    intervention_types['Behavioral'] += 1
                elif 'genetic:' in interv or 'gene therapy' in interv:
                    intervention_types['Gene Therapy'] += 1
                else:
                    intervention_types['Other'] += 1

        # Detect therapeutic area from conditions
        detected_area = "General"
        area_keywords = {
            'Oncology': ['cancer', 'carcinoma', 'tumor', 'lymphoma', 'leukemia', 'melanoma', 'sarcoma'],
            'Cardiology': ['heart', 'cardiac', 'cardiovascular', 'coronary', 'atrial', 'hypertension'],
            'Neurology': ['alzheimer', 'parkinson', 'multiple sclerosis', 'epilepsy', 'stroke', 'dementia'],
            'Rheumatology': ['arthritis', 'rheumatoid', 'lupus', 'psoriatic', 'spondylitis'],
            'Diabetes': ['diabetes', 'diabetic', 'glycemic', 'insulin'],
            'Respiratory': ['asthma', 'copd', 'pulmonary', 'lung disease'],
            'Psychiatry': ['depression', 'anxiety', 'bipolar', 'schizophrenia', 'ptsd'],
            'Infectious Disease': ['hiv', 'hepatitis', 'covid', 'infection', 'viral'],
            'Gastroenterology': ['crohn', 'colitis', 'ibd', 'liver', 'hepatic']
        }

        condition_lower = condition.lower()
        for area, keywords in area_keywords.items():
            if any(kw in condition_lower for kw in keywords):
                detected_area = area
                break

        return {
            "success": True,
            "total_recruiting": len(rows),
            "detected_therapeutic_area": detected_area,
            "insights": {
                "intervention_types": dict(intervention_types.most_common(6)),
                "top_drugs": dict(drug_names.most_common(8)),
                "phase_distribution": dict(phases.most_common()),
                "avg_enrollment": round(total_enrollment / len(rows)) if rows else 0
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.get("/api/related-conditions/{condition}")
async def get_related_conditions(condition: str):
    """
    Get related conditions that might be of interest.
    Useful when few trials are found for the searched condition.
    """
    # Define related condition mappings
    related_mappings = {
        'rheumatoid arthritis': ['Psoriatic Arthritis', 'Ankylosing Spondylitis', 'Lupus', 'Juvenile Arthritis'],
        'breast cancer': ['Triple Negative Breast Cancer', 'HER2+ Breast Cancer', 'Metastatic Breast Cancer', 'Ovarian Cancer'],
        'lung cancer': ['Non-Small Cell Lung Cancer', 'Small Cell Lung Cancer', 'Mesothelioma', 'Lung Adenocarcinoma'],
        'type 2 diabetes': ['Type 1 Diabetes', 'Prediabetes', 'Diabetic Neuropathy', 'Diabetic Kidney Disease'],
        'depression': ['Major Depressive Disorder', 'Bipolar Depression', 'Anxiety Disorder', 'PTSD'],
        'alzheimer': ["Alzheimer's Disease", 'Mild Cognitive Impairment', 'Dementia', 'Frontotemporal Dementia'],
        'asthma': ['Severe Asthma', 'Allergic Asthma', 'COPD', 'Eosinophilic Asthma'],
        'heart failure': ['Chronic Heart Failure', 'HFrEF', 'HFpEF', 'Cardiomyopathy'],
        'crohn': ["Crohn's Disease", 'Ulcerative Colitis', 'Inflammatory Bowel Disease', 'Fistulizing Crohn'],
        'multiple sclerosis': ['Relapsing MS', 'Progressive MS', 'Clinically Isolated Syndrome', 'Neuromyelitis Optica'],
        'psoriasis': ['Plaque Psoriasis', 'Psoriatic Arthritis', 'Atopic Dermatitis', 'Hidradenitis Suppurativa'],
    }

    condition_lower = condition.lower()

    # Find matching related conditions
    related = []
    for key, values in related_mappings.items():
        if key in condition_lower or condition_lower in key:
            related = values
            break

    # If no direct match, try partial matching
    if not related:
        for key, values in related_mappings.items():
            if any(word in condition_lower for word in key.split()):
                related = values
                break

    return {
        "success": True,
        "condition": condition,
        "related_conditions": related[:5]
    }


@app.get("/api/trial/{nct_id}")
async def get_trial_details(nct_id: str):
    """Get detailed information about a specific trial."""

    db = get_database()
    if not db:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        from sqlalchemy import text

        with db.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    nct_id, title, conditions, phase, status, enrollment,
                    interventions, therapeutic_area, eligibility_criteria,
                    min_age, max_age, sex, brief_summary, study_type,
                    start_date, completion_date, sponsor
                FROM trials
                WHERE nct_id = :nct_id
            """), {"nct_id": nct_id})
            row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Trial not found")

        # Get locations
        with db.engine.connect() as conn:
            loc_result = conn.execute(text("""
                SELECT facility_name, city, state, country, zip_code,
                       contact_name, contact_phone, contact_email
                FROM trial_locations
                WHERE nct_id = :nct_id
            """), {"nct_id": nct_id})
            locations = [
                {
                    "facility_name": loc[0],
                    "city": loc[1],
                    "state": loc[2],
                    "country": loc[3],
                    "zip_code": loc[4],
                    "contact_name": loc[5],
                    "contact_phone": loc[6],
                    "contact_email": loc[7]
                }
                for loc in loc_result.fetchall()
            ]

        return {
            "success": True,
            "trial": {
                "nct_id": row[0],
                "title": row[1],
                "condition": row[2],
                "phase": row[3],
                "status": row[4],
                "enrollment": row[5],
                "interventions": row[6],
                "therapeutic_area": row[7],
                "eligibility_criteria": row[8],
                "min_age": row[9],
                "max_age": row[10],
                "sex": row[11],
                "brief_summary": row[12],
                "study_type": row[13],
                "start_date": row[14],
                "completion_date": row[15],
                "sponsor": row[16],
                "locations": locations,
                "clinicaltrials_url": f"https://clinicaltrials.gov/study/{row[0]}"
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "clinanalytica"}


# ============== PATIENT LOCATION FEATURES ==============

# US ZIP code to lat/lon mapping (common zip codes for quick lookup)
# For production, use a proper geocoding service or full ZIP database
US_ZIP_COORDINATES = {
    # Major cities - approximate center coordinates
    "10001": (40.7484, -73.9967),  # New York
    "90001": (33.9425, -118.2551),  # Los Angeles
    "60601": (41.8819, -87.6278),  # Chicago
    "77001": (29.7604, -95.3698),  # Houston
    "85001": (33.4484, -112.0740),  # Phoenix
    "19101": (39.9526, -75.1652),  # Philadelphia
    "78201": (29.4241, -98.4936),  # San Antonio
    "92101": (32.7157, -117.1611),  # San Diego
    "75201": (32.7767, -96.7970),  # Dallas
    "95101": (37.3382, -121.8863),  # San Jose
    "94102": (37.7749, -122.4194),  # San Francisco
    "98101": (47.6062, -122.3321),  # Seattle
    "80201": (39.7392, -104.9903),  # Denver
    "02101": (42.3601, -71.0589),  # Boston
    "20001": (38.9072, -77.0369),  # Washington DC
    "30301": (33.7490, -84.3880),  # Atlanta
    "33101": (25.7617, -80.1918),  # Miami
    "48201": (42.3314, -83.0458),  # Detroit
    "55401": (44.9778, -93.2650),  # Minneapolis
    "28201": (35.2271, -80.8431),  # Charlotte
    "97201": (45.5152, -122.6784),  # Portland
    "84101": (40.7608, -111.8910),  # Salt Lake City
    "89101": (36.1699, -115.1398),  # Las Vegas
    "63101": (38.6270, -90.1994),  # St. Louis
    "21201": (39.2904, -76.6122),  # Baltimore
    "53201": (43.0389, -87.9065),  # Milwaukee
    "37201": (36.1627, -86.7816),  # Nashville
    "73101": (35.4676, -97.5164),  # Oklahoma City
    "40201": (38.2527, -85.7585),  # Louisville
    "46201": (39.7684, -86.1581),  # Indianapolis
    "27601": (35.7796, -78.6382),  # Raleigh
    "32801": (28.5383, -81.3792),  # Orlando
    "44101": (41.4993, -81.6944),  # Cleveland
    "15201": (40.4406, -79.9959),  # Pittsburgh
    "64101": (39.0997, -94.5786),  # Kansas City
    "45201": (39.1031, -84.5120),  # Cincinnati
    "78701": (30.2672, -97.7431),  # Austin
    "43201": (39.9612, -82.9988),  # Columbus
}


def get_coordinates_from_zip(zip_code: str) -> tuple:
    """
    Get lat/lon coordinates from a US ZIP code.
    Uses cached lookup with fallback to region-based estimation.
    """
    if not zip_code:
        return None

    # Clean zip code
    zip_clean = zip_code.strip()[:5]

    # Direct lookup
    if zip_clean in US_ZIP_COORDINATES:
        return US_ZIP_COORDINATES[zip_clean]

    # Try 3-digit prefix matching for regional approximation
    zip_prefix = zip_clean[:3]

    # Regional approximations based on ZIP prefix
    regional_coords = {
        # Northeast
        "010": (42.1, -72.6), "011": (42.2, -72.5), "012": (42.4, -73.2),
        "013": (42.5, -72.5), "014": (42.3, -71.8), "015": (42.2, -71.8),
        "016": (42.4, -71.4), "017": (42.5, -71.2), "018": (42.4, -71.0),
        "019": (42.5, -70.9), "020": (42.3, -71.1), "021": (42.4, -71.0),
        "022": (42.3, -71.0), "023": (42.0, -71.5), "024": (42.4, -71.2),
        "025": (42.0, -70.8), "026": (41.8, -70.8), "027": (41.7, -71.4),
        # New York
        "100": (40.8, -74.0), "101": (40.7, -73.9), "102": (40.7, -74.0),
        "103": (40.6, -74.1), "104": (40.8, -73.9), "105": (41.0, -73.8),
        "106": (41.0, -73.8), "107": (41.0, -73.9), "108": (41.1, -73.9),
        "109": (41.2, -73.8), "110": (40.8, -73.5), "111": (40.7, -73.7),
        "112": (40.6, -73.9), "113": (40.7, -73.9), "114": (40.7, -73.8),
        # Mid-Atlantic
        "190": (40.0, -75.2), "191": (40.0, -75.1), "192": (39.9, -75.2),
        "200": (38.9, -77.0), "201": (38.9, -77.0), "202": (38.9, -77.0),
        "210": (39.3, -76.6), "211": (39.2, -76.7), "212": (39.3, -76.6),
        # Southeast
        "300": (33.8, -84.4), "301": (33.4, -84.8), "302": (33.4, -84.5),
        "303": (33.7, -84.4), "304": (33.8, -84.5), "305": (34.0, -84.1),
        "320": (30.3, -81.7), "321": (29.2, -81.0), "322": (29.7, -82.4),
        "323": (27.9, -82.5), "324": (30.0, -84.3), "325": (28.8, -81.4),
        "326": (29.2, -82.1), "327": (28.5, -81.4), "328": (28.0, -82.5),
        "329": (27.5, -82.6), "330": (25.8, -80.2), "331": (25.8, -80.2),
        # Texas
        "750": (32.8, -97.0), "751": (32.8, -96.8), "752": (32.8, -96.8),
        "753": (32.8, -97.3), "760": (32.7, -97.3), "761": (32.4, -99.7),
        "762": (32.4, -99.7), "763": (33.9, -98.5), "770": (29.8, -95.4),
        "771": (29.8, -95.4), "772": (29.8, -95.4), "773": (29.5, -95.1),
        "774": (29.3, -94.8), "775": (29.9, -95.3), "776": (30.1, -94.2),
        "777": (30.0, -94.1), "778": (28.8, -96.9), "779": (27.8, -97.4),
        "780": (29.4, -98.5), "781": (29.4, -98.5), "782": (29.4, -98.5),
        "783": (27.5, -99.5), "784": (26.2, -98.2), "785": (26.2, -98.2),
        "786": (30.3, -97.7), "787": (30.3, -97.7), "788": (31.8, -106.4),
        "789": (31.8, -106.4), "790": (35.2, -101.8), "791": (35.2, -101.8),
        "792": (34.4, -103.2), "793": (33.6, -101.8), "794": (31.5, -102.0),
        "795": (33.5, -101.9), "796": (31.1, -97.3), "797": (32.5, -94.7),
        # California
        "900": (34.0, -118.3), "901": (34.0, -118.3), "902": (33.9, -118.2),
        "903": (33.9, -118.2), "904": (33.8, -118.3), "905": (33.9, -118.4),
        "906": (34.0, -118.2), "907": (34.1, -118.3), "908": (34.0, -118.3),
        "910": (34.2, -118.5), "911": (34.1, -118.3), "912": (34.2, -118.2),
        "913": (34.2, -118.3), "914": (34.2, -118.4), "915": (34.4, -118.5),
        "916": (34.2, -118.4), "917": (34.1, -117.3), "918": (34.0, -117.5),
        "919": (34.4, -117.3), "920": (32.7, -117.2), "921": (32.8, -117.2),
        "922": (33.1, -117.3), "923": (33.1, -117.1), "924": (33.4, -117.2),
        "925": (33.7, -116.3), "926": (33.7, -117.9), "927": (33.7, -118.0),
        "928": (33.1, -115.5), "930": (34.4, -119.7), "931": (34.4, -119.7),
        "932": (36.7, -119.8), "933": (35.4, -119.0), "934": (36.3, -119.3),
        "935": (35.2, -118.9), "936": (36.7, -119.8), "937": (36.3, -119.3),
        "940": (37.8, -122.4), "941": (37.8, -122.4), "942": (37.8, -122.3),
        "943": (37.6, -122.4), "944": (37.8, -122.4), "945": (37.9, -122.3),
        "946": (37.9, -122.5), "947": (37.9, -122.3), "948": (37.8, -122.3),
        "949": (37.9, -122.1), "950": (37.3, -121.9), "951": (37.4, -122.1),
        "952": (37.4, -122.2), "953": (36.6, -121.9), "954": (37.7, -122.2),
        "955": (38.8, -123.0), "956": (38.6, -121.5), "957": (38.4, -121.4),
        "958": (38.6, -121.5), "959": (40.6, -122.4), "960": (39.5, -121.6),
        "961": (40.8, -124.2),
        # Mountain/West
        "800": (39.7, -105.0), "801": (39.7, -105.0), "802": (39.7, -105.0),
        "803": (39.8, -104.9), "804": (40.0, -105.3), "805": (40.6, -105.1),
        "806": (40.4, -104.7), "809": (37.3, -108.6), "810": (38.8, -104.8),
        "840": (40.8, -111.9), "841": (40.8, -111.9), "843": (41.2, -111.9),
        "844": (41.7, -111.8), "845": (40.2, -111.7), "846": (40.2, -111.7),
        "847": (40.5, -111.4), "850": (33.4, -112.1), "852": (33.4, -112.1),
        "853": (33.4, -111.9), "854": (33.3, -111.9), "855": (33.4, -111.9),
        "856": (33.3, -111.8), "857": (32.2, -110.9), "859": (31.5, -110.3),
        "860": (34.5, -114.4), "863": (35.2, -111.6), "864": (35.2, -111.7),
        "865": (34.8, -114.6), "870": (35.1, -106.6), "871": (35.1, -106.6),
        "872": (35.1, -106.6), "873": (35.8, -106.3), "874": (36.2, -105.6),
        "875": (34.5, -105.7), "877": (32.9, -105.9), "878": (32.3, -106.8),
        "879": (32.3, -104.2), "880": (31.8, -106.4), "881": (32.4, -106.8),
        "882": (33.4, -104.5), "883": (34.2, -103.2), "884": (34.5, -103.2),
        # Pacific Northwest
        "970": (45.5, -122.7), "971": (45.5, -122.7), "972": (45.5, -122.7),
        "973": (45.5, -122.8), "974": (44.9, -123.0), "975": (44.1, -123.1),
        "976": (42.3, -122.9), "977": (42.2, -121.8), "978": (44.3, -121.2),
        "979": (44.6, -121.2), "980": (47.6, -122.3), "981": (47.6, -122.3),
        "982": (47.2, -122.5), "983": (47.6, -117.4), "984": (47.7, -117.4),
        "985": (47.6, -117.4), "986": (46.6, -120.5), "988": (46.3, -119.3),
        "989": (47.9, -122.2), "990": (47.7, -117.4), "991": (47.7, -117.4),
        "992": (47.7, -117.4), "993": (46.1, -118.3), "994": (46.6, -120.5),
    }

    if zip_prefix in regional_coords:
        return regional_coords[zip_prefix]

    return None


def calculate_distance_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula."""
    import math

    R = 3959  # Earth's radius in miles

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


@app.get("/api/geocode")
async def geocode_location(zip_code: str = "", city: str = "", state: str = ""):
    """
    Geocode a location (zip code or city/state) to coordinates.
    Used for patient location-based trial searching.
    """
    coords = None

    # Try ZIP code first
    if zip_code:
        coords = get_coordinates_from_zip(zip_code)
        if coords:
            return {
                "success": True,
                "latitude": coords[0],
                "longitude": coords[1],
                "location_type": "zip_code",
                "formatted": f"ZIP {zip_code}"
            }

    # Fallback - could integrate with external geocoding API
    # For now, return a helpful message
    return {
        "success": False,
        "message": "Could not geocode location. Please enter a US ZIP code.",
        "supported_format": "5-digit US ZIP code"
    }


@app.get("/api/trial-sites/{nct_id}")
async def get_trial_sites(
    nct_id: str,
    patient_lat: float = None,
    patient_lon: float = None,
    max_distance: int = None,
    country: str = None
):
    """
    Get all trial sites for a specific trial with optional distance filtering.
    Returns sites sorted by distance if patient location is provided.
    """
    db = get_database()
    if not db:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        from sqlalchemy import text

        # Try with lat/lon columns first, fallback if they don't exist
        queries = [
            """
            SELECT
                facility_name, city, state, country, zip_code,
                contact_name, contact_phone, contact_email,
                latitude, longitude
            FROM trial_locations
            WHERE nct_id = :nct_id
            """,
            """
            SELECT
                facility_name, city, state, country, zip_code,
                contact_name, contact_phone, contact_email,
                NULL as latitude, NULL as longitude
            FROM trial_locations
            WHERE nct_id = :nct_id
            """
        ]

        params = {"nct_id": nct_id}
        rows = []

        for query in queries:
            full_query = query
            if country:
                full_query += " AND LOWER(country) = LOWER(:country)"
                params["country"] = country

            try:
                with db.engine.connect() as conn:
                    result = conn.execute(text(full_query), params)
                    rows = result.fetchall()
                break  # Success, exit loop
            except Exception as e:
                if "latitude" in str(e) or "longitude" in str(e):
                    continue  # Try fallback query
                raise

        sites = []
        for row in rows:
            site = {
                "facility_name": row[0] or "Study Site",
                "city": row[1],
                "state": row[2],
                "country": row[3],
                "zip_code": row[4],
                "contact_name": row[5],
                "contact_phone": row[6],
                "contact_email": row[7],
                "latitude": row[8],
                "longitude": row[9],
                "distance_miles": None,
                "directions_url": None
            }

            # Calculate distance if patient location provided
            if patient_lat and patient_lon and site["latitude"] and site["longitude"]:
                try:
                    site["distance_miles"] = round(
                        calculate_distance_miles(
                            patient_lat, patient_lon,
                            float(site["latitude"]), float(site["longitude"])
                        ), 1
                    )
                except:
                    pass

            # Generate Google Maps directions URL
            address_parts = [p for p in [site["facility_name"], site["city"], site["state"], site["country"]] if p]
            if address_parts:
                import urllib.parse
                address = ", ".join(address_parts)
                site["directions_url"] = f"https://www.google.com/maps/dir/?api=1&destination={urllib.parse.quote(address)}"

            sites.append(site)

        # Filter by distance if specified
        if max_distance and patient_lat and patient_lon:
            sites = [s for s in sites if s["distance_miles"] is not None and s["distance_miles"] <= max_distance]

        # Sort by distance if patient location provided
        if patient_lat and patient_lon:
            sites.sort(key=lambda x: x["distance_miles"] if x["distance_miles"] is not None else float('inf'))

        # Group sites by state/region for better display
        sites_by_region = {}
        for site in sites:
            region = site["state"] or site["country"] or "Other"
            if region not in sites_by_region:
                sites_by_region[region] = []
            sites_by_region[region].append(site)

        return {
            "success": True,
            "nct_id": nct_id,
            "total_sites": len(sites),
            "patient_location": {
                "latitude": patient_lat,
                "longitude": patient_lon
            } if patient_lat and patient_lon else None,
            "sites": sites,
            "sites_by_region": sites_by_region,
            "nearest_site": sites[0] if sites else None
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


class LocationSearchInput(BaseModel):
    """Input for location-aware trial search."""
    condition: str
    zip_code: str = ""
    max_distance_miles: int = 100
    age: int = None
    phase: str = None


@app.post("/api/trial-search-nearby")
async def trial_search_nearby(input_data: LocationSearchInput):
    """
    Search for recruiting trials near a patient's location.
    Returns trials with sites within the specified distance.
    """
    if not input_data.condition or len(input_data.condition.strip()) < 2:
        raise HTTPException(status_code=400, detail="Please enter a condition to search for")

    db = get_database()
    if not db:
        raise HTTPException(status_code=500, detail="Database connection failed")

    # Get patient coordinates from ZIP code
    patient_coords = None
    if input_data.zip_code:
        patient_coords = get_coordinates_from_zip(input_data.zip_code)

    try:
        from src.matching import TrialSearcher, QuestionGenerator
        from src.analysis.vector_store import get_vector_store
        from sqlalchemy import text

        # Get vector store for semantic search
        vector_store = get_vector_store()

        # Search for matching trials
        searcher = TrialSearcher(db, vector_store=vector_store)
        trials = searcher.search_by_condition(
            condition=input_data.condition,
            max_results=200,  # Get more to filter by location
            min_age=input_data.age,
            use_semantic=True
        )

        if not trials:
            return {
                "success": True,
                "trial_count": 0,
                "message": f"No recruiting trials found for '{input_data.condition}'.",
                "questions": []
            }

        # If we have patient location, filter and sort by proximity
        trials_with_distance = []
        if patient_coords:
            patient_lat, patient_lon = patient_coords

            for trial in trials:
                # Get locations for this trial - handle missing lat/lon columns
                locations = []
                try:
                    with db.engine.connect() as conn:
                        result = conn.execute(text("""
                            SELECT latitude, longitude, facility_name, city, state, country,
                                   contact_phone, contact_email
                            FROM trial_locations
                            WHERE nct_id = :nct_id AND latitude IS NOT NULL AND longitude IS NOT NULL
                        """), {"nct_id": trial.nct_id})
                        locations = result.fetchall()
                except Exception as e:
                    # If lat/lon columns don't exist, get basic location info
                    if "latitude" in str(e) or "longitude" in str(e):
                        with db.engine.connect() as conn:
                            result = conn.execute(text("""
                                SELECT NULL, NULL, facility_name, city, state, country,
                                       contact_phone, contact_email
                                FROM trial_locations
                                WHERE nct_id = :nct_id
                            """), {"nct_id": trial.nct_id})
                            locations = result.fetchall()
                    else:
                        raise

                if locations:
                    # Find nearest site
                    min_distance = float('inf')
                    nearest_site = None

                    for loc in locations:
                        try:
                            dist = calculate_distance_miles(
                                patient_lat, patient_lon,
                                float(loc[0]), float(loc[1])
                            )
                            if dist < min_distance:
                                min_distance = dist
                                nearest_site = {
                                    "facility_name": loc[2],
                                    "city": loc[3],
                                    "state": loc[4],
                                    "country": loc[5],
                                    "contact_phone": loc[6],
                                    "contact_email": loc[7],
                                    "distance_miles": round(dist, 1)
                                }
                        except:
                            continue

                    # Only include if within max distance
                    if min_distance <= input_data.max_distance_miles:
                        trial.locations = [nearest_site] if nearest_site else []
                        trial.nearest_distance = min_distance
                        trials_with_distance.append(trial)
                else:
                    # No location data, include anyway but mark as unknown distance
                    trial.locations = []
                    trial.nearest_distance = float('inf')
                    trials_with_distance.append(trial)

            # Sort by distance
            trials_with_distance.sort(key=lambda t: t.nearest_distance)
            trials = trials_with_distance[:100]  # Limit results
        else:
            # No location filter, just use first 100
            trials = trials[:100]

        # Generate screening questions
        generator = QuestionGenerator()
        question_set = generator.generate_questions(
            condition=input_data.condition,
            trials=trials,
            max_questions=8
        )

        # Count trials by distance bands if we have location
        distance_bands = None
        if patient_coords:
            distance_bands = {
                "within_25_miles": len([t for t in trials if hasattr(t, 'nearest_distance') and t.nearest_distance <= 25]),
                "within_50_miles": len([t for t in trials if hasattr(t, 'nearest_distance') and t.nearest_distance <= 50]),
                "within_100_miles": len([t for t in trials if hasattr(t, 'nearest_distance') and t.nearest_distance <= 100]),
            }

        return {
            "success": True,
            "trial_count": len(trials),
            "condition": input_data.condition,
            "patient_location": {
                "zip_code": input_data.zip_code,
                "latitude": patient_coords[0] if patient_coords else None,
                "longitude": patient_coords[1] if patient_coords else None,
                "max_distance": input_data.max_distance_miles
            },
            "distance_bands": distance_bands,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "type": q.type,
                    "options": q.options,
                    "required": q.required,
                    "help_text": q.help_text
                }
                for q in question_set.questions
            ]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


class LocationMatchInput(BaseModel):
    """Input for location-aware trial matching."""
    condition: str
    answers: Dict[str, Any]
    zip_code: str = ""
    max_distance_miles: int = 100


@app.post("/api/trial-match-nearby")
async def trial_match_nearby(input_data: LocationMatchInput):
    """
    Match patient to trials with location-aware filtering and sorting.
    Returns matched trials with all nearby sites.
    """
    if not input_data.condition:
        raise HTTPException(status_code=400, detail="Condition is required")

    if not input_data.answers:
        raise HTTPException(status_code=400, detail="Patient answers are required")

    db = get_database()
    if not db:
        raise HTTPException(status_code=500, detail="Database connection failed")

    # Get patient coordinates
    patient_coords = None
    if input_data.zip_code:
        patient_coords = get_coordinates_from_zip(input_data.zip_code)

    try:
        from src.matching import TrialSearcher, EligibilityMatcher
        from src.analysis.vector_store import get_vector_store
        from sqlalchemy import text

        patient_age = input_data.answers.get('age')

        # Get vector store
        vector_store = get_vector_store()

        # Search for trials
        searcher = TrialSearcher(db, vector_store=vector_store)
        trials = searcher.search_by_condition(
            condition=input_data.condition,
            max_results=100,
            min_age=int(patient_age) if patient_age else None,
            use_semantic=True
        )

        if not trials:
            return {
                "success": True,
                "matches": [],
                "message": "No matching trials found."
            }

        # Get locations for each trial
        patient_lat = patient_coords[0] if patient_coords else None
        patient_lon = patient_coords[1] if patient_coords else None

        for trial in trials:
            # Get locations - handle missing lat/lon columns
            rows = []
            try:
                with db.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT facility_name, city, state, country, zip_code,
                               contact_name, contact_phone, contact_email,
                               latitude, longitude
                        FROM trial_locations
                        WHERE nct_id = :nct_id
                    """), {"nct_id": trial.nct_id})
                    rows = result.fetchall()
            except Exception as e:
                if "latitude" in str(e) or "longitude" in str(e):
                    with db.engine.connect() as conn:
                        result = conn.execute(text("""
                            SELECT facility_name, city, state, country, zip_code,
                                   contact_name, contact_phone, contact_email,
                                   NULL as latitude, NULL as longitude
                            FROM trial_locations
                            WHERE nct_id = :nct_id
                        """), {"nct_id": trial.nct_id})
                        rows = result.fetchall()
                else:
                    raise

            trial.locations = []
            for row in rows:
                site = {
                    "facility_name": row[0] or "Study Site",
                    "city": row[1],
                    "state": row[2],
                    "country": row[3],
                    "zip_code": row[4],
                    "contact_name": row[5],
                    "contact_phone": row[6],
                    "contact_email": row[7],
                    "latitude": row[8],
                    "longitude": row[9],
                    "distance_miles": None
                }

                # Calculate distance
                if patient_lat and patient_lon and site["latitude"] and site["longitude"]:
                    try:
                        site["distance_miles"] = round(
                            calculate_distance_miles(
                                patient_lat, patient_lon,
                                float(site["latitude"]), float(site["longitude"])
                            ), 1
                        )
                    except:
                        pass

                trial.locations.append(site)

            # Sort locations by distance
            if patient_coords:
                trial.locations.sort(key=lambda x: x["distance_miles"] if x["distance_miles"] is not None else float('inf'))

        # Match patient to trials
        matcher = EligibilityMatcher()
        matches = matcher.match_patient_to_trials(
            patient_answers=input_data.answers,
            trials=trials,
            patient_location=patient_coords,
            max_trials=20
        )

        # Format results with enhanced location data
        results = []
        for match in matches:
            # Find the trial to get all locations
            trial = next((t for t in trials if t.nct_id == match.nct_id), None)
            all_sites = trial.locations if trial else []

            # Filter sites by distance if specified
            nearby_sites = all_sites
            if patient_coords and input_data.max_distance_miles:
                nearby_sites = [
                    s for s in all_sites
                    if s["distance_miles"] is None or s["distance_miles"] <= input_data.max_distance_miles
                ]

            # Sort sites: US first, then by distance
            def site_sort_key(site):
                is_us = site.get("country", "").lower() in ["united states", "usa", "us"]
                distance = site.get("distance_miles") if site.get("distance_miles") is not None else float('inf')
                return (0 if is_us else 1, distance)

            nearby_sites = sorted(nearby_sites, key=site_sort_key)

            # Add directions URLs
            import urllib.parse
            for site in nearby_sites:
                address_parts = [p for p in [site["facility_name"], site["city"], site["state"], site["country"]] if p]
                if address_parts:
                    address = ", ".join(address_parts)
                    site["directions_url"] = f"https://www.google.com/maps/dir/?api=1&destination={urllib.parse.quote(address)}"

            results.append({
                "nct_id": match.nct_id,
                "title": match.title,
                "phase": match.phase,
                "status": match.status,
                "match_score": match.match_score,
                "match_level": match.match_level,
                "summary": match.summary,
                "criteria_met": [
                    {"criterion": c.criterion, "explanation": c.explanation, "patient_value": c.patient_value}
                    for c in match.criteria_met
                ],
                "criteria_not_met": [
                    {"criterion": c.criterion, "explanation": c.explanation, "patient_value": c.patient_value}
                    for c in match.criteria_not_met
                ],
                "criteria_unknown": [
                    {"criterion": c.criterion, "explanation": c.explanation}
                    for c in match.criteria_unknown
                ],
                "nearest_site": nearby_sites[0] if nearby_sites else None,
                "distance_miles": nearby_sites[0]["distance_miles"] if nearby_sites and nearby_sites[0].get("distance_miles") else None,
                "all_sites": nearby_sites,
                "total_sites": len(all_sites),
                "nearby_sites_count": len(nearby_sites)
            })

        # Sort results by distance if we have location
        if patient_coords:
            results.sort(key=lambda x: (
                -(x["match_score"]),  # Primary: match score descending
                x["distance_miles"] if x["distance_miles"] else float('inf')  # Secondary: distance ascending
            ))

        return {
            "success": True,
            "condition": input_data.condition,
            "total_evaluated": len(trials),
            "patient_location": {
                "zip_code": input_data.zip_code,
                "latitude": patient_lat,
                "longitude": patient_lon,
                "max_distance": input_data.max_distance_miles
            } if patient_coords else None,
            "matches": results
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


# ============== RUN ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
