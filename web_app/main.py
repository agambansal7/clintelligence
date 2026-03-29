"""
Clintelligence - Modern FastAPI Backend
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
import httpx
from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent.parent / '.env')

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
static_dir = BASE_DIR / "static"
if static_dir.exists() and any(static_dir.iterdir()):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
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
    try:
        db = get_database()
        if db:
            return db.get_stats()
    except Exception as e:
        print(f"Database stats error: {e}")
    return {"total_trials": 566622, "status": "cached"}


# ============== OPENAI EMBEDDINGS FOR SEMANTIC MATCHING ==============
def get_openai_embedding(text: str) -> List[float]:
    """Generate embedding using OpenAI text-embedding-3-small."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    text = text[:8000]  # Truncate to avoid token limits

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def get_openai_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    texts = [t[:8000] for t in texts]  # Truncate each

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def create_protocol_embedding_text(protocol_info: Dict) -> str:
    """Create text representation of protocol for embedding."""
    parts = []
    if protocol_info.get("condition"):
        parts.append(f"Condition: {protocol_info['condition']}")
    if protocol_info.get("therapeutic_area"):
        parts.append(f"Therapeutic Area: {protocol_info['therapeutic_area']}")
    if protocol_info.get("intervention_name"):
        parts.append(f"Intervention: {protocol_info['intervention_name']}")
    if protocol_info.get("intervention_type"):
        parts.append(f"Intervention Type: {protocol_info['intervention_type']}")
    if protocol_info.get("primary_endpoint"):
        parts.append(f"Primary Endpoint: {protocol_info['primary_endpoint']}")
    if protocol_info.get("phase"):
        parts.append(f"Phase: {protocol_info['phase']}")
    if protocol_info.get("inclusion_criteria"):
        criteria = protocol_info['inclusion_criteria']
        if isinstance(criteria, list):
            criteria = "; ".join(criteria[:5])
        parts.append(f"Inclusion: {criteria[:500]}")
    if protocol_info.get("exclusion_criteria"):
        criteria = protocol_info['exclusion_criteria']
        if isinstance(criteria, list):
            criteria = "; ".join(criteria[:5])
        parts.append(f"Exclusion: {criteria[:500]}")
    return " | ".join(parts)


def create_trial_embedding_text(trial: Dict) -> str:
    """Create text representation of trial for embedding."""
    parts = []
    if trial.get("title"):
        parts.append(f"Title: {trial['title']}")
    if trial.get("conditions"):
        parts.append(f"Conditions: {trial['conditions']}")
    if trial.get("interventions"):
        parts.append(f"Interventions: {trial['interventions']}")
    if trial.get("primary_outcomes"):
        parts.append(f"Outcomes: {trial['primary_outcomes'][:300]}")
    if trial.get("phase"):
        parts.append(f"Phase: {trial['phase']}")
    if trial.get("eligibility_criteria"):
        parts.append(f"Eligibility: {trial['eligibility_criteria'][:300]}")
    return " | ".join(parts)


async def semantic_match_trials(protocol_info: Dict, trials: List[Dict], top_n: int = 30) -> List[Dict]:
    """
    Use OpenAI embeddings to semantically match and rank trials against the protocol.
    """
    if not trials:
        return []

    try:
        # Create protocol embedding text
        protocol_text = create_protocol_embedding_text(protocol_info)

        # Create trial embedding texts
        trial_texts = [create_trial_embedding_text(t) for t in trials]

        # Get embeddings
        all_texts = [protocol_text] + trial_texts
        embeddings = get_openai_embeddings_batch(all_texts)

        protocol_embedding = embeddings[0]
        trial_embeddings = embeddings[1:]

        # Calculate similarity scores
        for i, trial in enumerate(trials):
            similarity = cosine_similarity(protocol_embedding, trial_embeddings[i])
            trial["semantic_score"] = round(similarity * 100, 1)

        # Sort by semantic score
        trials.sort(key=lambda x: x.get("semantic_score", 0), reverse=True)

        return trials[:top_n]

    except Exception as e:
        print(f"Semantic matching error: {e}")
        # Return original trials if embedding fails
        return trials[:top_n]


# ============== CLINICALTRIALS.GOV API ==============
def simplify_condition_query(condition: str) -> str:
    """Simplify condition string for better API search results."""
    # Remove common prefixes and parenthetical content
    import re

    simplified = condition

    # Remove parenthetical abbreviations like (NSCLC), (TNBC), etc.
    simplified = re.sub(r'\s*\([^)]*\)\s*', ' ', simplified)

    # Remove common prefixes
    prefixes_to_remove = [
        'advanced', 'metastatic', 'recurrent', 'refractory',
        'relapsed', 'newly diagnosed', 'previously treated',
        'treatment-naive', 'stage iv', 'stage iii', 'stage ii'
    ]
    for prefix in prefixes_to_remove:
        simplified = re.sub(rf'\b{prefix}\b', '', simplified, flags=re.IGNORECASE)

    # Clean up whitespace
    simplified = ' '.join(simplified.split())

    # If result is too short, use original but simplified
    if len(simplified) < 5:
        simplified = condition.split('(')[0].strip()

    return simplified.strip()


async def search_clinicaltrials_api(condition: str, intervention: str = None, phase: str = None, max_results: int = 50) -> List[Dict]:
    """Search ClinicalTrials.gov API directly."""
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    # Simplify condition for better search results
    search_condition = simplify_condition_query(condition) if condition else ""
    print(f"Original condition: {condition}")
    print(f"Simplified search: {search_condition}")

    params = {
        "query.cond": search_condition,
        "pageSize": min(max_results, 100),
        "format": "json",
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,EnrollmentCount,Condition,InterventionName,EligibilityCriteria,PrimaryOutcome,StartDate,CompletionDate,LeadSponsorName,StudyType"
    }

    if phase:
        # Simplify phase format
        phase_map = {
            "Phase 1": "PHASE1", "Phase 2": "PHASE2", "Phase 3": "PHASE3", "Phase 4": "PHASE4",
            "Phase I": "PHASE1", "Phase II": "PHASE2", "Phase III": "PHASE3", "Phase IV": "PHASE4"
        }
        params["filter.phase"] = phase_map.get(phase, phase)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            trials = []
            for study in data.get("studies", []):
                proto = study.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                status = proto.get("statusModule", {})
                design = proto.get("designModule", {})
                sponsor = proto.get("sponsorCollaboratorsModule", {})
                eligibility = proto.get("eligibilityModule", {})
                conditions = proto.get("conditionsModule", {})
                interventions = proto.get("armsInterventionsModule", {})
                outcomes = proto.get("outcomesModule", {})

                # Extract intervention names
                intervention_list = interventions.get("interventions", [])
                intervention_names = [i.get("name", "") for i in intervention_list]

                # Extract primary outcomes
                primary_outcomes = outcomes.get("primaryOutcomes", [])
                primary_outcome_text = "; ".join([o.get("measure", "") for o in primary_outcomes[:3]])

                trials.append({
                    "nct_id": ident.get("nctId", ""),
                    "title": ident.get("briefTitle", ""),
                    "status": status.get("overallStatus", ""),
                    "phase": design.get("phases", [""])[0] if design.get("phases") else "",
                    "enrollment": design.get("enrollmentInfo", {}).get("count", 0),
                    "conditions": ", ".join(conditions.get("conditions", [])),
                    "interventions": ", ".join(intervention_names),
                    "eligibility_criteria": eligibility.get("eligibilityCriteria", ""),
                    "primary_outcomes": primary_outcome_text,
                    "sponsor": sponsor.get("leadSponsor", {}).get("name", ""),
                    "start_date": status.get("startDateStruct", {}).get("date", ""),
                    "completion_date": status.get("completionDateStruct", {}).get("date", ""),
                })

            return trials
    except Exception as e:
        print(f"ClinicalTrials.gov API error: {e}")
        return []


async def extract_protocol_with_claude(protocol_text: str) -> Dict:
    """Extract structured protocol information using Claude."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = f"""Analyze this clinical trial protocol and extract key information in JSON format.

Protocol Text:
{protocol_text[:15000]}

Return a JSON object with these fields:
{{
    "condition": "primary disease/condition being studied",
    "therapeutic_area": "oncology, cardiology, neurology, etc.",
    "phase": "Phase 1, Phase 2, Phase 3, or Phase 4",
    "intervention_type": "drug, device, biological, etc.",
    "intervention_name": "name of the treatment",
    "target_enrollment": number or null,
    "primary_endpoint": "main outcome measure",
    "secondary_endpoints": ["list of secondary endpoints"],
    "inclusion_criteria": ["key inclusion criteria"],
    "exclusion_criteria": ["key exclusion criteria"],
    "study_duration_months": number or null,
    "comparator": "placebo, active comparator, or none",
    "sponsor": "sponsor name if mentioned"
}}

Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse JSON from response
    text = response.content[0].text
    # Try to extract JSON if wrapped in markdown
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


async def analyze_similar_trials_with_claude(protocol_info: Dict, similar_trials: List[Dict]) -> Dict:
    """Use Claude to analyze similar trials and provide comprehensive dashboard insights."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    trials_summary = json.dumps(similar_trials[:20], indent=2)
    target_enrollment = protocol_info.get('target_enrollment') or 100

    prompt = f"""You are an expert clinical trial intelligence analyst. Analyze the following protocol against similar historical trials and provide comprehensive risk assessment and recommendations.

NEW PROTOCOL DETAILS:
- Condition: {protocol_info.get('condition', 'Unknown')}
- Therapeutic Area: {protocol_info.get('therapeutic_area', 'Unknown')}
- Phase: {protocol_info.get('phase', 'Unknown')}
- Intervention Type: {protocol_info.get('intervention_type', 'Unknown')}
- Intervention Name: {protocol_info.get('intervention_name', 'Unknown')}
- Primary Endpoint: {protocol_info.get('primary_endpoint', 'Unknown')}
- Secondary Endpoints: {protocol_info.get('secondary_endpoints', [])}
- Target Enrollment: {target_enrollment}
- Study Duration: {protocol_info.get('study_duration_months', 'Unknown')} months
- Comparator: {protocol_info.get('comparator', 'Unknown')}
- Key Inclusion: {protocol_info.get('inclusion_criteria', [])}
- Key Exclusion: {protocol_info.get('exclusion_criteria', [])}

SIMILAR HISTORICAL TRIALS ({len(similar_trials)} found):
{trials_summary}

Based on your expert analysis, provide a comprehensive assessment in JSON format:

{{
    "risk_assessment": {{
        "overall_risk_score": <number 0-100, where 100 is highest risk>,
        "overall_risk_level": "LOW/MEDIUM/HIGH",
        "overall_risk_rationale": "detailed explanation of overall risk",
        "enrollment_risk_score": <number 0-100>,
        "enrollment_risk_rationale": "explanation of enrollment challenges",
        "timeline_risk_score": <number 0-100>,
        "timeline_risk_rationale": "explanation of timeline risks",
        "endpoint_risk_score": <number 0-100>,
        "endpoint_risk_rationale": "explanation of endpoint selection risks",
        "competition_risk_score": <number 0-100>,
        "competition_risk_rationale": "explanation of competitive landscape"
    }},
    "amendment_prediction": {{
        "amendment_probability": <number 0-100, percentage chance of amendments>,
        "predicted_amendments": <number, expected number of amendments>,
        "common_amendment_reasons": ["list of likely amendment reasons based on similar trials"]
    }},
    "protocol_complexity": {{
        "complexity_score": <number 0-100>,
        "complexity_factors": ["list of factors contributing to complexity"],
        "simplification_opportunities": ["potential ways to simplify"]
    }},
    "success_prediction": {{
        "predicted_success_rate": <number 0-100, percentage>,
        "success_factors": ["positive factors"],
        "risk_factors": ["negative factors"],
        "comparison_to_similar": "how this compares to similar trials"
    }},
    "enrollment_forecast": {{
        "estimated_duration_months": <number>,
        "duration_range_low": <number>,
        "duration_range_high": <number>,
        "enrollment_rate_per_site_month": <number>,
        "recommended_sites": <number>,
        "recommended_countries": <number>,
        "patients_per_site": <number>
    }},
    "competitive_landscape": {{
        "competition_level": "LOW/MEDIUM/HIGH",
        "active_competitors": <number of recruiting similar trials>,
        "key_competitors": ["list of major competing sponsors/trials"],
        "differentiation_opportunities": ["how to differentiate from competition"]
    }},
    "recommendations": {{
        "protocol_strengths": ["list of 3-5 protocol strengths"],
        "protocol_weaknesses": ["list of 3-5 areas needing improvement"],
        "critical_considerations": ["list of 3-5 critical items to address"],
        "site_selection_guidance": "guidance on site selection strategy",
        "enrollment_strategy": "recommended enrollment approach"
    }},
    "similar_trial_insights": {{
        "total_analyzed": <number>,
        "completed_trials": <number>,
        "ongoing_trials": <number>,
        "terminated_early": <number>,
        "average_enrollment": <number>,
        "average_duration_months": <number>,
        "key_learnings": ["insights from similar trials"]
    }}
}}

Be specific and data-driven in your analysis. Use the similar trials data to inform your predictions.
Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


# ============== PAGES ==============
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - Protocol Entry."""
    stats = get_db_stats()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"page": "home", "stats": stats}
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Analysis Dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"page": "dashboard"}
    )


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    """About page."""
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={"page": "about"}
    )


@app.get("/find-trials", response_class=HTMLResponse)
async def find_trials(request: Request):
    """Patient trial finder page."""
    recruiting_count = 95000  # Default fallback
    try:
        db = get_database()
        if db:
            from sqlalchemy import text
            with db.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM trials WHERE status IN ('RECRUITING', 'NOT_YET_RECRUITING')"
                ))
                recruiting_count = result.scalar() or 95000
    except Exception as e:
        print(f"Find trials DB error: {e}")

    return templates.TemplateResponse(
        request=request,
        name="find_trials.html",
        context={"page": "find_trials", "recruiting_count": recruiting_count}
    )


# ============== API ENDPOINTS ==============
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_protocol(input_data: ProtocolInput):
    """Analyze a protocol using OpenAI embeddings for semantic matching and Claude for analysis."""

    if not input_data.protocol_text or len(input_data.protocol_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Protocol text must be at least 100 characters")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured for semantic matching")

    try:
        # Step 1: Extract protocol information using Claude
        print("Step 1: Extracting protocol with Claude...")
        extracted_dict = await extract_protocol_with_claude(input_data.protocol_text)
        print(f"Extracted: {extracted_dict.get('condition')}, {extracted_dict.get('intervention_name')}")

        # Step 2: Search ClinicalTrials.gov for candidate trials
        condition = extracted_dict.get("condition", "")
        intervention = extracted_dict.get("intervention_name", "")
        phase = extracted_dict.get("phase", "")

        print(f"Step 2: Searching ClinicalTrials.gov for {condition}...")
        candidate_trials = await search_clinicaltrials_api(
            condition=condition,
            intervention=intervention,
            phase=phase,
            max_results=100  # Get more candidates for semantic filtering
        )
        print(f"Found {len(candidate_trials)} candidate trials")

        # Step 3: Use OpenAI embeddings to semantically match and rank trials
        print("Step 3: Semantic matching with OpenAI embeddings...")
        similar_trials = await semantic_match_trials(
            protocol_info=extracted_dict,
            trials=candidate_trials,
            top_n=input_data.max_similar
        )
        print(f"Top match score: {similar_trials[0].get('semantic_score') if similar_trials else 'N/A'}")

        # Step 4: Use Claude for comprehensive analysis
        print("Step 4: Analyzing with Claude...")
        analysis = await analyze_similar_trials_with_claude(extracted_dict, similar_trials)

        return AnalysisResponse(
            success=True,
            extracted_protocol=extracted_dict,
            risk_assessment=analysis.get("risk_assessment", {}),
            similar_trials=similar_trials[:20],
            metrics=analysis.get("benchmarks", {}),
            site_recommendations=analysis.get("recommendations", {}).get("critical_considerations", [])
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
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
    Analyze protocol using OpenAI embeddings for semantic matching and Claude for comprehensive analysis.
    """
    if not input_data.protocol_text or len(input_data.protocol_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="Protocol text must be at least 100 characters")

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured for semantic matching")

    try:
        # Step 1: Extract protocol information using Claude
        print("Step 1: Extracting protocol structure with Claude...")
        extracted_dict = await extract_protocol_with_claude(input_data.protocol_text)
        print(f"Extracted condition: {extracted_dict.get('condition')}")
        print(f"Extracted intervention: {extracted_dict.get('intervention_name')}")
        print(f"Extracted phase: {extracted_dict.get('phase')}")

        # Step 2: Search ClinicalTrials.gov for candidate trials (get more for filtering)
        condition = extracted_dict.get("condition", "")
        intervention = extracted_dict.get("intervention_name", "")
        phase = extracted_dict.get("phase", "")

        print(f"Step 2: Searching ClinicalTrials.gov for '{condition}'...")
        candidate_trials = await search_clinicaltrials_api(
            condition=condition,
            intervention=intervention,
            phase=phase,
            max_results=100  # Get more candidates for semantic filtering
        )
        print(f"Found {len(candidate_trials)} candidate trials from API")

        # Step 3: Use OpenAI embeddings to semantically match and rank trials
        print("Step 3: Semantic matching with OpenAI embeddings...")
        if input_data.use_reranking and candidate_trials:
            similar_trials = await semantic_match_trials(
                protocol_info=extracted_dict,
                trials=candidate_trials,
                top_n=input_data.max_results
            )
            # Filter by minimum score
            similar_trials = [t for t in similar_trials if t.get("semantic_score", 0) >= input_data.min_score]
            print(f"After semantic matching: {len(similar_trials)} trials (min score: {input_data.min_score})")
            if similar_trials:
                print(f"Top semantic score: {similar_trials[0].get('semantic_score')}")
        else:
            similar_trials = candidate_trials[:input_data.max_results]

        # Step 4: Use Claude for comprehensive analysis
        print("Step 4: Generating comprehensive analysis with Claude...")
        analysis = await analyze_similar_trials_with_claude(extracted_dict, similar_trials)
        print("Analysis complete")

        # Extract analysis components
        risk = analysis.get("risk_assessment", {})
        amendment = analysis.get("amendment_prediction", {})
        complexity = analysis.get("protocol_complexity", {})
        success = analysis.get("success_prediction", {})
        enrollment = analysis.get("enrollment_forecast", {})
        competition = analysis.get("competitive_landscape", {})
        recommendations_data = analysis.get("recommendations", {})
        insights = analysis.get("similar_trial_insights", {})

        # Count trial statuses
        completed_count = len([t for t in similar_trials if t.get("status") == "COMPLETED"])
        recruiting_count = len([t for t in similar_trials if t.get("status") in ["RECRUITING", "ACTIVE_NOT_RECRUITING"]])
        terminated_count = len([t for t in similar_trials if t.get("status") in ["TERMINATED", "WITHDRAWN"]])

        # Build recommendations in the format frontend expects (array with priority)
        formatted_recommendations = []
        for item in recommendations_data.get("critical_considerations", []):
            formatted_recommendations.append({"priority": "high", "recommendation": item, "rationale": "Critical consideration based on similar trials"})
        for item in recommendations_data.get("protocol_weaknesses", []):
            formatted_recommendations.append({"priority": "medium", "recommendation": item, "rationale": "Area for improvement"})

        # Build strengths in the format frontend expects
        formatted_strengths = [{"strength": s, "impact": "positive"} for s in recommendations_data.get("protocol_strengths", [])]

        # Build comprehensive dashboard data matching frontend structure
        dashboard_data = {
            # Risk Analysis - frontend expects overall_score (not overall_risk_score)
            "risk_analysis": {
                "overall_score": risk.get("overall_risk_score", 50),
                "overall_risk_level": risk.get("overall_risk_level", "MEDIUM"),
                "overall_risk_rationale": risk.get("overall_risk_rationale", ""),
                "competitive_landscape": {
                    "competing_count": recruiting_count,
                    "risk_level": competition.get("competition_level", "medium").lower()
                },
                "enrollment_risk": {"score": risk.get("enrollment_risk_score", 50)},
                "timeline_risk": {"score": risk.get("timeline_risk_score", 50)},
                "endpoint_risk": {"score": risk.get("endpoint_risk_score", 50)}
            },

            # Amendment Intelligence - frontend expects this structure
            "amendment_intelligence": {
                "overall_risk_score": amendment.get("amendment_probability", 50),
                "predicted_amendments": amendment.get("predicted_amendments", 2.0),
                "common_reasons": amendment.get("common_amendment_reasons", [])
            },

            # Protocol Optimization - frontend expects complexity_score.score, success_rate, recommendations array
            "protocol_optimization": {
                "complexity_score": {
                    "score": complexity.get("complexity_score", 50),
                    "comparison": "out of 100"
                },
                "success_rate": success.get("predicted_success_rate", 65),
                "trials_analyzed": len(similar_trials),
                "recommendations": formatted_recommendations,
                "strengths": formatted_strengths,
                "amendment_risk": {
                    "probability": amendment.get("amendment_probability", 50),
                    "drivers": [{"factor": r, "impact": "medium"} for r in amendment.get("common_amendment_reasons", [])[:5]]
                },
                "terminated_trials": [{"nct_id": t.get("nct_id"), "title": t.get("title"), "reason": "See ClinicalTrials.gov"}
                                     for t in similar_trials if t.get("status") in ["TERMINATED", "WITHDRAWN"]][:5],
                "design_comparison": []
            },

            # Enrollment Forecast - frontend expects target_enrollment, scenarios array, historical_benchmark
            "enrollment_forecast": {
                "target_enrollment": extracted_dict.get("target_enrollment") or 120,
                "scenarios": [
                    {"name": "base", "months": enrollment.get("estimated_duration_months", 24)},
                    {"name": "optimistic", "months": enrollment.get("duration_range_low", 18)},
                    {"name": "conservative", "months": enrollment.get("duration_range_high", 30)}
                ],
                "historical_benchmark": {
                    "range": f"{enrollment.get('duration_range_low', 18)}-{enrollment.get('duration_range_high', 28)}"
                }
            },

            # Site Intelligence - frontend expects strategy with recommended_sites, etc.
            "site_intelligence": {
                "strategy": {
                    "recommended_sites": enrollment.get("recommended_sites", 50),
                    "recommended_countries": enrollment.get("recommended_countries", 12),
                    "pts_per_site_target": enrollment.get("patients_per_site", 15)
                },
                "top_sites": []
            },

            # Competitive Landscape - frontend expects top_sponsors array
            "competitive_landscape": {
                "total_similar_trials": len(similar_trials),
                "completed": completed_count,
                "recruiting": recruiting_count,
                "terminated": terminated_count,
                "competition_level": competition.get("competition_level", "MEDIUM"),
                "top_sponsors": [{"name": s, "trial_count": 1} for s in competition.get("key_competitors", [])[:6]]
            },

            # Similar trials enhanced
            "similar_trials_enhanced": {
                "detected_therapeutic_area": extracted_dict.get("therapeutic_area", "General"),
                "total_matched": len(similar_trials),
                "showing": len(similar_trials),
                "trials": similar_trials
            },

            # Eligibility analysis
            "eligibility_analysis": {
                "screen_failure_prediction": {
                    "assessment": "moderate",
                    "rate": 25
                }
            }
        }

        return {
            "success": True,
            "extracted_protocol": extracted_dict,
            "similar_trials": similar_trials,
            "summary": {
                "total_matches": len(similar_trials),
                "condition": condition,
                "phase": phase
            },
            "matching_version": "v2-api-based",
            "dashboard": dashboard_data,
            "analysis": analysis  # Include raw analysis for debugging
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
        report_lines.append("CLINTELLIGENCE PROTOCOL ANALYSIS REPORT")
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
        report_lines.append("Report generated by Clintelligence - AI-Powered Protocol Intelligence")
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
    Search for recruiting trials using ClinicalTrials.gov API.
    """
    if not input_data.condition or len(input_data.condition.strip()) < 2:
        raise HTTPException(status_code=400, detail="Please enter a condition to search for")

    try:
        # Search ClinicalTrials.gov API for recruiting trials
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.cond": input_data.condition,
            "filter.overallStatus": "RECRUITING",
            "pageSize": 50,
            "format": "json",
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,EnrollmentCount,Condition,InterventionName,EligibilityCriteria,LeadSponsorName,LocationCity,LocationState"
        }

        if input_data.phase:
            params["query.phase"] = input_data.phase

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

        trials = []
        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            eligibility = proto.get("eligibilityModule", {})
            conditions = proto.get("conditionsModule", {})
            locations = proto.get("contactsLocationsModule", {}).get("locations", [])

            location_str = ""
            if locations:
                loc = locations[0]
                location_str = f"{loc.get('city', '')}, {loc.get('state', '')}"

            trials.append({
                "nct_id": ident.get("nctId", ""),
                "title": ident.get("briefTitle", ""),
                "status": status.get("overallStatus", ""),
                "phase": design.get("phases", [""])[0] if design.get("phases") else "",
                "conditions": ", ".join(conditions.get("conditions", [])),
                "eligibility": eligibility.get("eligibilityCriteria", "")[:500],
                "location": location_str
            })

        if not trials:
            return {
                "success": True,
                "trial_count": 0,
                "message": f"No recruiting trials found for '{input_data.condition}'. Try a different search term.",
                "questions": []
            }

        # Generate basic screening questions
        questions = [
            {"id": "age", "question": "What is your age?", "type": "number", "options": None, "required": True, "help_text": "Enter your current age in years"},
            {"id": "diagnosis", "question": f"Have you been diagnosed with {input_data.condition}?", "type": "boolean", "options": ["Yes", "No"], "required": True, "help_text": None},
            {"id": "treatment", "question": "Are you currently receiving any treatment?", "type": "boolean", "options": ["Yes", "No"], "required": True, "help_text": None},
            {"id": "location", "question": "What is your location (city/state)?", "type": "text", "options": None, "required": False, "help_text": "To find trials near you"}
        ]

        return {
            "success": True,
            "trial_count": len(trials),
            "condition": input_data.condition,
            "trials": trials,
            "questions": questions
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
    Match patient answers to trials using ClinicalTrials.gov API.
    """
    if not input_data.condition:
        raise HTTPException(status_code=400, detail="Condition is required")

    if not input_data.answers:
        raise HTTPException(status_code=400, detail="Patient answers are required")

    try:
        # Search ClinicalTrials.gov API
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.cond": input_data.condition,
            "filter.overallStatus": "RECRUITING",
            "pageSize": 30,
            "format": "json",
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,EnrollmentCount,Condition,InterventionName,EligibilityCriteria,LeadSponsorName,LocationCity,LocationState,LocationFacility"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

        trials = []
        patient_age = input_data.answers.get('age')

        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            eligibility = proto.get("eligibilityModule", {})
            conditions = proto.get("conditionsModule", {})
            locations = proto.get("contactsLocationsModule", {}).get("locations", [])

            # Basic age check
            min_age = eligibility.get("minimumAge", "")
            max_age = eligibility.get("maximumAge", "")

            location_str = ""
            facility = ""
            if locations:
                loc = locations[0]
                location_str = f"{loc.get('city', '')}, {loc.get('state', '')}"
                facility = loc.get('facility', '')

            # Simple scoring based on available info
            score = 70  # Base score
            if patient_age:
                try:
                    age = int(patient_age)
                    if min_age and "Year" in min_age:
                        min_val = int(min_age.split()[0])
                        if age >= min_val:
                            score += 10
                        else:
                            score -= 20
                    if max_age and "Year" in max_age:
                        max_val = int(max_age.split()[0])
                        if age <= max_val:
                            score += 10
                        else:
                            score -= 20
                except:
                    pass

            trials.append({
                "nct_id": ident.get("nctId", ""),
                "title": ident.get("briefTitle", ""),
                "status": status.get("overallStatus", ""),
                "phase": design.get("phases", [""])[0] if design.get("phases") else "",
                "match_score": min(100, max(0, score)),
                "match_level": "Good Match" if score >= 70 else "Potential Match",
                "summary": f"Recruiting trial for {', '.join(conditions.get('conditions', [])[:2])}",
                "criteria_met": [],
                "criteria_not_met": [],
                "criteria_unknown": [],
                "nearest_site": facility,
                "location": location_str,
                "eligibility_summary": eligibility.get("eligibilityCriteria", "")[:300]
            })

        # Sort by score
        trials.sort(key=lambda x: x["match_score"], reverse=True)

        return {
            "success": True,
            "condition": input_data.condition,
            "total_evaluated": len(trials),
            "matches": trials[:15]
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
    Returns curated list of common conditions.
    """
    # Return curated list of popular conditions
    popular = [
        {"name": "Breast Cancer", "trial_count": 3500},
        {"name": "Lung Cancer", "trial_count": 2800},
        {"name": "Type 2 Diabetes", "trial_count": 2200},
        {"name": "Alzheimer's Disease", "trial_count": 1800},
        {"name": "Rheumatoid Arthritis", "trial_count": 1500},
        {"name": "Heart Failure", "trial_count": 1400},
        {"name": "Depression", "trial_count": 1300},
        {"name": "Multiple Sclerosis", "trial_count": 1100},
        {"name": "Parkinson's Disease", "trial_count": 950},
        {"name": "Chronic Pain", "trial_count": 900},
        {"name": "COPD", "trial_count": 850},
        {"name": "Colorectal Cancer", "trial_count": 800},
        {"name": "Prostate Cancer", "trial_count": 780},
        {"name": "Leukemia", "trial_count": 750},
        {"name": "Asthma", "trial_count": 700},
    ]
    return {"success": True, "conditions": popular}


@app.get("/api/condition-autocomplete")
async def condition_autocomplete(q: str = ""):
    """
    Autocomplete suggestions for condition search using curated list.
    """
    if not q or len(q) < 2:
        return {"suggestions": []}

    # Curated list of common conditions
    all_conditions = [
        "Breast Cancer", "Lung Cancer", "Prostate Cancer", "Colorectal Cancer", "Pancreatic Cancer",
        "Ovarian Cancer", "Melanoma", "Leukemia", "Lymphoma", "Brain Cancer", "Liver Cancer",
        "Type 1 Diabetes", "Type 2 Diabetes", "Gestational Diabetes",
        "Alzheimer's Disease", "Parkinson's Disease", "Multiple Sclerosis", "ALS", "Epilepsy",
        "Rheumatoid Arthritis", "Osteoarthritis", "Psoriatic Arthritis", "Lupus", "Fibromyalgia",
        "Heart Failure", "Coronary Artery Disease", "Atrial Fibrillation", "Hypertension",
        "Depression", "Anxiety", "Bipolar Disorder", "Schizophrenia", "PTSD", "OCD",
        "Asthma", "COPD", "Pulmonary Fibrosis", "Cystic Fibrosis",
        "Crohn's Disease", "Ulcerative Colitis", "IBS", "Celiac Disease",
        "HIV/AIDS", "Hepatitis B", "Hepatitis C", "COVID-19",
        "Chronic Pain", "Migraine", "Chronic Fatigue Syndrome",
        "Obesity", "Anemia", "Osteoporosis"
    ]

    q_lower = q.lower()
    matches = [c for c in all_conditions if q_lower in c.lower()]

    # Sort by relevance
    sorted_matches = sorted(
        matches,
        key=lambda x: (0 if x.lower().startswith(q_lower) else 1, x.lower())
    )

    return {"suggestions": sorted_matches[:10]}


@app.get("/api/trial-insights/{condition}")
async def get_trial_insights(condition: str):
    """
    Get insights about trials for a condition using ClinicalTrials.gov API.
    """
    try:
        from collections import Counter

        # Search ClinicalTrials.gov API
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.cond": condition,
            "filter.overallStatus": "RECRUITING|NOT_YET_RECRUITING",
            "pageSize": 100,
            "format": "json",
            "fields": "Phase,EnrollmentCount,InterventionType,InterventionName"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

        studies = data.get("studies", [])
        if not studies:
            return {"success": True, "total_recruiting": 0, "insights": {}}

        # Analyze data
        intervention_types = Counter()
        phases = Counter()
        total_enrollment = 0

        for study in studies:
            proto = study.get("protocolSection", {})
            design = proto.get("designModule", {})
            arms = proto.get("armsInterventionsModule", {})

            phase = design.get("phases", ["Not specified"])[0] if design.get("phases") else "Not specified"
            phases[phase] += 1

            enrollment = design.get("enrollmentInfo", {}).get("count", 0)
            total_enrollment += enrollment or 0

            for interv in arms.get("interventions", []):
                itype = interv.get("type", "Other")
                intervention_types[itype] += 1

        # Detect therapeutic area
        detected_area = "General"
        area_keywords = {
            'Oncology': ['cancer', 'carcinoma', 'tumor', 'lymphoma', 'leukemia', 'melanoma'],
            'Cardiology': ['heart', 'cardiac', 'cardiovascular', 'coronary', 'hypertension'],
            'Neurology': ['alzheimer', 'parkinson', 'multiple sclerosis', 'epilepsy', 'dementia'],
            'Rheumatology': ['arthritis', 'rheumatoid', 'lupus', 'psoriatic'],
            'Diabetes': ['diabetes', 'diabetic', 'glycemic', 'insulin'],
            'Respiratory': ['asthma', 'copd', 'pulmonary'],
            'Psychiatry': ['depression', 'anxiety', 'bipolar', 'schizophrenia'],
            'Infectious Disease': ['hiv', 'hepatitis', 'covid', 'infection']
        }

        condition_lower = condition.lower()
        for area, keywords in area_keywords.items():
            if any(kw in condition_lower for kw in keywords):
                detected_area = area
                break

        return {
            "success": True,
            "total_recruiting": len(studies),
            "detected_therapeutic_area": detected_area,
            "insights": {
                "intervention_types": dict(intervention_types.most_common(6)),
                "phase_distribution": dict(phases.most_common()),
                "avg_enrollment": round(total_enrollment / len(studies)) if studies else 0
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
    """Get detailed information about a specific trial using ClinicalTrials.gov API."""

    try:
        # Fetch from ClinicalTrials.gov API
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
        params = {"format": "json"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Trial not found")
            response.raise_for_status()
            data = response.json()

        proto = data.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        sponsor = proto.get("sponsorCollaboratorsModule", {})
        eligibility = proto.get("eligibilityModule", {})
        conditions = proto.get("conditionsModule", {})
        arms = proto.get("armsInterventionsModule", {})
        desc = proto.get("descriptionModule", {})
        contacts = proto.get("contactsLocationsModule", {})

        # Extract interventions
        interventions = [i.get("name", "") for i in arms.get("interventions", [])]

        # Extract locations
        locations = []
        for loc in contacts.get("locations", []):
            locations.append({
                "facility_name": loc.get("facility", ""),
                "city": loc.get("city", ""),
                "state": loc.get("state", ""),
                "country": loc.get("country", ""),
                "zip_code": loc.get("zip", ""),
                "contact_name": "",
                "contact_phone": "",
                "contact_email": ""
            })

        return {
            "success": True,
            "trial": {
                "nct_id": ident.get("nctId", nct_id),
                "title": ident.get("briefTitle", ""),
                "condition": ", ".join(conditions.get("conditions", [])),
                "phase": design.get("phases", [""])[0] if design.get("phases") else "",
                "status": status.get("overallStatus", ""),
                "enrollment": design.get("enrollmentInfo", {}).get("count", 0),
                "interventions": ", ".join(interventions),
                "therapeutic_area": "",
                "eligibility_criteria": eligibility.get("eligibilityCriteria", ""),
                "min_age": eligibility.get("minimumAge", ""),
                "max_age": eligibility.get("maximumAge", ""),
                "sex": eligibility.get("sex", ""),
                "brief_summary": desc.get("briefSummary", ""),
                "study_type": design.get("studyType", ""),
                "start_date": status.get("startDateStruct", {}).get("date", ""),
                "completion_date": status.get("completionDateStruct", {}).get("date", ""),
                "sponsor": sponsor.get("leadSponsor", {}).get("name", ""),
                "locations": locations,
                "clinicaltrials_url": f"https://clinicaltrials.gov/study/{nct_id}"
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
    return {"status": "healthy", "service": "clintelligence"}


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
    Get all trial sites for a specific trial using ClinicalTrials.gov API.
    """
    try:
        import urllib.parse

        # Fetch from ClinicalTrials.gov API
        url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
        params = {"format": "json"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Trial not found")
            response.raise_for_status()
            data = response.json()

        proto = data.get("protocolSection", {})
        contacts = proto.get("contactsLocationsModule", {})

        sites = []
        for loc in contacts.get("locations", []):
            loc_country = loc.get("country", "")
            if country and loc_country.lower() != country.lower():
                continue

            site = {
                "facility_name": loc.get("facility", "Study Site"),
                "city": loc.get("city", ""),
                "state": loc.get("state", ""),
                "country": loc_country,
                "zip_code": loc.get("zip", ""),
                "contact_name": "",
                "contact_phone": "",
                "contact_email": "",
                "latitude": loc.get("geoPoint", {}).get("lat"),
                "longitude": loc.get("geoPoint", {}).get("lon"),
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
                address = ", ".join(address_parts)
                site["directions_url"] = f"https://www.google.com/maps/dir/?api=1&destination={urllib.parse.quote(address)}"

            sites.append(site)

        # Filter by distance if specified
        if max_distance and patient_lat and patient_lon:
            sites = [s for s in sites if s["distance_miles"] is not None and s["distance_miles"] <= max_distance]

        # Sort by distance if patient location provided
        if patient_lat and patient_lon:
            sites.sort(key=lambda x: x["distance_miles"] if x["distance_miles"] is not None else float('inf'))

        # Group sites by state/region
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
            "patient_location": {"latitude": patient_lat, "longitude": patient_lon} if patient_lat and patient_lon else None,
            "sites": sites,
            "sites_by_region": sites_by_region,
            "nearest_site": sites[0] if sites else None
        }

    except HTTPException:
        raise
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
    Search for recruiting trials near a patient's location using ClinicalTrials.gov API.
    """
    if not input_data.condition or len(input_data.condition.strip()) < 2:
        raise HTTPException(status_code=400, detail="Please enter a condition to search for")

    # Get patient coordinates from ZIP code
    patient_coords = None
    if input_data.zip_code:
        patient_coords = get_coordinates_from_zip(input_data.zip_code)

    try:
        # Search ClinicalTrials.gov API
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.cond": input_data.condition,
            "filter.overallStatus": "RECRUITING",
            "pageSize": 50,
            "format": "json",
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,Condition,LocationCity,LocationState,LocationFacility,LocationGeoPoint"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

        trials = []
        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            conditions = proto.get("conditionsModule", {})
            contacts = proto.get("contactsLocationsModule", {})

            locations = contacts.get("locations", [])
            nearest_distance = float('inf')
            nearest_site = None

            for loc in locations:
                geo = loc.get("geoPoint", {})
                if patient_coords and geo.get("lat") and geo.get("lon"):
                    try:
                        dist = calculate_distance_miles(
                            patient_coords[0], patient_coords[1],
                            float(geo["lat"]), float(geo["lon"])
                        )
                        if dist < nearest_distance:
                            nearest_distance = dist
                            nearest_site = {
                                "facility_name": loc.get("facility", ""),
                                "city": loc.get("city", ""),
                                "state": loc.get("state", ""),
                                "distance_miles": round(dist, 1)
                            }
                    except:
                        pass

            # Filter by max distance
            if patient_coords and nearest_distance > input_data.max_distance_miles:
                continue

            trials.append({
                "nct_id": ident.get("nctId", ""),
                "title": ident.get("briefTitle", ""),
                "status": status.get("overallStatus", ""),
                "phase": design.get("phases", [""])[0] if design.get("phases") else "",
                "conditions": ", ".join(conditions.get("conditions", [])),
                "nearest_site": nearest_site,
                "nearest_distance": nearest_distance if nearest_distance != float('inf') else None
            })

        # Sort by distance
        trials.sort(key=lambda t: t.get("nearest_distance") or float('inf'))

        # Generate basic questions
        questions = [
            {"id": "age", "question": "What is your age?", "type": "number", "options": None, "required": True, "help_text": "Enter your current age"},
            {"id": "diagnosis", "question": f"Have you been diagnosed with {input_data.condition}?", "type": "boolean", "options": ["Yes", "No"], "required": True, "help_text": None},
        ]

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
            "trials": trials,
            "questions": questions
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
    Match patient to trials with location-aware filtering using ClinicalTrials.gov API.
    """
    if not input_data.condition:
        raise HTTPException(status_code=400, detail="Condition is required")

    if not input_data.answers:
        raise HTTPException(status_code=400, detail="Patient answers are required")

    # Get patient coordinates
    patient_coords = None
    if input_data.zip_code:
        patient_coords = get_coordinates_from_zip(input_data.zip_code)

    try:
        import urllib.parse
        patient_age = input_data.answers.get('age')
        patient_lat = patient_coords[0] if patient_coords else None
        patient_lon = patient_coords[1] if patient_coords else None

        # Search ClinicalTrials.gov API
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.cond": input_data.condition,
            "filter.overallStatus": "RECRUITING",
            "pageSize": 50,
            "format": "json",
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,Condition,EligibilityCriteria,LocationCity,LocationState,LocationFacility,LocationGeoPoint,MinimumAge,MaximumAge"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

        results = []
        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            eligibility = proto.get("eligibilityModule", {})
            conditions = proto.get("conditionsModule", {})
            contacts = proto.get("contactsLocationsModule", {})

            # Get all sites
            all_sites = []
            for loc in contacts.get("locations", []):
                geo = loc.get("geoPoint", {})
                site = {
                    "facility_name": loc.get("facility", "Study Site"),
                    "city": loc.get("city", ""),
                    "state": loc.get("state", ""),
                    "country": loc.get("country", ""),
                    "distance_miles": None
                }

                if patient_lat and patient_lon and geo.get("lat") and geo.get("lon"):
                    try:
                        site["distance_miles"] = round(
                            calculate_distance_miles(patient_lat, patient_lon, float(geo["lat"]), float(geo["lon"])), 1
                        )
                    except:
                        pass

                # Add directions URL
                address_parts = [p for p in [site["facility_name"], site["city"], site["state"], site["country"]] if p]
                if address_parts:
                    site["directions_url"] = f"https://www.google.com/maps/dir/?api=1&destination={urllib.parse.quote(', '.join(address_parts))}"

                all_sites.append(site)

            # Filter by distance
            nearby_sites = all_sites
            if patient_coords and input_data.max_distance_miles:
                nearby_sites = [s for s in all_sites if s["distance_miles"] is None or s["distance_miles"] <= input_data.max_distance_miles]

            # Sort by distance
            nearby_sites.sort(key=lambda x: x["distance_miles"] if x["distance_miles"] is not None else float('inf'))

            # Simple scoring
            score = 70
            if patient_age:
                try:
                    age = int(patient_age)
                    min_age = eligibility.get("minimumAge", "")
                    max_age = eligibility.get("maximumAge", "")
                    if min_age and "Year" in min_age and age >= int(min_age.split()[0]):
                        score += 10
                    if max_age and "Year" in max_age and age <= int(max_age.split()[0]):
                        score += 10
                except:
                    pass

            results.append({
                "nct_id": ident.get("nctId", ""),
                "title": ident.get("briefTitle", ""),
                "phase": design.get("phases", [""])[0] if design.get("phases") else "",
                "status": status.get("overallStatus", ""),
                "match_score": min(100, max(0, score)),
                "match_level": "Good Match" if score >= 70 else "Potential Match",
                "summary": f"Trial for {', '.join(conditions.get('conditions', [])[:2])}",
                "criteria_met": [],
                "criteria_not_met": [],
                "criteria_unknown": [],
                "nearest_site": nearby_sites[0] if nearby_sites else None,
                "distance_miles": nearby_sites[0]["distance_miles"] if nearby_sites else None,
                "all_sites": nearby_sites[:5],
                "total_sites": len(all_sites),
                "nearby_sites_count": len(nearby_sites)
            })

        # Sort by score and distance
        results.sort(key=lambda x: (-(x["match_score"]), x["distance_miles"] if x["distance_miles"] else float('inf')))

        return {
            "success": True,
            "condition": input_data.condition,
            "total_evaluated": len(results),
            "patient_location": {
                "zip_code": input_data.zip_code,
                "latitude": patient_lat,
                "longitude": patient_lon,
                "max_distance": input_data.max_distance_miles
            } if patient_coords else None,
            "matches": results[:20]
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
