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


async def search_trials_from_database(condition: str, intervention: str = None, phase: str = None, max_results: int = 100) -> List[Dict]:
    """
    Search trials from PostgreSQL database.
    This is the primary search method when database is available.
    """
    try:
        db = get_database()
        if not db:
            print("Database not available, falling back to API")
            return []

        from sqlalchemy import text

        # Build search query with ILIKE for case-insensitive matching
        query_parts = ["SELECT * FROM trials WHERE 1=1"]
        params = {}

        if condition:
            # Search in conditions and title fields
            query_parts.append("AND (conditions ILIKE :condition OR title ILIKE :condition)")
            params["condition"] = f"%{condition}%"

        if intervention:
            query_parts.append("AND interventions ILIKE :intervention")
            params["intervention"] = f"%{intervention}%"

        if phase:
            # Normalize phase format
            phase_normalized = phase.replace("Phase ", "PHASE").replace(" ", "")
            query_parts.append("AND phase ILIKE :phase")
            params["phase"] = f"%{phase_normalized}%"

        # Prioritize recruiting studies
        query_parts.append("ORDER BY CASE WHEN status = 'RECRUITING' THEN 0 ELSE 1 END, nct_id DESC")
        query_parts.append(f"LIMIT {max_results}")

        query = " ".join(query_parts)
        print(f"Database query: {query[:200]}...")
        print(f"Params: {params}")

        with db.session() as session:
            result = session.execute(text(query), params)
            rows = result.fetchall()

            trials = []
            for row in rows:
                trials.append({
                    "nct_id": row.nct_id,
                    "title": row.title,
                    "status": row.status,
                    "phase": row.phase,
                    "enrollment": row.enrollment,
                    "conditions": row.conditions,
                    "interventions": row.interventions,
                    "eligibility_criteria": row.eligibility_criteria,
                    "primary_outcomes": row.primary_outcomes,
                    "sponsor": row.sponsor,
                    "start_date": row.start_date,
                    "completion_date": row.completion_date,
                })

            print(f"Found {len(trials)} trials from database")
            return trials

    except Exception as e:
        print(f"Database search error: {e}")
        import traceback
        traceback.print_exc()
        return []


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

        # Get top trials and score dimensions with Claude
        top_trials = trials[:top_n]

        # Score dimensions using Claude for top trials
        try:
            scored_trials = await score_trial_dimensions_with_claude(protocol_info, top_trials[:10])
            # Merge dimension scores back
            score_map = {t["nct_id"]: t.get("dimension_scores", {}) for t in scored_trials}
            for trial in top_trials:
                if trial["nct_id"] in score_map:
                    trial["dimension_scores"] = score_map[trial["nct_id"]]
                else:
                    # Default scores for trials not scored by Claude
                    trial["dimension_scores"] = {
                        "condition": 50, "intervention": 50, "endpoint": 50,
                        "population": 50, "design": 50
                    }
        except Exception as e:
            print(f"Dimension scoring error: {e}")
            for trial in top_trials:
                trial["dimension_scores"] = {
                    "condition": 50, "intervention": 50, "endpoint": 50,
                    "population": 50, "design": 50
                }

        return top_trials

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
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,EnrollmentCount,Condition,InterventionName,EligibilityCriteria,PrimaryOutcome,StartDate,CompletionDate,LeadSponsorName,StudyType,LocationCity,LocationState,LocationCountry,LocationFacility"
    }

    # Note: Removed filter.phase and intervention term search
    # The intervention from the protocol is the drug being tested, which won't exist in other trials
    # Phase matching and similarity are handled by semantic matching with OpenAI embeddings

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

                # Extract locations
                contacts_locations = proto.get("contactsLocationsModule", {})
                locations_list = contacts_locations.get("locations", [])
                locations = []
                countries = set()
                for loc in locations_list[:20]:  # Limit to 20 locations
                    country = loc.get("country", "")
                    if country:
                        countries.add(country)
                    locations.append({
                        "facility": loc.get("facility", ""),
                        "city": loc.get("city", ""),
                        "state": loc.get("state", ""),
                        "country": country
                    })

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
                    "locations": locations,
                    "countries": list(countries),
                    "num_sites": len(locations_list)
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


async def analyze_enrollment_forecast_with_claude(protocol_info: Dict, similar_trials: List[Dict]) -> Dict:
    """
    Use Claude to generate detailed enrollment forecast with bottlenecks, assumptions, and scenario simulations.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Calculate stats from similar trials
    recruiting_trials = [t for t in similar_trials if t.get("status") == "RECRUITING"]
    completed_trials = [t for t in similar_trials if t.get("status") == "COMPLETED"]
    terminated_trials = [t for t in similar_trials if t.get("status") in ["TERMINATED", "WITHDRAWN"]]

    avg_enrollment = sum(t.get("enrollment", 0) for t in completed_trials) / len(completed_trials) if completed_trials else 100

    prompt = f"""You are an expert clinical operations strategist specializing in enrollment forecasting and trial optimization.

PROTOCOL BEING ANALYZED:
- Condition: {protocol_info.get('condition', 'Unknown')}
- Therapeutic Area: {protocol_info.get('therapeutic_area', 'Unknown')}
- Phase: {protocol_info.get('phase', 'Unknown')}
- Intervention: {protocol_info.get('intervention_name', 'Unknown')} ({protocol_info.get('intervention_type', 'Unknown')})
- Target Enrollment: {protocol_info.get('target_enrollment', 100)}
- Study Duration: {protocol_info.get('study_duration_months', 'Unknown')} months
- Inclusion Criteria Count: {len(protocol_info.get('inclusion_criteria', []))}
- Exclusion Criteria Count: {len(protocol_info.get('exclusion_criteria', []))}
- Key Exclusions: {protocol_info.get('exclusion_criteria', [])[:5]}

SIMILAR TRIALS LANDSCAPE:
- Total Similar Trials Found: {len(similar_trials)}
- Currently Recruiting: {len(recruiting_trials)}
- Completed: {len(completed_trials)}
- Terminated/Withdrawn: {len(terminated_trials)}
- Average Enrollment in Completed Trials: {avg_enrollment:.0f}

Based on this protocol and competitive landscape, provide a DETAILED enrollment forecast analysis in JSON format:

{{
    "target_enrollment": {protocol_info.get('target_enrollment', 100)},
    "estimated_duration_months": <realistic estimate based on similar trials>,
    "duration_range_low": <optimistic scenario months>,
    "duration_range_high": <conservative scenario months>,
    "scenarios": [
        {{
            "name": "base",
            "months": <base case duration>,
            "probability": 50,
            "assumptions": [
                "<specific assumption about site activation>",
                "<specific assumption about enrollment rate per site>",
                "<specific assumption about screen failure rate>",
                "<specific assumption about competitive environment>",
                "<specific assumption about regulatory timeline>"
            ]
        }},
        {{
            "name": "optimistic",
            "months": <optimistic duration>,
            "probability": 25,
            "assumptions": [
                "<what would need to happen for this scenario>"
            ]
        }},
        {{
            "name": "conservative",
            "months": <conservative duration>,
            "probability": 25,
            "assumptions": [
                "<what risks could cause this delay>"
            ]
        }}
    ],
    "scenarios_simulator": [
        {{
            "change": "<specific actionable change, e.g., 'Add 15 sites in Eastern Europe'>",
            "impact_months": <negative number for time saved, e.g., -3>
        }},
        {{
            "change": "<another specific change>",
            "impact_months": <impact>
        }},
        {{
            "change": "<enrollment rate improvement action>",
            "impact_months": <impact>
        }},
        {{
            "change": "<eligibility criteria adjustment>",
            "impact_months": <impact>
        }},
        {{
            "change": "<patient engagement strategy>",
            "impact_months": <impact>
        }}
    ],
    "bottlenecks": [
        {{
            "issue": "<specific bottleneck title>",
            "severity": "critical|moderate|manageable",
            "impact": "<detailed description of how this affects enrollment, be specific with numbers>",
            "mitigation": "<actionable mitigation strategy with specifics>"
        }},
        {{
            "issue": "<second bottleneck>",
            "severity": "<severity>",
            "impact": "<impact description>",
            "mitigation": "<mitigation strategy>"
        }},
        {{
            "issue": "<third bottleneck>",
            "severity": "<severity>",
            "impact": "<impact description>",
            "mitigation": "<mitigation strategy>"
        }},
        {{
            "issue": "<fourth bottleneck if applicable>",
            "severity": "<severity>",
            "impact": "<impact description>",
            "mitigation": "<mitigation strategy>"
        }}
    ],
    "enrollment_rate_per_site_month": <realistic rate based on therapeutic area>,
    "recommended_sites": <number>,
    "recommended_countries": <number>,
    "patients_per_site": <realistic target>,
    "screen_failure_prediction": <percentage, e.g., 35>,
    "site_activation_timeline_weeks": <realistic estimate>,
    "key_enrollment_risks": [
        "<risk 1>",
        "<risk 2>",
        "<risk 3>"
    ],
    "enrollment_acceleration_strategies": [
        "<strategy 1>",
        "<strategy 2>",
        "<strategy 3>"
    ]
}}

Be specific and realistic. Use data from the similar trials landscape to inform your estimates. Consider the competitive environment with {len(recruiting_trials)} currently recruiting trials.
Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


async def analyze_detailed_risks_with_claude(protocol_info: Dict, similar_trials: List[Dict]) -> Dict:
    """
    Use Claude to generate comprehensive risk analysis with detailed breakdowns and mitigation strategies.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Calculate outcome stats
    completed_trials = [t for t in similar_trials if t.get("status") == "COMPLETED"]
    terminated_trials = [t for t in similar_trials if t.get("status") in ["TERMINATED", "WITHDRAWN"]]
    success_rate = len(completed_trials) / len(similar_trials) * 100 if similar_trials else 50
    termination_rate = len(terminated_trials) / len(similar_trials) * 100 if similar_trials else 20

    # Get termination reasons if available
    termination_reasons = [t.get("why_stopped", "") for t in terminated_trials if t.get("why_stopped")]

    prompt = f"""You are a clinical trial risk management expert with deep expertise in protocol design and trial operations.

PROTOCOL UNDER ANALYSIS:
- Condition: {protocol_info.get('condition', 'Unknown')}
- Therapeutic Area: {protocol_info.get('therapeutic_area', 'Unknown')}
- Phase: {protocol_info.get('phase', 'Unknown')}
- Intervention: {protocol_info.get('intervention_name', 'Unknown')} ({protocol_info.get('intervention_type', 'Unknown')})
- Primary Endpoint: {protocol_info.get('primary_endpoint', 'Unknown')}
- Secondary Endpoints: {protocol_info.get('secondary_endpoints', [])[:3]}
- Target Enrollment: {protocol_info.get('target_enrollment', 100)}
- Comparator: {protocol_info.get('comparator', 'Unknown')}
- Inclusion Criteria: {protocol_info.get('inclusion_criteria', [])[:5]}
- Exclusion Criteria: {protocol_info.get('exclusion_criteria', [])[:5]}

SIMILAR TRIALS OUTCOMES:
- Total Analyzed: {len(similar_trials)}
- Completion Rate: {success_rate:.1f}%
- Termination Rate: {termination_rate:.1f}%
- Known Termination Reasons: {termination_reasons[:5] if termination_reasons else 'Not available'}

Provide a COMPREHENSIVE risk analysis in JSON format:

{{
    "overall_risk_assessment": {{
        "risk_score": <0-100, higher = more risk>,
        "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
        "risk_summary": "<2-3 sentence executive summary of key risks>",
        "confidence_level": "<HIGH|MEDIUM|LOW based on available data>"
    }},
    "risk_categories": [
        {{
            "category": "Enrollment Risk",
            "score": <0-100>,
            "level": "LOW|MEDIUM|HIGH|CRITICAL",
            "key_factors": [
                "<specific factor affecting enrollment>",
                "<second factor>",
                "<third factor>"
            ],
            "detailed_analysis": "<paragraph explaining enrollment risks specific to this protocol>",
            "mitigation_strategies": [
                {{
                    "strategy": "<specific mitigation>",
                    "effort": "LOW|MEDIUM|HIGH",
                    "impact": "LOW|MEDIUM|HIGH",
                    "timeline": "<when to implement>"
                }}
            ]
        }},
        {{
            "category": "Endpoint Risk",
            "score": <0-100>,
            "level": "<level>",
            "key_factors": ["<factors>"],
            "detailed_analysis": "<analysis of endpoint selection risks>",
            "mitigation_strategies": [<strategies>]
        }},
        {{
            "category": "Operational Risk",
            "score": <0-100>,
            "level": "<level>",
            "key_factors": ["<factors>"],
            "detailed_analysis": "<analysis of operational complexity>",
            "mitigation_strategies": [<strategies>]
        }},
        {{
            "category": "Regulatory Risk",
            "score": <0-100>,
            "level": "<level>",
            "key_factors": ["<factors>"],
            "detailed_analysis": "<regulatory considerations>",
            "mitigation_strategies": [<strategies>]
        }},
        {{
            "category": "Competitive Risk",
            "score": <0-100>,
            "level": "<level>",
            "key_factors": ["<factors>"],
            "detailed_analysis": "<competitive landscape analysis>",
            "mitigation_strategies": [<strategies>]
        }}
    ],
    "amendment_risk": {{
        "probability_percentage": <0-100>,
        "expected_amendments": <number>,
        "likely_amendment_areas": [
            {{
                "area": "<e.g., Eligibility Criteria>",
                "probability": <percentage>,
                "reason": "<why this might need amendment>",
                "prevention": "<how to prevent>"
            }}
        ],
        "amendment_impact_assessment": "<description of how amendments could affect timeline and cost>"
    }},
    "success_probability": {{
        "overall_success_rate": <percentage>,
        "phase_benchmark": <typical success rate for this phase>,
        "therapeutic_area_benchmark": <typical success rate for this therapeutic area>,
        "factors_increasing_success": [
            "<positive factor 1>",
            "<positive factor 2>",
            "<positive factor 3>"
        ],
        "factors_decreasing_success": [
            "<negative factor 1>",
            "<negative factor 2>",
            "<negative factor 3>"
        ],
        "comparison_to_similar": "<how this protocol compares to similar trials that succeeded vs failed>"
    }},
    "critical_watch_items": [
        {{
            "item": "<specific item to monitor>",
            "trigger": "<what would indicate a problem>",
            "response": "<recommended response if triggered>",
            "monitoring_frequency": "<how often to check>"
        }}
    ],
    "risk_mitigation_priorities": [
        {{
            "priority": 1,
            "risk": "<highest priority risk>",
            "action": "<recommended action>",
            "owner": "<who should own this>",
            "deadline": "<when to complete>"
        }},
        {{
            "priority": 2,
            "risk": "<second priority>",
            "action": "<action>",
            "owner": "<owner>",
            "deadline": "<deadline>"
        }},
        {{
            "priority": 3,
            "risk": "<third priority>",
            "action": "<action>",
            "owner": "<owner>",
            "deadline": "<deadline>"
        }}
    ]
}}

Be specific to this protocol. Reference the similar trial data to support your assessments. Provide actionable recommendations.
Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


async def analyze_protocol_optimization_with_claude(protocol_info: Dict, similar_trials: List[Dict]) -> Dict:
    """
    Use Claude to generate detailed protocol optimization analysis including complexity assessment,
    design recommendations, and success predictions.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Calculate stats
    completed_trials = [t for t in similar_trials if t.get("status") == "COMPLETED"]
    terminated_trials = [t for t in similar_trials if t.get("status") in ["TERMINATED", "WITHDRAWN"]]

    prompt = f"""You are an expert clinical trial protocol designer with deep expertise in protocol optimization and regulatory strategy.

PROTOCOL UNDER ANALYSIS:
- Condition: {protocol_info.get('condition', 'Unknown')}
- Therapeutic Area: {protocol_info.get('therapeutic_area', 'Unknown')}
- Phase: {protocol_info.get('phase', 'Unknown')}
- Intervention: {protocol_info.get('intervention_name', 'Unknown')} ({protocol_info.get('intervention_type', 'Unknown')})
- Primary Endpoint: {protocol_info.get('primary_endpoint', 'Unknown')}
- Secondary Endpoints: {protocol_info.get('secondary_endpoints', [])[:5]}
- Target Enrollment: {protocol_info.get('target_enrollment', 100)}
- Study Duration: {protocol_info.get('study_duration_months', 'Unknown')} months
- Comparator: {protocol_info.get('comparator', 'Unknown')}
- Number of Arms: {protocol_info.get('number_of_arms', 2)}
- Blinding: {protocol_info.get('blinding', 'Unknown')}
- Inclusion Criteria: {protocol_info.get('inclusion_criteria', [])[:5]}
- Exclusion Criteria: {protocol_info.get('exclusion_criteria', [])[:5]}

SIMILAR TRIALS DATA:
- Total Similar: {len(similar_trials)}
- Completed: {len(completed_trials)}
- Terminated/Withdrawn: {len(terminated_trials)}
- Termination Rate: {len(terminated_trials)/len(similar_trials)*100 if similar_trials else 0:.1f}%

Provide a COMPREHENSIVE protocol optimization analysis in JSON format:

{{
    "complexity_assessment": {{
        "overall_score": <0-100, higher = more complex>,
        "complexity_level": "LOW|MODERATE|HIGH|VERY_HIGH",
        "complexity_rationale": "<detailed explanation of complexity drivers>",
        "complexity_factors": [
            {{
                "factor": "<specific complexity factor>",
                "score": <0-100>,
                "impact": "LOW|MEDIUM|HIGH",
                "description": "<how this affects trial execution>",
                "simplification_opportunity": "<how to reduce this complexity>"
            }}
        ],
        "comparison_to_similar": "<how this protocol's complexity compares to similar trials>",
        "operational_burden_estimate": "<LOW|MODERATE|HIGH|VERY_HIGH>"
    }},
    "design_strengths": [
        {{
            "strength": "<specific protocol strength>",
            "impact": "positive|very_positive",
            "explanation": "<why this is a strength>",
            "leverage_recommendation": "<how to maximize this advantage>"
        }}
    ],
    "design_weaknesses": [
        {{
            "weakness": "<specific protocol weakness>",
            "severity": "minor|moderate|major|critical",
            "impact": "<how this affects trial success>",
            "recommendation": "<specific actionable improvement>",
            "priority": <1-5, 1 being highest priority>
        }}
    ],
    "optimization_recommendations": [
        {{
            "category": "<e.g., Eligibility, Endpoints, Visit Schedule, etc.>",
            "recommendation": "<specific recommendation>",
            "rationale": "<why this improvement is recommended>",
            "expected_impact": "<quantified impact if possible>",
            "implementation_effort": "LOW|MEDIUM|HIGH",
            "priority": "critical|high|medium|low"
        }}
    ],
    "success_prediction": {{
        "overall_success_probability": <0-100 percentage>,
        "confidence_level": "LOW|MEDIUM|HIGH",
        "phase_benchmark_success_rate": <typical success rate for this phase>,
        "therapeutic_area_benchmark": <typical success rate for this TA>,
        "relative_assessment": "<better_than_average|average|below_average>",
        "key_success_drivers": [
            "<factor 1 that will drive success>",
            "<factor 2>",
            "<factor 3>"
        ],
        "key_risk_drivers": [
            "<factor 1 that threatens success>",
            "<factor 2>",
            "<factor 3>"
        ],
        "success_optimization_actions": [
            "<action 1 to improve success probability>",
            "<action 2>",
            "<action 3>"
        ]
    }},
    "terminated_trial_learnings": [
        {{
            "pattern": "<common termination pattern from similar trials>",
            "frequency": "<how often this occurred>",
            "relevance_to_protocol": "HIGH|MEDIUM|LOW",
            "preventive_action": "<what to do to avoid this>"
        }}
    ],
    "design_comparison": [
        {{
            "aspect": "<design aspect being compared>",
            "your_protocol": "<how your protocol handles this>",
            "industry_standard": "<what similar trials typically do>",
            "recommendation": "<keep|modify|reconsider>",
            "rationale": "<why>"
        }}
    ],
    "regulatory_design_considerations": {{
        "fda_alignment_score": <0-100>,
        "ema_alignment_score": <0-100>,
        "regulatory_strengths": ["<strength 1>", "<strength 2>"],
        "regulatory_concerns": ["<concern 1>", "<concern 2>"],
        "recommended_regulatory_strategy": "<brief strategy recommendation>"
    }}
}}

Be specific and actionable. Reference similar trial data to support recommendations. Focus on practical improvements.
Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3500,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


async def analyze_eligibility_with_claude(protocol_info: Dict, similar_trials: List[Dict]) -> Dict:
    """
    Use Claude to analyze eligibility criteria, predict screen failure rates,
    and provide optimization recommendations.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    inclusion = protocol_info.get('inclusion_criteria', [])
    exclusion = protocol_info.get('exclusion_criteria', [])

    prompt = f"""You are an expert clinical trial eligibility specialist with extensive experience in patient recruitment and protocol design.

PROTOCOL ELIGIBILITY CRITERIA:

INCLUSION CRITERIA ({len(inclusion)} criteria):
{json.dumps(inclusion, indent=2) if inclusion else "Not specified"}

EXCLUSION CRITERIA ({len(exclusion)} criteria):
{json.dumps(exclusion, indent=2) if exclusion else "Not specified"}

PROTOCOL CONTEXT:
- Condition: {protocol_info.get('condition', 'Unknown')}
- Therapeutic Area: {protocol_info.get('therapeutic_area', 'Unknown')}
- Phase: {protocol_info.get('phase', 'Unknown')}
- Target Enrollment: {protocol_info.get('target_enrollment', 100)}
- Intervention: {protocol_info.get('intervention_name', 'Unknown')}

SIMILAR TRIALS: {len(similar_trials)} trials analyzed in this therapeutic area

Provide a COMPREHENSIVE eligibility analysis in JSON format:

{{
    "screen_failure_prediction": {{
        "predicted_rate": <percentage, e.g., 35>,
        "rate_range": {{"low": <optimistic>, "high": <conservative>}},
        "assessment": "excellent|good|moderate|concerning|poor",
        "screen_to_enroll_ratio": "<e.g., 2.5:1>",
        "benchmark_ratio": "<industry benchmark for this TA>",
        "best_in_class_ratio": "<best observed in similar trials>",
        "rationale": "<detailed explanation of prediction>",
        "primary_failure_drivers": [
            {{
                "criterion": "<specific criterion causing failures>",
                "estimated_failure_contribution": <percentage>,
                "modifiability": "easily_modifiable|modifiable|difficult|fixed"
            }}
        ]
    }},
    "criteria_analysis": {{
        "inclusion_assessment": {{
            "count": {len(inclusion)},
            "complexity": "LOW|MODERATE|HIGH",
            "restrictiveness": "BROAD|MODERATE|NARROW|VERY_NARROW",
            "problematic_criteria": [
                {{
                    "criterion": "<specific criterion>",
                    "issue": "<what's problematic>",
                    "impact": "HIGH|MEDIUM|LOW",
                    "recommendation": "<how to improve>"
                }}
            ]
        }},
        "exclusion_assessment": {{
            "count": {len(exclusion)},
            "complexity": "LOW|MODERATE|HIGH",
            "restrictiveness": "PERMISSIVE|MODERATE|STRICT|VERY_STRICT",
            "problematic_criteria": [
                {{
                    "criterion": "<specific criterion>",
                    "issue": "<what's problematic>",
                    "impact": "HIGH|MEDIUM|LOW",
                    "recommendation": "<how to improve>"
                }}
            ]
        }}
    }},
    "criterion_benchmarks": [
        {{
            "criterion_type": "<e.g., Age, Lab values, Prior therapy>",
            "your_criteria": "<your protocol's criterion>",
            "industry_standard": "<what similar trials typically use>",
            "assessment": "more_restrictive|aligned|less_restrictive",
            "impact_on_recruitment": "HIGH|MEDIUM|LOW",
            "recommendation": "<keep|relax|tighten|remove>"
        }}
    ],
    "patient_pool_estimation": {{
        "addressable_population": "<estimated global patient population>",
        "after_inclusion_filter": "<population meeting inclusion>",
        "after_exclusion_filter": "<population meeting all criteria>",
        "estimated_eligible_percentage": <percentage of total population>,
        "recruitment_feasibility": "EXCELLENT|GOOD|MODERATE|CHALLENGING|VERY_CHALLENGING",
        "geographic_considerations": "<where patients are most available>",
        "stages": [
            {{
                "stage": "<e.g., Total patients with condition>",
                "count": "<estimated number>",
                "percentage": <percentage remaining>
            }}
        ]
    }},
    "optimization_recommendations": [
        {{
            "priority": <1-5, 1 is highest>,
            "criterion_type": "inclusion|exclusion",
            "current": "<current criterion>",
            "recommended": "<recommended change>",
            "rationale": "<why this change>",
            "expected_impact": "<e.g., +15% eligible population>",
            "risk_assessment": "<any risks of this change>"
        }}
    ],
    "competitive_differentiation": {{
        "vs_recruiting_trials": "<how your criteria compare to recruiting competitors>",
        "patient_experience_burden": "LOW|MODERATE|HIGH",
        "suggestions_for_differentiation": ["<suggestion 1>", "<suggestion 2>"]
    }},
    "special_populations": {{
        "elderly_inclusion": "included|excluded|limited",
        "pediatric_consideration": "applicable|not_applicable",
        "pregnancy_handling": "<how handled>",
        "comorbidity_flexibility": "flexible|moderate|strict",
        "recommendations": ["<recommendation for special populations>"]
    }},
    "regulatory_alignment": {{
        "fda_guidance_alignment": "ALIGNED|PARTIALLY_ALIGNED|MISALIGNED",
        "ema_requirements": "MET|PARTIALLY_MET|NOT_MET",
        "potential_regulatory_concerns": ["<concern 1>"],
        "label_implications": "<how criteria might affect eventual label>"
    }}
}}

Be specific with criterion-level recommendations. Quantify impacts where possible.
Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3500,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


async def analyze_endpoints_with_claude(protocol_info: Dict, similar_trials: List[Dict]) -> Dict:
    """
    Use Claude to analyze endpoint selection, FDA alignment, and sample size considerations.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Extract endpoint info from similar trials
    similar_endpoints = [t.get("primary_outcomes", "")[:200] for t in similar_trials[:10] if t.get("primary_outcomes")]

    prompt = f"""You are an expert clinical trial endpoint strategist with deep expertise in regulatory requirements and statistical powering.

PROTOCOL ENDPOINTS:
- Primary Endpoint: {protocol_info.get('primary_endpoint', 'Not specified')}
- Secondary Endpoints: {json.dumps(protocol_info.get('secondary_endpoints', []), indent=2)}

PROTOCOL CONTEXT:
- Condition: {protocol_info.get('condition', 'Unknown')}
- Therapeutic Area: {protocol_info.get('therapeutic_area', 'Unknown')}
- Phase: {protocol_info.get('phase', 'Unknown')}
- Intervention: {protocol_info.get('intervention_name', 'Unknown')} ({protocol_info.get('intervention_type', 'Unknown')})
- Target Enrollment: {protocol_info.get('target_enrollment', 100)}
- Comparator: {protocol_info.get('comparator', 'Unknown')}

ENDPOINTS FROM SIMILAR TRIALS:
{json.dumps(similar_endpoints, indent=2)}

Provide a COMPREHENSIVE endpoint analysis in JSON format:

{{
    "primary_endpoint_analysis": {{
        "endpoint": "{protocol_info.get('primary_endpoint', 'Not specified')}",
        "endpoint_type": "<e.g., clinical, surrogate, PRO, composite>",
        "measurement_complexity": "LOW|MODERATE|HIGH|VERY_HIGH",
        "assessment_frequency_recommendation": "<how often to assess>",
        "strengths": [
            "<strength 1>",
            "<strength 2>"
        ],
        "weaknesses": [
            "<weakness 1>",
            "<weakness 2>"
        ],
        "overall_assessment": "EXCELLENT|GOOD|ADEQUATE|CONCERNING|POOR"
    }},
    "fda_alignment": {{
        "status": "STRONG|MODERATE|WEAK|MISALIGNED",
        "alignment_score": <0-100>,
        "rationale": "<detailed explanation>",
        "relevant_guidance_documents": ["<FDA guidance 1>", "<FDA guidance 2>"],
        "precedent_approvals": ["<similar approved drugs that used this endpoint>"],
        "potential_concerns": ["<concern 1>"],
        "recommendations_for_alignment": ["<recommendation 1>"]
    }},
    "ema_alignment": {{
        "status": "STRONG|MODERATE|WEAK|MISALIGNED",
        "alignment_score": <0-100>,
        "key_differences_from_fda": "<any differences in EMA requirements>",
        "recommendations": ["<recommendation for EMA alignment>"]
    }},
    "endpoint_benchmarking": {{
        "primary_endpoint_distribution": [
            {{
                "endpoint_type": "<e.g., Clinical response>",
                "percentage_of_trials": <percentage>,
                "typical_effect_size": "<observed effect sizes>",
                "your_alignment": "USING|SIMILAR|DIFFERENT"
            }}
        ],
        "your_endpoint_vs_field": "<how your endpoint compares to field standard>",
        "differentiation_assessment": "STANDARD|INNOVATIVE|RISKY"
    }},
    "sample_size_analysis": {{
        "recommended_sample_size": <number>,
        "sample_size_range": {{"minimum": <number>, "optimal": <number>, "conservative": <number>}},
        "power_assumptions": {{
            "alpha": 0.05,
            "power": 0.80,
            "expected_effect_size": "<based on similar trials>",
            "control_rate_estimate": "<estimated from literature>",
            "treatment_rate_estimate": "<expected treatment effect>"
        }},
        "scenarios": [
            {{
                "scenario": "Conservative",
                "control_rate": "<rate>",
                "treatment_rate": "<rate>",
                "effect_size": "<difference>",
                "patients_needed": <number>,
                "power": "80%",
                "rationale": "<why this scenario>"
            }},
            {{
                "scenario": "Base Case",
                "control_rate": "<rate>",
                "treatment_rate": "<rate>",
                "effect_size": "<difference>",
                "patients_needed": <number>,
                "power": "80%",
                "rationale": "<why this is base case>"
            }},
            {{
                "scenario": "Optimistic",
                "control_rate": "<rate>",
                "treatment_rate": "<rate>",
                "effect_size": "<difference>",
                "patients_needed": <number>,
                "power": "80%",
                "rationale": "<why this is optimistic>"
            }}
        ],
        "dropout_adjustment": {{
            "expected_dropout_rate": <percentage>,
            "adjusted_enrollment_target": <number accounting for dropout>
        }},
        "sample_size_risks": [
            "<risk 1 - e.g., effect size assumption too optimistic>",
            "<risk 2>"
        ]
    }},
    "secondary_endpoints_analysis": [
        {{
            "endpoint": "<secondary endpoint>",
            "purpose": "<why included - supportive, safety, exploratory>",
            "assessment": "APPROPRIATE|CONSIDER_REMOVING|CONSIDER_ADDING",
            "recommendation": "<any changes recommended>"
        }}
    ],
    "missing_endpoints_recommendations": [
        {{
            "suggested_endpoint": "<endpoint to consider adding>",
            "rationale": "<why this would be valuable>",
            "regulatory_value": "HIGH|MEDIUM|LOW",
            "operational_burden": "LOW|MEDIUM|HIGH"
        }}
    ],
    "endpoint_collection_burden": {{
        "patient_burden_score": <0-100, higher = more burdensome>,
        "site_burden_score": <0-100>,
        "assessment": "<overall burden assessment>",
        "simplification_opportunities": ["<opportunity 1>"]
    }},
    "historical_benchmarks": {{
        "similar_trials_effect_sizes": [
            {{
                "trial_id": "<NCT ID if available>",
                "endpoint": "<endpoint used>",
                "effect_size_achieved": "<what was observed>",
                "met_primary_endpoint": "YES|NO|UNKNOWN"
            }}
        ],
        "field_average_effect_size": "<typical effect size in this area>",
        "your_target_vs_benchmark": "AGGRESSIVE|ALIGNED|CONSERVATIVE"
    }}
}}

Be specific about regulatory requirements and statistical assumptions. Provide evidence-based recommendations.
Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


async def analyze_amendment_risk_with_claude(protocol_info: Dict, similar_trials: List[Dict]) -> Dict:
    """
    Use Claude to analyze amendment risk with detailed predictions and prevention strategies.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = f"""You are an expert clinical trial protocol amendment specialist with extensive experience analyzing amendment patterns and prevention strategies.

PROTOCOL UNDER ANALYSIS:
- Condition: {protocol_info.get('condition', 'Unknown')}
- Therapeutic Area: {protocol_info.get('therapeutic_area', 'Unknown')}
- Phase: {protocol_info.get('phase', 'Unknown')}
- Intervention: {protocol_info.get('intervention_name', 'Unknown')}
- Primary Endpoint: {protocol_info.get('primary_endpoint', 'Unknown')}
- Target Enrollment: {protocol_info.get('target_enrollment', 100)}
- Number of Arms: {protocol_info.get('number_of_arms', 2)}
- Inclusion Criteria Count: {len(protocol_info.get('inclusion_criteria', []))}
- Exclusion Criteria Count: {len(protocol_info.get('exclusion_criteria', []))}

SIMILAR TRIALS ANALYZED: {len(similar_trials)} trials

Provide a COMPREHENSIVE amendment risk analysis in JSON format:

{{
    "overall_amendment_risk": {{
        "risk_score": <0-100>,
        "risk_level": "LOW|MODERATE|HIGH|VERY_HIGH",
        "predicted_amendments": <expected number of amendments>,
        "amendment_probability": <percentage likelihood of at least one amendment>,
        "rationale": "<detailed explanation of risk assessment>",
        "industry_benchmark": "<typical amendment rate for this phase/TA>"
    }},
    "risk_factors": [
        {{
            "category": "<e.g., Eligibility Criteria, Endpoints, Visit Schedule>",
            "risk_level": "LOW|MODERATE|HIGH|CRITICAL",
            "risk_score": <0-100>,
            "specific_concerns": ["<concern 1>", "<concern 2>"],
            "likelihood_of_amendment": <percentage>,
            "typical_timing": "<when this type of amendment usually occurs>",
            "impact_if_amended": "MINOR|MODERATE|MAJOR|CRITICAL",
            "prevention_strategies": [
                {{
                    "strategy": "<specific prevention action>",
                    "effort": "LOW|MEDIUM|HIGH",
                    "effectiveness": "LOW|MEDIUM|HIGH"
                }}
            ]
        }}
    ],
    "likely_amendment_areas": [
        {{
            "area": "<specific protocol section>",
            "probability": <percentage>,
            "reason": "<why this area is likely to need amendment>",
            "typical_change": "<what kind of change is usually made>",
            "prevention": "<how to prevent this amendment>",
            "early_warning_signs": ["<sign 1>", "<sign 2>"]
        }}
    ],
    "historical_patterns": [
        {{
            "category": "<amendment category>",
            "frequency": <number of similar trials with this amendment>,
            "percentage": <percentage of similar trials>,
            "typical_timing": "<when in trial lifecycle>",
            "common_triggers": ["<trigger 1>", "<trigger 2>"],
            "common_solutions": ["<solution 1>", "<solution 2>"],
            "relevance_to_your_protocol": "HIGH|MEDIUM|LOW"
        }}
    ],
    "amendment_timeline_prediction": {{
        "highest_risk_period": "<e.g., First 6 months of enrollment>",
        "risk_by_phase": [
            {{"phase": "Startup", "risk_level": "<level>", "common_amendments": ["<type>"]}},
            {{"phase": "Early Enrollment", "risk_level": "<level>", "common_amendments": ["<type>"]}},
            {{"phase": "Mid-Study", "risk_level": "<level>", "common_amendments": ["<type>"]}},
            {{"phase": "Late-Study", "risk_level": "<level>", "common_amendments": ["<type>"]}}
        ]
    }},
    "cost_impact_analysis": {{
        "estimated_cost_per_amendment": "<cost range>",
        "estimated_delay_per_amendment": "<typical delay in weeks/months>",
        "total_risk_exposure": "<estimated total cost/time risk>",
        "roi_of_prevention": "<value of preventing amendments>"
    }},
    "prevention_action_plan": [
        {{
            "priority": <1-5>,
            "action": "<specific preventive action>",
            "target_risk": "<which risk this addresses>",
            "timeline": "<when to implement>",
            "responsible_party": "<who should own this>",
            "success_metric": "<how to measure effectiveness>"
        }}
    ],
    "protocol_quality_indicators": {{
        "clarity_score": <0-100>,
        "completeness_score": <0-100>,
        "consistency_score": <0-100>,
        "operational_feasibility_score": <0-100>,
        "areas_needing_clarification": ["<area 1>", "<area 2>"],
        "ambiguous_language_flags": ["<flagged item 1>"]
    }},
    "benchmarks": {{
        "industry_avg_amendment_rate": <percentage>,
        "phase_specific_benchmark": <percentage for this phase>,
        "therapeutic_area_benchmark": <percentage for this TA>,
        "best_in_class_rate": <lowest observed rate>,
        "your_predicted_vs_benchmark": "BETTER|ALIGNED|WORSE"
    }}
}}

Be specific and actionable. Provide data-driven predictions based on similar trial patterns.
Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


async def analyze_site_intelligence_with_claude(protocol_info: Dict, similar_trials: List[Dict], site_data: Dict) -> Dict:
    """
    Use Claude to generate detailed site selection strategy and intelligence.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Extract site statistics
    top_countries = site_data.get("top_countries", [])
    top_sites = site_data.get("top_sites", [])
    avg_sites = site_data.get("avg_sites_per_trial", 0)

    prompt = f"""You are an expert clinical trial site selection strategist with deep experience in global trial operations.

PROTOCOL DETAILS:
- Condition: {protocol_info.get('condition', 'Unknown')}
- Therapeutic Area: {protocol_info.get('therapeutic_area', 'Unknown')}
- Phase: {protocol_info.get('phase', 'Unknown')}
- Target Enrollment: {protocol_info.get('target_enrollment', 100)}
- Intervention: {protocol_info.get('intervention_name', 'Unknown')}

SIMILAR TRIALS DATA:
- Total Similar Trials: {len(similar_trials)}
- Currently Recruiting: {len([t for t in similar_trials if t.get('status') == 'RECRUITING'])}
- Average Sites per Trial: {avg_sites}
- Top Countries Used: {json.dumps(top_countries[:10], indent=2)}
- Top Performing Sites: {json.dumps(top_sites[:10], indent=2)}

Provide a COMPREHENSIVE site selection strategy in JSON format:

{{
    "site_strategy": {{
        "recommended_total_sites": <number>,
        "recommended_countries": <number>,
        "patients_per_site_target": <number>,
        "site_activation_timeline_weeks": <number>,
        "strategy_rationale": "<detailed explanation of strategy>",
        "key_considerations": ["<consideration 1>", "<consideration 2>", "<consideration 3>"]
    }},
    "geographic_strategy": {{
        "primary_regions": [
            {{
                "region": "<e.g., North America, Western Europe>",
                "recommended_sites": <number>,
                "rationale": "<why this region>",
                "key_countries": ["<country 1>", "<country 2>"],
                "regulatory_considerations": "<specific regulatory notes>"
            }}
        ],
        "emerging_regions": [
            {{
                "region": "<e.g., Eastern Europe, Asia Pacific>",
                "recommended_sites": <number>,
                "advantages": ["<advantage 1>", "<advantage 2>"],
                "challenges": ["<challenge 1>", "<challenge 2>"],
                "mitigation_strategies": ["<strategy 1>"]
            }}
        ],
        "regions_to_avoid": [
            {{
                "region": "<region>",
                "reason": "<why to avoid>"
            }}
        ]
    }},
    "country_recommendations": [
        {{
            "country": "<country name>",
            "priority": "HIGH|MEDIUM|LOW",
            "recommended_sites": <number>,
            "patient_availability": "EXCELLENT|GOOD|MODERATE|LIMITED",
            "regulatory_timeline": "<expected timeline>",
            "cost_index": "HIGH|MEDIUM|LOW",
            "experience_in_ta": "EXTENSIVE|MODERATE|LIMITED",
            "key_advantages": ["<advantage 1>"],
            "key_challenges": ["<challenge 1>"],
            "recommended_site_types": ["<academic centers, community sites, etc>"]
        }}
    ],
    "site_selection_criteria": {{
        "must_have": [
            "<criterion 1>",
            "<criterion 2>",
            "<criterion 3>"
        ],
        "preferred": [
            "<criterion 1>",
            "<criterion 2>"
        ],
        "red_flags": [
            "<warning sign 1>",
            "<warning sign 2>"
        ]
    }},
    "site_performance_benchmarks": {{
        "enrollment_rate_target": "<patients per site per month>",
        "screen_failure_rate_target": "<percentage>",
        "protocol_deviation_threshold": "<percentage>",
        "query_rate_threshold": "<queries per patient>",
        "retention_rate_target": "<percentage>"
    }},
    "competitive_site_analysis": {{
        "sites_with_competing_trials": <estimated number>,
        "impact_assessment": "<how competition affects site selection>",
        "differentiation_strategies": [
            "<strategy 1>",
            "<strategy 2>"
        ]
    }},
    "risk_mitigation": {{
        "backup_sites_recommended": <number>,
        "regional_diversification": "<recommendation>",
        "contingency_countries": ["<country 1>", "<country 2>"],
        "early_warning_indicators": [
            "<indicator 1>",
            "<indicator 2>"
        ]
    }},
    "timeline_optimization": {{
        "parallel_activation_strategy": "<recommendation>",
        "fast_track_countries": ["<country 1>", "<country 2>"],
        "bottleneck_countries": ["<country 1>"],
        "recommended_activation_sequence": [
            {{"phase": "Wave 1", "countries": ["<country>"], "sites": <number>, "timeline": "<weeks>"}},
            {{"phase": "Wave 2", "countries": ["<country>"], "sites": <number>, "timeline": "<weeks>"}}
        ]
    }},
    "budget_considerations": {{
        "high_cost_regions": ["<region 1>"],
        "cost_effective_regions": ["<region 1>"],
        "cost_optimization_strategies": [
            "<strategy 1>",
            "<strategy 2>"
        ]
    }},
    "special_considerations": {{
        "rare_disease_factors": "<if applicable>",
        "pediatric_considerations": "<if applicable>",
        "regulatory_harmonization": "<recommendations for multi-regional submissions>"
    }}
}}

Be specific and actionable. Consider the competitive landscape from similar trials. Provide data-driven recommendations.
Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


async def analyze_competitive_landscape_with_claude(protocol_info: Dict, similar_trials: List[Dict]) -> Dict:
    """
    Use Claude to generate comprehensive competitive landscape analysis.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Extract competitive data
    recruiting = [t for t in similar_trials if t.get("status") == "RECRUITING"]
    completed = [t for t in similar_trials if t.get("status") == "COMPLETED"]
    terminated = [t for t in similar_trials if t.get("status") in ["TERMINATED", "WITHDRAWN"]]

    # Extract sponsors
    sponsors = {}
    for t in similar_trials:
        sponsor = t.get("sponsor", "Unknown")
        if sponsor not in sponsors:
            sponsors[sponsor] = {"count": 0, "recruiting": 0, "completed": 0}
        sponsors[sponsor]["count"] += 1
        if t.get("status") == "RECRUITING":
            sponsors[sponsor]["recruiting"] += 1
        elif t.get("status") == "COMPLETED":
            sponsors[sponsor]["completed"] += 1

    top_sponsors = sorted(sponsors.items(), key=lambda x: x[1]["count"], reverse=True)[:10]

    # Extract phases
    phases = {}
    for t in similar_trials:
        phase = t.get("phase", "Unknown")
        phases[phase] = phases.get(phase, 0) + 1

    prompt = f"""You are an expert competitive intelligence analyst specializing in clinical trials.

PROTOCOL UNDER ANALYSIS:
- Condition: {protocol_info.get('condition', 'Unknown')}
- Therapeutic Area: {protocol_info.get('therapeutic_area', 'Unknown')}
- Phase: {protocol_info.get('phase', 'Unknown')}
- Intervention: {protocol_info.get('intervention_name', 'Unknown')} ({protocol_info.get('intervention_type', 'Unknown')})
- Target Enrollment: {protocol_info.get('target_enrollment', 100)}
- Primary Endpoint: {protocol_info.get('primary_endpoint', 'Unknown')}

COMPETITIVE LANDSCAPE DATA:
- Total Similar Trials: {len(similar_trials)}
- Currently Recruiting: {len(recruiting)}
- Completed: {len(completed)}
- Terminated/Withdrawn: {len(terminated)}
- Phase Distribution: {json.dumps(phases, indent=2)}
- Top Sponsors: {json.dumps(dict(top_sponsors), indent=2)}

Provide a COMPREHENSIVE competitive landscape analysis in JSON format:

{{
    "overall_assessment": {{
        "competition_level": "LOW|MODERATE|HIGH|VERY_HIGH",
        "competition_score": <0-100>,
        "market_saturation": "<assessment of market saturation>",
        "executive_summary": "<2-3 sentence summary of competitive landscape>",
        "key_implications": ["<implication 1>", "<implication 2>", "<implication 3>"]
    }},
    "competitor_analysis": [
        {{
            "sponsor": "<sponsor name>",
            "threat_level": "HIGH|MEDIUM|LOW",
            "total_trials": <number>,
            "recruiting_trials": <number>,
            "competitive_position": "<leader|challenger|follower>",
            "key_programs": ["<program 1>", "<program 2>"],
            "strengths": ["<strength 1>", "<strength 2>"],
            "weaknesses": ["<weakness 1>", "<weakness 2>"],
            "strategic_implications": "<how to compete with this sponsor>"
        }}
    ],
    "market_dynamics": {{
        "stage_of_development": "<early|growing|mature|declining>",
        "unmet_medical_need": "HIGH|MODERATE|LOW",
        "recent_approvals": ["<recent approval 1>"],
        "pipeline_trends": "<description of pipeline trends>",
        "emerging_modalities": ["<modality 1>", "<modality 2>"],
        "market_size_estimate": "<if applicable>"
    }},
    "enrollment_competition": {{
        "competing_for_patients": <number of recruiting trials>,
        "patient_competition_intensity": "HIGH|MODERATE|LOW",
        "geographic_hotspots": ["<region with most competition>"],
        "underserved_regions": ["<region with less competition>"],
        "impact_on_enrollment_timeline": "<assessment>",
        "mitigation_strategies": [
            "<strategy 1>",
            "<strategy 2>",
            "<strategy 3>"
        ]
    }},
    "differentiation_analysis": {{
        "your_unique_factors": [
            {{
                "factor": "<differentiating factor>",
                "competitive_advantage": "<how it helps>",
                "leverage_strategy": "<how to leverage>"
            }}
        ],
        "areas_of_parity": ["<area where you're similar to competitors>"],
        "potential_disadvantages": [
            {{
                "disadvantage": "<potential weakness>",
                "mitigation": "<how to address>"
            }}
        ],
        "recommended_positioning": "<strategic positioning recommendation>"
    }},
    "timing_analysis": {{
        "first_mover_opportunities": ["<opportunity 1>"],
        "fast_follower_risks": ["<risk 1>"],
        "optimal_launch_window": "<timing recommendation>",
        "competitive_timeline": [
            {{
                "competitor": "<competitor>",
                "expected_milestone": "<what>",
                "expected_timing": "<when>"
            }}
        ]
    }},
    "regulatory_landscape": {{
        "recent_regulatory_actions": ["<action 1>"],
        "breakthrough_designations": <number in similar space>,
        "fast_track_potential": "HIGH|MODERATE|LOW",
        "regulatory_considerations": ["<consideration 1>"]
    }},
    "investment_risk": {{
        "competitive_risk_level": "HIGH|MODERATE|LOW",
        "probability_of_differentiation": <percentage>,
        "market_entry_barriers": ["<barrier 1>", "<barrier 2>"],
        "exit_scenarios": ["<scenario 1>", "<scenario 2>"]
    }},
    "strategic_recommendations": [
        {{
            "priority": <1-5>,
            "recommendation": "<specific recommendation>",
            "rationale": "<why this is important>",
            "implementation": "<how to implement>",
            "expected_impact": "<expected outcome>"
        }}
    ],
    "monitoring_plan": {{
        "key_competitors_to_watch": ["<competitor 1>", "<competitor 2>"],
        "key_milestones_to_track": ["<milestone 1>", "<milestone 2>"],
        "recommended_frequency": "<how often to review>",
        "intelligence_sources": ["<source 1>", "<source 2>"]
    }}
}}

Be specific and strategic. Provide actionable competitive intelligence.
Return ONLY valid JSON, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text.strip())


async def score_trial_dimensions_with_claude(protocol_info: Dict, trials: List[Dict]) -> List[Dict]:
    """
    Use Claude to score each trial on multiple dimensions compared to the protocol.
    Returns trials with dimension_scores added.
    """
    if not trials:
        return trials

    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Build protocol summary
    protocol_summary = f"""
Protocol Condition: {protocol_info.get('condition', 'N/A')}
Protocol Intervention: {protocol_info.get('intervention_name', 'N/A')} ({protocol_info.get('intervention_type', 'N/A')})
Protocol Phase: {protocol_info.get('phase', 'N/A')}
Primary Endpoint: {protocol_info.get('primary_endpoint', 'N/A')}
Target Population: {', '.join(protocol_info.get('inclusion_criteria', [])[:3])}
Target Enrollment: {protocol_info.get('target_enrollment', 'N/A')}
"""

    # Build trials summary for scoring
    trials_for_scoring = []
    for t in trials[:10]:  # Score top 10 trials
        trials_for_scoring.append({
            "nct_id": t.get("nct_id"),
            "conditions": t.get("conditions", "")[:200],
            "interventions": t.get("interventions", "")[:200],
            "phase": t.get("phase", ""),
            "primary_outcomes": t.get("primary_outcomes", "")[:300],
            "eligibility": t.get("eligibility_criteria", "")[:500]
        })

    prompt = f"""You are a clinical trial analyst. Score how similar each trial is to the protocol on 5 dimensions.

PROTOCOL:
{protocol_summary}

TRIALS TO SCORE:
{json.dumps(trials_for_scoring, indent=2)}

For each trial, score these dimensions from 0-100:
- condition: How similar is the disease/condition being studied?
- intervention: How similar is the drug/intervention type and mechanism?
- endpoint: How similar are the primary endpoints/outcome measures?
- population: How similar are the eligibility criteria and target population?
- design: How similar is the trial design (phase, randomization, blinding)?

Return ONLY a JSON array with scores for each trial:
[
    {{"nct_id": "NCT...", "condition": 85, "intervention": 70, "endpoint": 65, "population": 75, "design": 80}},
    ...
]

Be precise and analytical. Higher scores mean more similarity to the protocol."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        # Extract JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        scores = json.loads(text.strip())

        # Merge scores back into trials
        score_map = {s["nct_id"]: s for s in scores}
        for trial in trials:
            if trial["nct_id"] in score_map:
                s = score_map[trial["nct_id"]]
                trial["dimension_scores"] = {
                    "condition": s.get("condition", 50),
                    "intervention": s.get("intervention", 50),
                    "endpoint": s.get("endpoint", 50),
                    "population": s.get("population", 50),
                    "design": s.get("design", 50)
                }

        return trials

    except Exception as e:
        print(f"Claude dimension scoring error: {e}")
        return trials


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

        # Step 2: Search for candidate trials (API first, database backup)
        condition = extracted_dict.get("condition", "")
        intervention = extracted_dict.get("intervention_name", "")
        phase = extracted_dict.get("phase", "")

        print(f"Step 2: Searching for trials matching {condition}...")

        # Use ClinicalTrials.gov API directly (most reliable)
        print("Fetching trials from ClinicalTrials.gov API...")
        candidate_trials = await search_clinicaltrials_api(
            condition=condition,
            intervention=intervention,
            phase=phase,
            max_results=100
        )

        # If API returns no results, try database as backup
        if not candidate_trials:
            print("API returned no results, trying database...")
            candidate_trials = await search_trials_from_database(
                condition=simplify_condition_query(condition) if condition else "",
                intervention=intervention,
                phase=phase,
                max_results=100
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

        # Step 2: Search for candidate trials (database first, API fallback)
        condition = extracted_dict.get("condition", "")
        intervention = extracted_dict.get("intervention_name", "")
        phase = extracted_dict.get("phase", "")

        print(f"Step 2: Searching for '{condition}'...")

        # Use ClinicalTrials.gov API directly (most reliable)
        print("Fetching trials from ClinicalTrials.gov API...")
        candidate_trials = await search_clinicaltrials_api(
            condition=condition,
            intervention=intervention,
            phase=phase,
            max_results=100
        )

        # If API returns no results, try database as backup
        if not candidate_trials:
            print("API returned no results, trying database...")
            candidate_trials = await search_trials_from_database(
                condition=simplify_condition_query(condition) if condition else "",
                intervention=intervention,
                phase=phase,
                max_results=100
            )

        print(f"Found {len(candidate_trials)} candidate trials")

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

        # Step 5: Get detailed enrollment forecast from Claude
        print("Step 5: Generating detailed enrollment forecast...")
        try:
            detailed_enrollment = await analyze_enrollment_forecast_with_claude(extracted_dict, similar_trials)
            print("Enrollment forecast complete")
        except Exception as e:
            print(f"Enrollment forecast error: {e}")
            detailed_enrollment = {}

        # Step 6: Get detailed risk analysis from Claude
        print("Step 6: Generating detailed risk analysis...")
        try:
            detailed_risk = await analyze_detailed_risks_with_claude(extracted_dict, similar_trials)
            print("Risk analysis complete")
        except Exception as e:
            print(f"Risk analysis error: {e}")
            detailed_risk = {}

        # Step 7: Get protocol optimization analysis from Claude
        print("Step 7: Generating protocol optimization analysis...")
        try:
            detailed_optimization = await analyze_protocol_optimization_with_claude(extracted_dict, similar_trials)
            print("Protocol optimization analysis complete")
        except Exception as e:
            print(f"Protocol optimization error: {e}")
            detailed_optimization = {}

        # Step 8: Get eligibility analysis from Claude
        print("Step 8: Generating eligibility analysis...")
        try:
            detailed_eligibility = await analyze_eligibility_with_claude(extracted_dict, similar_trials)
            print("Eligibility analysis complete")
        except Exception as e:
            print(f"Eligibility analysis error: {e}")
            detailed_eligibility = {}

        # Step 9: Get endpoint analysis from Claude
        print("Step 9: Generating endpoint analysis...")
        try:
            detailed_endpoints = await analyze_endpoints_with_claude(extracted_dict, similar_trials)
            print("Endpoint analysis complete")
        except Exception as e:
            print(f"Endpoint analysis error: {e}")
            detailed_endpoints = {}

        # Step 10: Get amendment risk analysis from Claude
        print("Step 10: Generating amendment risk analysis...")
        try:
            detailed_amendment = await analyze_amendment_risk_with_claude(extracted_dict, similar_trials)
            print("Amendment risk analysis complete")
        except Exception as e:
            print(f"Amendment risk analysis error: {e}")
            detailed_amendment = {}

        # Step 11: Get competitive landscape analysis from Claude
        print("Step 11: Generating competitive landscape analysis...")
        try:
            detailed_competitive = await analyze_competitive_landscape_with_claude(extracted_dict, similar_trials)
            print("Competitive landscape analysis complete")
        except Exception as e:
            print(f"Competitive landscape analysis error: {e}")
            detailed_competitive = {}

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

        # Compute enrollment stats from similar trials
        enrollments = [t.get("enrollment", 0) for t in similar_trials if t.get("enrollment")]
        enrollment_stats = {
            "your_target": extracted_dict.get("target_enrollment") or 400,
            "mean": int(sum(enrollments) / len(enrollments)) if enrollments else 0,
            "median": sorted(enrollments)[len(enrollments)//2] if enrollments else 0,
            "min": min(enrollments) if enrollments else 0,
            "max": max(enrollments) if enrollments else 0
        }

        # Aggregate site data from similar trials
        all_countries = {}
        all_sites = {}
        total_sites = 0
        for trial in similar_trials:
            total_sites += trial.get("num_sites", 0)
            for country in trial.get("countries", []):
                all_countries[country] = all_countries.get(country, 0) + 1
            for loc in trial.get("locations", []):
                facility = loc.get("facility", "")
                if facility:
                    key = f"{facility}|{loc.get('city', '')}|{loc.get('country', '')}"
                    if key not in all_sites:
                        all_sites[key] = {"facility": facility, "city": loc.get("city", ""), "country": loc.get("country", ""), "trials": 0}
                    all_sites[key]["trials"] += 1

        top_countries = sorted(all_countries.items(), key=lambda x: x[1], reverse=True)[:10]
        top_sites = sorted(all_sites.values(), key=lambda x: x["trials"], reverse=True)[:10]
        avg_sites_per_trial = int(total_sites / len(similar_trials)) if similar_trials else 0

        # Step 12: Get site intelligence analysis from Claude
        print("Step 12: Generating site intelligence analysis...")
        try:
            site_data_for_claude = {
                "top_countries": [{"country": c[0], "trials": c[1]} for c in top_countries],
                "top_sites": top_sites,
                "avg_sites_per_trial": avg_sites_per_trial
            }
            detailed_sites = await analyze_site_intelligence_with_claude(extracted_dict, similar_trials, site_data_for_claude)
            print("Site intelligence analysis complete")
        except Exception as e:
            print(f"Site intelligence analysis error: {e}")
            detailed_sites = {}

        # Extract detailed risk components
        overall_risk = detailed_risk.get("overall_risk_assessment", {})
        risk_categories = detailed_risk.get("risk_categories", [])
        amendment_risk_detail = detailed_risk.get("amendment_risk", {})
        success_prob = detailed_risk.get("success_probability", {})
        critical_items = detailed_risk.get("critical_watch_items", [])
        risk_priorities = detailed_risk.get("risk_mitigation_priorities", [])

        # Build comprehensive dashboard data matching frontend structure
        dashboard_data = {
            # Risk Analysis - Claude-generated detailed analysis
            "risk_analysis": {
                "overall_score": overall_risk.get("risk_score") or risk.get("overall_risk_score", 50),
                "overall_risk_level": overall_risk.get("risk_level") or risk.get("overall_risk_level", "MEDIUM"),
                "overall_risk_rationale": overall_risk.get("risk_summary") or risk.get("overall_risk_rationale", ""),
                "confidence_level": overall_risk.get("confidence_level", "MEDIUM"),
                "competitive_landscape": {
                    "competing_count": recruiting_count,
                    "risk_level": competition.get("competition_level", "medium").lower()
                },
                # Detailed risk categories from Claude
                "risk_categories": risk_categories,
                "enrollment_risk": {
                    "score": next((c.get("score") for c in risk_categories if c.get("category") == "Enrollment Risk"), risk.get("enrollment_risk_score", 50)),
                    "level": next((c.get("level") for c in risk_categories if c.get("category") == "Enrollment Risk"), "MEDIUM"),
                    "key_factors": next((c.get("key_factors", []) for c in risk_categories if c.get("category") == "Enrollment Risk"), []),
                    "detailed_analysis": next((c.get("detailed_analysis", "") for c in risk_categories if c.get("category") == "Enrollment Risk"), ""),
                    "mitigation_strategies": next((c.get("mitigation_strategies", []) for c in risk_categories if c.get("category") == "Enrollment Risk"), [])
                },
                "timeline_risk": {"score": risk.get("timeline_risk_score", 50)},
                "endpoint_risk": {
                    "score": next((c.get("score") for c in risk_categories if c.get("category") == "Endpoint Risk"), risk.get("endpoint_risk_score", 50)),
                    "level": next((c.get("level") for c in risk_categories if c.get("category") == "Endpoint Risk"), "MEDIUM"),
                    "key_factors": next((c.get("key_factors", []) for c in risk_categories if c.get("category") == "Endpoint Risk"), []),
                    "detailed_analysis": next((c.get("detailed_analysis", "") for c in risk_categories if c.get("category") == "Endpoint Risk"), "")
                },
                "operational_risk": {
                    "score": next((c.get("score") for c in risk_categories if c.get("category") == "Operational Risk"), 50),
                    "level": next((c.get("level") for c in risk_categories if c.get("category") == "Operational Risk"), "MEDIUM"),
                    "key_factors": next((c.get("key_factors", []) for c in risk_categories if c.get("category") == "Operational Risk"), []),
                    "detailed_analysis": next((c.get("detailed_analysis", "") for c in risk_categories if c.get("category") == "Operational Risk"), ""),
                    "mitigation_strategies": next((c.get("mitigation_strategies", []) for c in risk_categories if c.get("category") == "Operational Risk"), [])
                },
                "regulatory_risk": {
                    "score": next((c.get("score") for c in risk_categories if c.get("category") == "Regulatory Risk"), 40),
                    "level": next((c.get("level") for c in risk_categories if c.get("category") == "Regulatory Risk"), "LOW"),
                    "key_factors": next((c.get("key_factors", []) for c in risk_categories if c.get("category") == "Regulatory Risk"), []),
                    "detailed_analysis": next((c.get("detailed_analysis", "") for c in risk_categories if c.get("category") == "Regulatory Risk"), "")
                },
                "competitive_risk": {
                    "score": next((c.get("score") for c in risk_categories if c.get("category") == "Competitive Risk"), 50),
                    "level": next((c.get("level") for c in risk_categories if c.get("category") == "Competitive Risk"), "MEDIUM"),
                    "key_factors": next((c.get("key_factors", []) for c in risk_categories if c.get("category") == "Competitive Risk"), []),
                    "detailed_analysis": next((c.get("detailed_analysis", "") for c in risk_categories if c.get("category") == "Competitive Risk"), "")
                },
                "predictions": {
                    "termination_risk": {"probability": int(terminated_count / len(similar_trials) * 100) if similar_trials else 20},
                    "enrollment_delay": {"probability": risk.get("enrollment_risk_score", 35)},
                    "amendment_required": {"probability": amendment_risk_detail.get("probability_percentage") or amendment.get("amendment_probability", 65)}
                },
                "success_probability": {
                    "overall_success_rate": success_prob.get("overall_success_rate", success.get("predicted_success_rate", 65)),
                    "phase_benchmark": success_prob.get("phase_benchmark", 50),
                    "therapeutic_area_benchmark": success_prob.get("therapeutic_area_benchmark", 55),
                    "factors_increasing_success": success_prob.get("factors_increasing_success", success.get("success_factors", [])),
                    "factors_decreasing_success": success_prob.get("factors_decreasing_success", success.get("risk_factors", [])),
                    "comparison_to_similar": success_prob.get("comparison_to_similar", "")
                },
                "critical_watch_items": critical_items,
                "risk_mitigation_priorities": risk_priorities
            },

            # Amendment Intelligence - Claude-generated detailed analysis
            "amendment_intelligence": {
                "overall_risk_score": detailed_amendment.get("overall_amendment_risk", {}).get("risk_score") or amendment_risk_detail.get("probability_percentage") or amendment.get("amendment_probability", 50),
                "risk_level": detailed_amendment.get("overall_amendment_risk", {}).get("risk_level", "MODERATE"),
                "predicted_amendments": detailed_amendment.get("overall_amendment_risk", {}).get("predicted_amendments") or amendment_risk_detail.get("expected_amendments") or amendment.get("predicted_amendments", 2.0),
                "amendment_probability": detailed_amendment.get("overall_amendment_risk", {}).get("amendment_probability", 65),
                "rationale": detailed_amendment.get("overall_amendment_risk", {}).get("rationale", ""),
                "common_reasons": amendment.get("common_amendment_reasons", []),
                "risk_factors": detailed_amendment.get("risk_factors", []),
                "likely_amendment_areas": detailed_amendment.get("likely_amendment_areas", []) or amendment_risk_detail.get("likely_amendment_areas", []),
                "historical_patterns": detailed_amendment.get("historical_patterns", []),
                "amendment_timeline_prediction": detailed_amendment.get("amendment_timeline_prediction", {}),
                "cost_impact_analysis": detailed_amendment.get("cost_impact_analysis", {}),
                "prevention_action_plan": detailed_amendment.get("prevention_action_plan", []),
                "protocol_quality_indicators": detailed_amendment.get("protocol_quality_indicators", {}),
                "benchmarks": detailed_amendment.get("benchmarks", {}),
                "impact_assessment": amendment_risk_detail.get("amendment_impact_assessment", ""),
                "trials_analyzed": len(similar_trials)
            },

            # Protocol Optimization - Claude-generated detailed analysis
            "protocol_optimization": {
                "complexity_score": {
                    "score": detailed_optimization.get("complexity_assessment", {}).get("overall_score") or complexity.get("complexity_score", 50),
                    "level": detailed_optimization.get("complexity_assessment", {}).get("complexity_level", "MODERATE"),
                    "rationale": detailed_optimization.get("complexity_assessment", {}).get("complexity_rationale", ""),
                    "comparison": "out of 100",
                    "operational_burden": detailed_optimization.get("complexity_assessment", {}).get("operational_burden_estimate", "MODERATE")
                },
                "complexity_factors": detailed_optimization.get("complexity_assessment", {}).get("complexity_factors", []),
                "success_rate": detailed_optimization.get("success_prediction", {}).get("overall_success_probability") or success.get("predicted_success_rate", 65),
                "success_prediction": detailed_optimization.get("success_prediction", {}),
                "trials_analyzed": len(similar_trials),
                "recommendations": [
                    {
                        "priority": r.get("priority", "medium"),
                        "recommendation": r.get("recommendation", ""),
                        "rationale": r.get("rationale", ""),
                        "category": r.get("category", ""),
                        "expected_impact": r.get("expected_impact", ""),
                        "implementation_effort": r.get("implementation_effort", "MEDIUM")
                    }
                    for r in detailed_optimization.get("optimization_recommendations", [])
                ] or formatted_recommendations,
                "strengths": [
                    {
                        "strength": s.get("strength", ""),
                        "impact": s.get("impact", "positive"),
                        "explanation": s.get("explanation", ""),
                        "leverage_recommendation": s.get("leverage_recommendation", "")
                    }
                    for s in detailed_optimization.get("design_strengths", [])
                ] or formatted_strengths,
                "weaknesses": detailed_optimization.get("design_weaknesses", []),
                "amendment_risk": {
                    "probability": amendment.get("amendment_probability", 50),
                    "drivers": [{"factor": r, "impact": "medium"} for r in amendment.get("common_amendment_reasons", [])[:5]]
                },
                "terminated_trials": [{"nct_id": t.get("nct_id"), "title": t.get("title"), "reason": "See ClinicalTrials.gov"}
                                     for t in similar_trials if t.get("status") in ["TERMINATED", "WITHDRAWN"]][:5],
                "terminated_trial_learnings": detailed_optimization.get("terminated_trial_learnings", []),
                "design_comparison": detailed_optimization.get("design_comparison", []),
                "regulatory_design_considerations": detailed_optimization.get("regulatory_design_considerations", {})
            },

            # Enrollment Forecast - Claude-generated detailed analysis
            "enrollment_forecast": {
                "target_enrollment": detailed_enrollment.get("target_enrollment") or extracted_dict.get("target_enrollment") or 120,
                "scenarios": detailed_enrollment.get("scenarios", [
                    {
                        "name": "base",
                        "months": enrollment.get("estimated_duration_months", 24),
                        "probability": 50,
                        "assumptions": [
                            f"{enrollment.get('recommended_sites', 50)} sites activated over 4-6 months",
                            f"Standard enrollment rate for {extracted_dict.get('therapeutic_area', 'this therapeutic area')}",
                            "Standard regulatory approval timeline"
                        ]
                    },
                    {"name": "optimistic", "months": enrollment.get("duration_range_low", 18), "probability": 25},
                    {"name": "conservative", "months": enrollment.get("duration_range_high", 30), "probability": 25}
                ]),
                "historical_benchmark": {
                    "range": f"{detailed_enrollment.get('duration_range_low', enrollment.get('duration_range_low', 18))}-{detailed_enrollment.get('duration_range_high', enrollment.get('duration_range_high', 28))}"
                },
                "scenarios_simulator": detailed_enrollment.get("scenarios_simulator", [
                    {"change": "Add 10 additional sites", "impact_months": -2},
                    {"change": "Expand to 3 more countries", "impact_months": -3}
                ]),
                "bottlenecks": detailed_enrollment.get("bottlenecks", []),
                "enrollment_rate_per_site_month": detailed_enrollment.get("enrollment_rate_per_site_month", enrollment.get("enrollment_rate_per_site_month", 1.5)),
                "recommended_sites": detailed_enrollment.get("recommended_sites", enrollment.get("recommended_sites", 50)),
                "recommended_countries": detailed_enrollment.get("recommended_countries", enrollment.get("recommended_countries", 12)),
                "patients_per_site": detailed_enrollment.get("patients_per_site", enrollment.get("patients_per_site", 15)),
                "screen_failure_prediction": detailed_enrollment.get("screen_failure_prediction", 30),
                "site_activation_timeline_weeks": detailed_enrollment.get("site_activation_timeline_weeks", 12),
                "key_enrollment_risks": detailed_enrollment.get("key_enrollment_risks", []),
                "enrollment_acceleration_strategies": detailed_enrollment.get("enrollment_acceleration_strategies", [])
            },

            # Site Intelligence - Claude-generated detailed analysis
            "site_intelligence": {
                "strategy": detailed_sites.get("site_strategy", {
                    "recommended_total_sites": enrollment.get("recommended_sites", 50),
                    "recommended_countries": len(all_countries) if all_countries else enrollment.get("recommended_countries", 12),
                    "patients_per_site_target": enrollment.get("patients_per_site", 15),
                    "site_activation_timeline_weeks": 12,
                    "strategy_rationale": "",
                    "key_considerations": []
                }),
                "geographic_strategy": detailed_sites.get("geographic_strategy", {}),
                "country_recommendations": detailed_sites.get("country_recommendations", []),
                "site_selection_criteria": detailed_sites.get("site_selection_criteria", {}),
                "site_performance_benchmarks": detailed_sites.get("site_performance_benchmarks", {}),
                "competitive_site_analysis": detailed_sites.get("competitive_site_analysis", {}),
                "risk_mitigation": detailed_sites.get("risk_mitigation", {}),
                "timeline_optimization": detailed_sites.get("timeline_optimization", {}),
                "budget_considerations": detailed_sites.get("budget_considerations", {}),
                "special_considerations": detailed_sites.get("special_considerations", {}),
                "top_sites": top_sites,
                "top_countries": [{"country": c[0], "trials": c[1]} for c in top_countries],
                "regional_distribution": {c[0]: c[1] for c in top_countries},
                "benchmark_avg": str(avg_sites_per_trial) if avg_sites_per_trial else "-"
            },

            # Competitive Landscape - Claude-generated detailed analysis
            "competitive_landscape": {
                "total_similar_trials": len(similar_trials),
                "completed": completed_count,
                "recruiting": recruiting_count,
                "terminated": terminated_count,
                "overall_assessment": detailed_competitive.get("overall_assessment", {
                    "competition_level": competition.get("competition_level", "MEDIUM"),
                    "competition_score": 50,
                    "executive_summary": "",
                    "key_implications": []
                }),
                "competition_level": detailed_competitive.get("overall_assessment", {}).get("competition_level") or competition.get("competition_level", "MEDIUM"),
                "competition_score": detailed_competitive.get("overall_assessment", {}).get("competition_score", 50),
                "competitor_analysis": detailed_competitive.get("competitor_analysis", []),
                "market_dynamics": detailed_competitive.get("market_dynamics", {}),
                "enrollment_competition": detailed_competitive.get("enrollment_competition", {}),
                "differentiation_analysis": detailed_competitive.get("differentiation_analysis", {}),
                "timing_analysis": detailed_competitive.get("timing_analysis", {}),
                "regulatory_landscape": detailed_competitive.get("regulatory_landscape", {}),
                "investment_risk": detailed_competitive.get("investment_risk", {}),
                "strategic_recommendations": detailed_competitive.get("strategic_recommendations", []),
                "monitoring_plan": detailed_competitive.get("monitoring_plan", {}),
                "top_sponsors": [{"name": s, "trial_count": 1} for s in competition.get("key_competitors", [])[:6]]
            },

            # Similar trials enhanced
            "similar_trials_enhanced": {
                "detected_therapeutic_area": extracted_dict.get("therapeutic_area", "General"),
                "total_matched": len(similar_trials),
                "showing": len(similar_trials),
                "trials": similar_trials
            },

            # Eligibility analysis - Claude-generated detailed analysis
            "eligibility_analysis": {
                "screen_failure_prediction": {
                    "assessment": detailed_eligibility.get("screen_failure_prediction", {}).get("assessment", "moderate"),
                    "rate": detailed_eligibility.get("screen_failure_prediction", {}).get("predicted_rate", 25),
                    "rate_range": detailed_eligibility.get("screen_failure_prediction", {}).get("rate_range", {}),
                    "predicted_ratio": detailed_eligibility.get("screen_failure_prediction", {}).get("screen_to_enroll_ratio", "2.5:1"),
                    "benchmark_ratio": detailed_eligibility.get("screen_failure_prediction", {}).get("benchmark_ratio", "2.5:1"),
                    "best_in_class_ratio": detailed_eligibility.get("screen_failure_prediction", {}).get("best_in_class_ratio", "2.0:1"),
                    "your_exclusions": len(extracted_dict.get("exclusion_criteria", [])),
                    "rationale": detailed_eligibility.get("screen_failure_prediction", {}).get("rationale", ""),
                    "primary_failure_drivers": detailed_eligibility.get("screen_failure_prediction", {}).get("primary_failure_drivers", [])
                },
                "criteria_analysis": detailed_eligibility.get("criteria_analysis", {}),
                "criterion_benchmark": detailed_eligibility.get("criterion_benchmarks", []),
                "patient_pool_estimation": detailed_eligibility.get("patient_pool_estimation", {
                    "stages": [],
                    "global_estimate": enrollment_stats.get("mean", 0) * 10
                }),
                "optimization_suggestions": detailed_eligibility.get("optimization_recommendations", []),
                "competitive_differentiation": detailed_eligibility.get("competitive_differentiation", {}),
                "special_populations": detailed_eligibility.get("special_populations", {}),
                "regulatory_alignment": detailed_eligibility.get("regulatory_alignment", {})
            },

            # Endpoint Intelligence - Claude-generated detailed analysis
            "endpoint_intelligence": {
                "primary_endpoint": extracted_dict.get("primary_endpoint", ""),
                "primary_endpoint_analysis": detailed_endpoints.get("primary_endpoint_analysis", {}),
                "fda_alignment": detailed_endpoints.get("fda_alignment", {"status": "moderate"}),
                "ema_alignment": detailed_endpoints.get("ema_alignment", {}),
                "endpoint_benchmarking": detailed_endpoints.get("endpoint_benchmarking", {}),
                "primary_endpoint_distribution": detailed_endpoints.get("endpoint_benchmarking", {}).get("primary_endpoint_distribution", []),
                "historical_benchmarks": detailed_endpoints.get("historical_benchmarks", {}).get("similar_trials_effect_sizes", [
                    {"nct_id": t.get("nct_id"), "endpoint": t.get("primary_outcomes", "")[:100], "result": "See trial"}
                    for t in similar_trials[:5] if t.get("primary_outcomes")
                ]),
                "enrollment_stats": enrollment_stats,
                "sample_size_analysis": detailed_endpoints.get("sample_size_analysis", {}),
                "sample_size_scenarios": detailed_endpoints.get("sample_size_analysis", {}).get("scenarios", [
                    {"scenario": "Conservative", "control_rate": "25%", "treatment_rate": "40%", "effect_size": "15%", "patients_needed": 350, "power": "80%"},
                    {"scenario": "Base Case", "control_rate": "25%", "treatment_rate": "45%", "effect_size": "20%", "patients_needed": 200, "power": "80%"},
                    {"scenario": "Optimistic", "control_rate": "25%", "treatment_rate": "50%", "effect_size": "25%", "patients_needed": 130, "power": "80%"}
                ]),
                "sample_size_assumptions": detailed_endpoints.get("sample_size_analysis", {}).get("power_assumptions", {
                    "alpha": "0.05 (two-sided)",
                    "power": "80%",
                    "dropout_rate": "15-20%",
                    "analysis_method": "ITT"
                }),
                "dropout_adjustment": detailed_endpoints.get("sample_size_analysis", {}).get("dropout_adjustment", {}),
                "sample_size_risks": detailed_endpoints.get("sample_size_analysis", {}).get("sample_size_risks", []),
                "sample_size_insights": {
                    "comparison": f"Your target ({enrollment_stats.get('your_target', 0)}) vs similar trial mean ({enrollment_stats.get('mean', 0)})",
                    "recommendation": detailed_endpoints.get("sample_size_analysis", {}).get("dropout_adjustment", {}).get("expected_dropout_rate", "Consider planning for 15-20% over-enrollment to account for dropouts."),
                    "interim_analysis": "Plan interim analysis at 50% enrollment for futility/efficacy assessment."
                },
                "secondary_endpoints_analysis": detailed_endpoints.get("secondary_endpoints_analysis", []),
                "missing_endpoints_recommendations": detailed_endpoints.get("missing_endpoints_recommendations", []),
                "endpoint_collection_burden": detailed_endpoints.get("endpoint_collection_burden", {})
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


# ============== AUTHENTICATION ==============
try:
    from auth import (
        UserCreate, UserLogin, create_user, authenticate_user,
        create_session, validate_session, logout_session, get_user_by_email
    )
except ImportError:
    from web_app.auth import (
        UserCreate, UserLogin, create_user, authenticate_user,
        create_session, validate_session, logout_session, get_user_by_email
    )


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    organization: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """Register a new user with email as username."""
    try:
        # Validate email format
        if not request.email or "@" not in request.email:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid email format"}
            )

        # Validate password
        if not request.password or len(request.password) < 6:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Password must be at least 6 characters"}
            )

        # Create user
        user_data = UserCreate(
            email=request.email,
            password=request.password,
            full_name=request.full_name,
            organization=request.organization
        )
        user = create_user(user_data)

        # Create session
        token = create_session(user["id"], user["email"])

        return {
            "success": True,
            "message": "Registration successful",
            "token": token,
            "user": user
        }

    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """Login with email and password."""
    try:
        user = authenticate_user(request.email, request.password)

        if not user:
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "Invalid email or password"}
            )

        # Create session
        token = create_session(user["id"], user["email"])

        return {
            "success": True,
            "message": "Login successful",
            "token": token,
            "user": user
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/auth/logout")
async def logout(request: Request):
    """Logout and invalidate session."""
    try:
        # Get token from header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            logout_session(token)

        return {"success": True, "message": "Logged out successfully"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/auth/me")
async def get_current_user(request: Request):
    """Get current user from session token."""
    try:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "Not authenticated"}
            )

        token = auth_header[7:]
        user = validate_session(token)

        if not user:
            return JSONResponse(
                status_code=401,
                content={"success": False, "error": "Invalid or expired session"}
            )

        return {"success": True, "user": user}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/login")
async def login_page(request: Request):
    """Render login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register")
async def register_page(request: Request):
    """Render registration page."""
    return templates.TemplateResponse("register.html", {"request": request})


# ============== RUN ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
