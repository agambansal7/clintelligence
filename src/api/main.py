"""
TrialIntel API Server

This FastAPI server exposes TrialIntel capabilities as REST APIs that can be
integrated into Jeeva's platform.

Endpoints:
- POST /api/v1/protocol/risk-score - Score a protocol for risks
- GET /api/v1/sites/recommend - Get site recommendations
- GET /api/v1/investigators/search - Find investigators
- GET /api/v1/endpoints/analyze - Analyze endpoints for an indication
- GET /api/v1/competitive/monitor - Monitor competitive trials

Integration with Jeeva:
These APIs are designed to plug into Jeeva's protocol configuration workflow.
When a sponsor sets up a new trial in Jeeva:
1. Protocol Risk Score is called during protocol entry
2. Site Recommendations are shown during site selection
3. Endpoint Analysis helps with outcome selection
"""

import hashlib
import hmac
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

# Anthropic API for AI-powered assistant
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.endpoint_benchmarking import EndpointAnalysis, EndpointBenchmarker
from analysis.protocol_risk_scorer import ProtocolRiskScorer, RiskAssessment
from analysis.site_intelligence import SiteInvestigatorIntelligence, SiteRecommendation
from config import get_settings

# Get settings
settings = get_settings()

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level.upper()),
    format=settings.logging.format,
)
logger = logging.getLogger(__name__)

# Global database manager and analysis modules
db_manager = None
risk_scorer = None
site_intel = None
endpoint_benchmarker = None


# ============================================================
# INPUT VALIDATION HELPERS
# ============================================================

def sanitize_text_input(text: str, max_length: int = 50000) -> str:
    """Sanitize text input to prevent injection attacks."""
    if not text:
        return ""
    # Truncate if too long
    text = text[:max_length]
    # Remove null bytes
    text = text.replace("\x00", "")
    return text


def validate_nct_id(nct_id: str) -> bool:
    """Validate NCT ID format."""
    return bool(re.match(r"^NCT\d{8}$", nct_id))


def validate_phase(phase: str) -> bool:
    """Validate phase value."""
    valid_phases = {"PHASE1", "PHASE2", "PHASE3", "PHASE4", "EARLY_PHASE1", "NA"}
    return phase.upper() in valid_phases


# ============================================================
# DATABASE SETUP
# ============================================================

def get_db() -> Session:
    """Get database session dependency."""
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


def init_analysis_modules():
    """Initialize analysis modules with database connection."""
    global risk_scorer, site_intel, endpoint_benchmarker, db_manager

    try:
        from src.database import DatabaseManager

        # Initialize database
        db_manager = DatabaseManager.get_instance()
        db_manager.create_tables()

        logger.info(f"Database initialized: {db_manager.database_url}")

        # Get database stats
        stats = db_manager.get_stats()
        logger.info(f"Database contains {stats.get('total_trials', 0)} trials")

        # Initialize analysis modules with database sessions
        session = db_manager.get_session()

        risk_scorer = ProtocolRiskScorer(db_session=session)
        logger.info("Protocol Risk Scorer initialized with database")

        site_intel = SiteInvestigatorIntelligence(db_session=session)
        site_intel.load_from_database()
        logger.info(f"Site Intelligence loaded {len(site_intel.sites)} sites")

        endpoint_benchmarker = EndpointBenchmarker(db_session=session)
        endpoint_benchmarker.load_from_database()
        logger.info("Endpoint Benchmarker initialized with database")

    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        logger.info("Falling back to default analysis modules")

        # Fall back to modules without database
        risk_scorer = ProtocolRiskScorer()
        site_intel = SiteInvestigatorIntelligence()
        endpoint_benchmarker = EndpointBenchmarker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    logger.info("Starting TrialIntel API...")
    logger.info(f"Environment: {settings.environment}")
    init_analysis_modules()
    logger.info("TrialIntel API started successfully")
    yield
    # Shutdown
    logger.info("Shutting down TrialIntel API...")
    if db_manager is not None:
        from src.database import DatabaseManager
        DatabaseManager.reset_instance()
    logger.info("TrialIntel API shutdown complete")


# API Tags for documentation organization
tags_metadata = [
    {
        "name": "AI Assistant",
        "description": "Natural language interface powered by Claude AI for clinical trial intelligence queries.",
    },
    {
        "name": "Protocol Analysis",
        "description": "Analyze draft protocols against historical trial data to identify risks and optimization opportunities.",
    },
    {
        "name": "Site Intelligence",
        "description": "Find and recommend optimal clinical trial sites based on historical performance data.",
    },
    {
        "name": "Investigator Search",
        "description": "Search for investigators with experience in specific therapeutic areas.",
    },
    {
        "name": "Endpoint Benchmarking",
        "description": "Analyze endpoint usage patterns and get recommendations for optimal endpoint selection.",
    },
    {
        "name": "Competitive Intelligence",
        "description": "Monitor competitive landscape and track competitor trials.",
    },
    {
        "name": "Webhooks",
        "description": "Integration webhooks for external platforms like Jeeva.",
    },
    {
        "name": "System",
        "description": "Health checks, statistics, and system status endpoints.",
    },
]

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="TrialIntel API",
    description="""
# Clinical Trial Intelligence API

Powered by comprehensive ClinicalTrials.gov data with 100,000+ trials across 113 therapeutic areas.

## Key Capabilities

### Protocol Risk Scoring
Analyze draft protocols against historical trial data to identify:
- Eligibility criteria that frequently lead to amendments
- Enrollment targets that may be unrealistic
- Endpoints with historically problematic outcomes
- Predicted amendment, delay, and termination probabilities

### Site & Investigator Intelligence
Get data-driven recommendations for:
- Top-performing sites by therapeutic area
- Experienced investigators with high completion rates
- Geographic diversity optimization
- Projected enrollment timelines

### Endpoint Benchmarking
Analyze endpoint usage across completed trials:
- Most common primary/secondary endpoints by indication
- Typical timeframes and success rates
- Regulatory insights and recommendations

### Competitive Intelligence
Monitor the competitive landscape:
- Active trials in your therapeutic area
- Competitor enrollment patterns
- Site overlap analysis

## Authentication

When API key authentication is enabled, include your API key in the `X-API-Key` header.

## Rate Limiting

Default: 100 requests per 60 seconds per IP address.
Rate limit headers are included in all responses.

## Data Sources

- **ClinicalTrials.gov**: 100,000+ trials with full protocol details
- **Site Performance**: 120,000+ sites with historical metrics
- **Endpoint Patterns**: Normalized endpoint database with success indicators
    """,
    version="1.0.0",
    contact={
        "name": "TrialIntel Support",
        "email": "support@trialintel.com",
    },
    license_info={
        "name": "Proprietary",
    },
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# Setup middleware
from api.middleware import setup_middleware
setup_middleware(app, settings)


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class ProtocolRiskRequest(BaseModel):
    """Request model for protocol risk scoring."""
    condition: str = Field(..., min_length=1, max_length=200, description="Primary indication (e.g., 'Type 2 Diabetes')")
    phase: str = Field(default="PHASE3", description="Trial phase (PHASE1, PHASE2, PHASE3, PHASE4)")
    eligibility_criteria: str = Field(..., min_length=10, max_length=50000, description="Full text of inclusion/exclusion criteria")
    primary_endpoints: List[str] = Field(default=[], max_length=20, description="List of primary endpoints")
    target_enrollment: int = Field(..., ge=1, le=1000000, description="Target number of patients")
    planned_sites: int = Field(..., ge=1, le=10000, description="Number of planned sites")
    planned_duration_months: int = Field(default=24, ge=1, le=240, description="Planned trial duration")
    min_age: int = Field(default=18, ge=0, le=120, description="Minimum age requirement")
    max_age: int = Field(default=99, ge=0, le=120, description="Maximum age requirement")

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, v: str) -> str:
        valid_phases = {"PHASE1", "PHASE2", "PHASE3", "PHASE4", "EARLY_PHASE1", "NA"}
        v_upper = v.upper()
        if v_upper not in valid_phases:
            raise ValueError(f"Invalid phase. Must be one of: {valid_phases}")
        return v_upper

    @field_validator("condition")
    @classmethod
    def sanitize_condition(cls, v: str) -> str:
        return sanitize_text_input(v, max_length=200)

    @field_validator("eligibility_criteria")
    @classmethod
    def sanitize_criteria(cls, v: str) -> str:
        return sanitize_text_input(v, max_length=50000)

    class Config:
        json_schema_extra = {
            "example": {
                "condition": "Type 2 Diabetes",
                "phase": "PHASE3",
                "eligibility_criteria": """
                Inclusion Criteria:
                - Male or female, age 18-65 years
                - Diagnosed with type 2 diabetes ≥180 days prior
                - HbA1c between 7.5% and 10.0%

                Exclusion Criteria:
                - History of pancreatitis
                - eGFR < 60 mL/min/1.73m²
                """,
                "primary_endpoints": ["Change in HbA1c from baseline at week 52"],
                "target_enrollment": 2000,
                "planned_sites": 150,
                "planned_duration_months": 18,
            }
        }


class RiskScoreResponse(BaseModel):
    """Response model for protocol risk scoring."""
    overall_risk_score: float = Field(..., description="Risk score 0-100 (lower is better)")
    risk_level: str = Field(..., description="Risk level: low, medium, high")
    amendment_probability: float = Field(..., description="Probability of protocol amendment (0-1)")
    enrollment_delay_probability: float = Field(..., description="Probability of enrollment delays (0-1)")
    termination_probability: float = Field(..., description="Probability of early termination (0-1)")
    risk_factors: List[Dict[str, Any]] = Field(..., description="Identified risk factors")
    recommendations: List[str] = Field(..., description="Prioritized recommendations")
    benchmark_trials: List[str] = Field(..., description="Similar trials for reference")


class SiteRecommendationRequest(BaseModel):
    """Request model for site recommendations."""
    therapeutic_area: str = Field(..., min_length=1, max_length=200, description="Therapeutic area (e.g., 'diabetes', 'oncology')")
    target_enrollment: int = Field(..., ge=1, le=1000000, description="Target enrollment number")
    num_sites: int = Field(default=10, ge=1, le=100, description="Number of sites to recommend")
    country: Optional[str] = Field(default=None, max_length=100, description="Limit to specific country")
    prioritize_diversity: bool = Field(default=True, description="Boost diverse sites in rankings")

    @field_validator("therapeutic_area")
    @classmethod
    def sanitize_therapeutic_area(cls, v: str) -> str:
        return sanitize_text_input(v, max_length=200)


class SiteResponse(BaseModel):
    """Response model for a single site recommendation."""
    facility_name: str
    city: str
    state: Optional[str]
    country: str
    match_score: float
    reasons: List[str]
    projected_enrollment: float
    projected_timeline_months: float
    therapeutic_experience: int
    diversity_score: float


class InvestigatorResponse(BaseModel):
    """Response model for investigator search."""
    name: str
    affiliations: List[str]
    therapeutic_areas: List[str]
    total_trials: int
    completed_trials: int
    completion_rate: float
    experience_score: float


class EndpointAnalysisResponse(BaseModel):
    """Response model for endpoint analysis."""
    condition: str
    total_trials_analyzed: int
    primary_endpoints: List[Dict[str, Any]]
    secondary_endpoints: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    regulatory_insights: List[str]


class CompetitiveTrialResponse(BaseModel):
    """Response model for competitive trial monitoring."""
    nct_id: str
    title: str
    sponsor: str
    status: str
    phase: List[str]
    start_date: Optional[str]
    enrollment: Optional[int]
    primary_endpoints: List[str]


class AIAssistantRequest(BaseModel):
    """Request model for AI assistant queries."""
    query: str = Field(..., min_length=5, max_length=2000, description="Natural language query about clinical trials")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context (e.g., specific condition, phase, or protocol details)"
    )
    include_data: bool = Field(
        default=True,
        description="Whether to include relevant database data in the response"
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return sanitize_text_input(v, max_length=2000)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the typical enrollment rates for Phase 3 diabetes trials?",
                "context": {"condition": "diabetes", "phase": "PHASE3"},
                "include_data": True,
            }
        }


class AIAssistantResponse(BaseModel):
    """Response model for AI assistant."""
    answer: str = Field(..., description="AI-generated answer to the query")
    confidence: str = Field(..., description="Confidence level: high, medium, low")
    data_used: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Database data that informed the response"
    )
    suggestions: List[str] = Field(
        default=[],
        description="Follow-up questions or related queries"
    )
    sources: List[str] = Field(
        default=[],
        description="NCT IDs or data sources used"
    )


# ============================================================
# WEBHOOK MODELS
# ============================================================

class JeevaWebhookPayload(BaseModel):
    """Payload for Jeeva webhook events."""
    event_type: str = Field(..., description="Type of event (protocol_created, site_selection_started, etc.)")
    trial_id: str = Field(..., min_length=1, max_length=100, description="Unique trial identifier")
    data: Dict[str, Any] = Field(..., description="Event-specific data")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Event timestamp")

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        valid_events = {
            "protocol_created",
            "protocol_updated",
            "site_selection_started",
            "enrollment_updated",
            "trial_completed",
        }
        if v not in valid_events:
            raise ValueError(f"Invalid event type. Must be one of: {valid_events}")
        return v


class WebhookResponse(BaseModel):
    """Response model for webhook acknowledgment."""
    status: str = Field(..., description="accepted, processing, ignored, or error")
    message: str = Field(..., description="Status message")
    task_id: Optional[str] = Field(default=None, description="Background task ID if applicable")


class WebhookCallbackPayload(BaseModel):
    """Payload for sending results back to Jeeva."""
    event_type: str
    trial_id: str
    results: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """API root - health check and info."""
    return {
        "service": "TrialIntel API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "environment": settings.environment,
    }


@app.post("/api/v1/assistant/query", response_model=AIAssistantResponse, tags=["AI Assistant"])
async def ai_assistant_query(request: AIAssistantRequest):
    """
    Natural language interface for clinical trial intelligence.

    Ask questions in plain English about:
    - Trial design and protocol optimization
    - Historical enrollment rates and timelines
    - Site and investigator recommendations
    - Endpoint selection guidance
    - Competitive landscape analysis
    - Risk factors and mitigation strategies

    The AI assistant uses Claude to analyze your query against our database
    of 100,000+ clinical trials.

    **Examples:**
    - "What are typical enrollment rates for Phase 3 diabetes trials?"
    - "Which sites have the best track record for oncology trials?"
    - "What endpoints are commonly used in Alzheimer's trials?"
    - "How can I reduce amendment risk for a cardiovascular trial?"
    """
    if not ANTHROPIC_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI Assistant requires the anthropic package. Install with: pip install anthropic"
        )

    api_key = settings.ai.anthropic_api_key
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY environment variable not set"
        )

    try:
        # Gather relevant data based on query context
        data_context = {}
        sources = []

        if request.include_data and db_manager is not None:
            # Get database stats for context
            stats = db_manager.get_stats()
            data_context["database_stats"] = {
                "total_trials": stats.get("total_trials", 0),
                "total_sites": stats.get("total_sites", 0),
                "total_endpoints": stats.get("total_endpoints", 0),
            }

            # If context includes a condition, get relevant stats
            if request.context and request.context.get("condition"):
                condition = request.context["condition"]
                try:
                    # Get historical stats for the condition
                    from src.database import TrialRepository
                    repo = TrialRepository(db_manager.get_session())
                    historical = repo.get_historical_stats(condition)
                    if historical:
                        data_context["condition_stats"] = historical

                    # Get terminated trials for risk context
                    terminated = repo.get_terminated_trials_with_reasons(condition, limit=5)
                    if terminated:
                        data_context["terminated_examples"] = [
                            {"nct_id": t.nct_id, "reason": t.why_stopped}
                            for t in terminated if t.why_stopped
                        ]
                        sources.extend([t.nct_id for t in terminated])
                except Exception as e:
                    logger.warning(f"Could not fetch condition data: {e}")

        # Build system prompt with clinical trial expertise
        system_prompt = """You are TrialIntel AI, an expert clinical trial intelligence assistant.
You help pharmaceutical companies, CROs, and research institutions optimize their clinical trials.

Your expertise includes:
- Protocol design and risk assessment
- Site selection and investigator identification
- Enrollment optimization strategies
- Endpoint selection and benchmarking
- Regulatory considerations
- Competitive landscape analysis

When answering questions:
1. Be specific and data-driven when possible
2. Reference historical trial patterns
3. Provide actionable recommendations
4. Highlight potential risks and mitigation strategies
5. Suggest follow-up considerations

Database Context:
""" + str(data_context) if data_context else ""

        # Build the user message
        user_message = request.query
        if request.context:
            user_message += f"\n\nContext: {request.context}"

        # Call Claude API
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model=settings.ai.model,
            max_tokens=settings.ai.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        # Extract the response
        answer = message.content[0].text if message.content else "Unable to generate response."

        # Determine confidence based on data availability
        if data_context and data_context.get("condition_stats"):
            confidence = "high"
        elif data_context:
            confidence = "medium"
        else:
            confidence = "low"

        # Generate follow-up suggestions
        suggestions = [
            "Would you like site recommendations for this indication?",
            "Should I analyze the competitive landscape?",
            "Would you like endpoint benchmarking data?",
        ]

        return AIAssistantResponse(
            answer=answer,
            confidence=confidence,
            data_used=data_context if data_context else None,
            suggestions=suggestions[:3],
            sources=sources[:10],
        )

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")
    except Exception as e:
        logger.error(f"AI Assistant error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI query failed. Please try again.")


@app.post("/api/v1/protocol/risk-score", response_model=RiskScoreResponse, tags=["Protocol Analysis"])
async def score_protocol(request: ProtocolRiskRequest):
    """
    Score a protocol for risks based on historical trial data.

    This endpoint analyzes draft protocols against 500,000+ historical trials
    to identify:
    - Eligibility criteria that frequently lead to amendments
    - Enrollment targets that may be unrealistic
    - Endpoints that have historically been problematic

    Use this during protocol development to catch issues early.
    """
    try:
        assessment = risk_scorer.score_protocol(
            condition=request.condition,
            phase=request.phase,
            eligibility_criteria=request.eligibility_criteria,
            primary_endpoints=request.primary_endpoints,
            target_enrollment=request.target_enrollment,
            planned_sites=request.planned_sites,
            planned_duration_months=request.planned_duration_months,
            age_range=(request.min_age, request.max_age),
        )

        # Determine risk level
        if assessment.overall_risk_score < 30:
            risk_level = "low"
        elif assessment.overall_risk_score < 60:
            risk_level = "medium"
        else:
            risk_level = "high"

        return RiskScoreResponse(
            overall_risk_score=assessment.overall_risk_score,
            risk_level=risk_level,
            amendment_probability=assessment.amendment_probability,
            enrollment_delay_probability=assessment.enrollment_delay_probability,
            termination_probability=assessment.termination_probability,
            risk_factors=[
                {
                    "category": rf.category,
                    "description": rf.description,
                    "severity": rf.severity,
                    "evidence": rf.historical_evidence,
                    "recommendation": rf.recommendation,
                }
                for rf in assessment.risk_factors
            ],
            recommendations=assessment.recommendations,
            benchmark_trials=assessment.benchmark_trials,
        )

    except ValueError as e:
        logger.warning(f"Invalid protocol risk request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Risk scoring failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Risk scoring failed. Please check your input and try again.")


@app.post("/api/v1/sites/recommend", response_model=List[SiteResponse], tags=["Site Intelligence"])
async def recommend_sites(request: SiteRecommendationRequest):
    """
    Get site recommendations for a trial.

    Analyzes historical site performance to recommend optimal sites based on:
    - Experience in the therapeutic area
    - Historical enrollment rates
    - Trial completion rates
    - Geographic diversity access

    Returns sites ranked by match score with projected enrollment.
    """
    try:
        recommendations = site_intel.recommend_sites(
            therapeutic_area=request.therapeutic_area,
            target_enrollment=request.target_enrollment,
            num_sites=request.num_sites,
            country_filter=request.country,
            prioritize_diversity=request.prioritize_diversity,
        )

        return [
            SiteResponse(
                facility_name=rec.site.facility_name,
                city=rec.site.city,
                state=rec.site.state,
                country=rec.site.country,
                match_score=rec.match_score,
                reasons=rec.reasons,
                projected_enrollment=rec.projected_enrollment,
                projected_timeline_months=rec.projected_timeline_months,
                therapeutic_experience=rec.site.total_trials,
                diversity_score=rec.site.diversity_score,
            )
            for rec in recommendations
        ]

    except ValueError as e:
        logger.warning(f"Invalid site recommendation request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Site recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Site recommendation failed. Please try again.")


@app.get("/api/v1/investigators/search", response_model=List[InvestigatorResponse], tags=["Investigator Search"])
async def search_investigators(
    therapeutic_area: str = Query(..., min_length=1, max_length=200, description="Therapeutic area to search"),
    min_trials: int = Query(default=3, ge=1, le=1000, description="Minimum trials to qualify"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum results"),
):
    """
    Search for investigators with experience in a therapeutic area.

    Returns investigators ranked by:
    - Number of trials in the indication
    - Trial completion rate
    - Overall experience
    """
    try:
        investigators = site_intel.find_investigators(
            therapeutic_area=sanitize_text_input(therapeutic_area, 200),
            min_trials=min_trials,
            top_n=limit,
        )

        return [
            InvestigatorResponse(
                name=inv.name,
                affiliations=inv.affiliations,
                therapeutic_areas=inv.therapeutic_areas,
                total_trials=inv.total_trials,
                completed_trials=inv.completed_trials,
                completion_rate=inv.completion_rate,
                experience_score=inv.experience_score,
            )
            for inv in investigators
        ]

    except ValueError as e:
        logger.warning(f"Invalid investigator search request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Investigator search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Investigator search failed. Please try again.")


@app.get("/api/v1/endpoints/analyze", response_model=EndpointAnalysisResponse, tags=["Endpoint Benchmarking"])
async def analyze_endpoints(
    condition: str = Query(..., min_length=1, max_length=200, description="Condition to analyze (e.g., 'diabetes', 'breast_cancer')"),
    phase: Optional[str] = Query(default=None, description="Filter by phase (e.g., 'PHASE3')"),
):
    """
    Analyze endpoints used in trials for a specific condition.

    Returns:
    - Most common primary and secondary endpoints
    - Typical timeframes
    - Regulatory insights
    - Endpoint recommendations with confidence levels
    """
    try:
        # Validate phase if provided
        if phase and not validate_phase(phase):
            raise HTTPException(status_code=400, detail="Invalid phase value")

        phase_filter = [phase.upper()] if phase else None
        analysis = endpoint_benchmarker.analyze_condition(
            condition=sanitize_text_input(condition, 200),
            phase_filter=phase_filter,
        )

        return EndpointAnalysisResponse(
            condition=analysis.condition,
            total_trials_analyzed=analysis.total_trials_analyzed,
            primary_endpoints=[
                {
                    "measure": ep.measure,
                    "frequency": ep.frequency,
                    "typical_timeframes": ep.typical_timeframes,
                    "success_rate": (
                        ep.success_indicators["completed"] /
                        max(sum(ep.success_indicators.values()), 1)
                    ),
                }
                for ep in analysis.primary_endpoints
            ],
            secondary_endpoints=[
                {
                    "measure": ep.measure,
                    "frequency": ep.frequency,
                    "typical_timeframes": ep.typical_timeframes,
                }
                for ep in analysis.secondary_endpoints
            ],
            recommendations=[
                {
                    "endpoint": rec.endpoint,
                    "confidence": rec.confidence,
                    "rationale": rec.rationale,
                    "typical_timeframe": rec.typical_timeframe,
                    "considerations": rec.considerations,
                }
                for rec in analysis.recommendations
            ],
            regulatory_insights=analysis.regulatory_insights,
        )

    except ValueError as e:
        logger.warning(f"Invalid endpoint analysis request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Endpoint analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Endpoint analysis failed. Please try again.")


@app.get("/api/v1/competitive/landscape", tags=["Competitive Intelligence"])
async def get_competitive_landscape(
    therapeutic_area: str = Query(..., min_length=1, max_length=200, description="Therapeutic area to analyze"),
):
    """
    Get competitive landscape overview for a therapeutic area.

    Returns:
    - Total sites and investigators in the space
    - Top countries by site count
    - Average experience metrics
    """
    try:
        landscape = site_intel.get_competitive_landscape(
            sanitize_text_input(therapeutic_area, 200)
        )
        return landscape

    except ValueError as e:
        logger.warning(f"Invalid competitive landscape request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Landscape analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Landscape analysis failed. Please try again.")


# ============================================================
# SIMILAR TRIALS & PROTOCOL COMPARISON
# ============================================================

class SimilarTrialsRequest(BaseModel):
    """Request model for finding similar trials."""
    condition: str = Field(..., min_length=1, max_length=200, description="Target condition/indication")
    phase: Optional[str] = Field(default=None, description="Trial phase")
    eligibility_criteria: Optional[str] = Field(default=None, max_length=50000, description="Eligibility criteria text")
    primary_endpoint: Optional[str] = Field(default=None, max_length=500, description="Primary endpoint")
    enrollment_target: Optional[int] = Field(default=None, ge=1, le=1000000, description="Target enrollment")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")

    @field_validator("condition")
    @classmethod
    def sanitize_condition(cls, v: str) -> str:
        return sanitize_text_input(v, max_length=200)


class SimilarTrialResponse(BaseModel):
    """Response model for a similar trial."""
    nct_id: str
    title: str
    similarity_score: float
    status: str
    phase: str
    sponsor: str
    enrollment: int
    duration_months: Optional[float]
    num_sites: int
    why_stopped: Optional[str]
    primary_endpoints: List[str]
    similarity_breakdown: Dict[str, float]
    enrollment_rate: Optional[float]


@app.post("/api/v1/trials/similar", response_model=List[SimilarTrialResponse], tags=["Protocol Analysis"])
async def find_similar_trials(request: SimilarTrialsRequest):
    """
    Find individual trials most similar to your protocol.

    Unlike therapeutic-area averages, this returns specific historical trials
    that match your protocol characteristics. Use this to:
    - Learn from successful similar trials
    - Avoid mistakes from failed similar trials
    - Benchmark your design against real examples
    """
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        from analysis.trial_similarity import TrialSimilarityEngine

        engine = TrialSimilarityEngine(db_manager)
        similar_trials = engine.find_similar_trials(
            condition=request.condition,
            phase=request.phase,
            eligibility_criteria=request.eligibility_criteria,
            primary_endpoint=request.primary_endpoint,
            enrollment_target=request.enrollment_target,
            limit=request.limit,
        )

        return [
            SimilarTrialResponse(
                nct_id=t.nct_id,
                title=t.title,
                similarity_score=t.similarity_score,
                status=t.status,
                phase=t.phase,
                sponsor=t.sponsor,
                enrollment=t.enrollment,
                duration_months=t.duration_months,
                num_sites=t.num_sites,
                why_stopped=t.why_stopped,
                primary_endpoints=t.primary_endpoints,
                similarity_breakdown={
                    "condition": t.condition_match,
                    "phase": t.phase_match,
                    "eligibility": t.eligibility_similarity,
                    "endpoint": t.endpoint_similarity,
                    "enrollment": t.enrollment_similarity,
                },
                enrollment_rate=t.enrollment_rate,
            )
            for t in similar_trials
        ]

    except Exception as e:
        logger.error(f"Similar trials search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Similar trials search failed")


@app.get("/api/v1/trials/{nct_id_1}/compare/{nct_id_2}", tags=["Protocol Analysis"])
async def compare_trials(
    nct_id_1: str,
    nct_id_2: str,
):
    """
    Compare two trials side-by-side.

    Shows differences in:
    - Eligibility criteria
    - Endpoints
    - Design elements (enrollment, sites, duration)
    - Outcome insights (if comparison trial is complete)
    """
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Validate NCT IDs
    if not validate_nct_id(nct_id_1) or not validate_nct_id(nct_id_2):
        raise HTTPException(status_code=400, detail="Invalid NCT ID format")

    try:
        from analysis.trial_similarity import TrialSimilarityEngine

        engine = TrialSimilarityEngine(db_manager)
        comparison = engine.compare_trials(nct_id_1, nct_id_2)

        return {
            "trial_1": {
                "nct_id": comparison.query_trial.get("nct_id"),
                "title": comparison.query_trial.get("title"),
                "status": comparison.query_trial.get("status"),
                "enrollment": comparison.query_trial.get("enrollment"),
            },
            "trial_2": {
                "nct_id": comparison.comparison_trial.get("nct_id"),
                "title": comparison.comparison_trial.get("title"),
                "status": comparison.comparison_trial.get("status"),
                "enrollment": comparison.comparison_trial.get("enrollment"),
            },
            "eligibility_differences": comparison.eligibility_differences,
            "endpoint_differences": comparison.endpoint_differences,
            "design_differences": comparison.design_differences,
            "outcome_insights": comparison.outcome_insights,
            "risk_factors_from_comparison": comparison.risk_factors_from_comparison,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Trial comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Trial comparison failed")


# ============================================================
# ENROLLMENT FORECASTING
# ============================================================

class EnrollmentForecastRequest(BaseModel):
    """Request model for enrollment forecasting."""
    condition: str = Field(..., min_length=1, max_length=200, description="Target condition")
    phase: str = Field(default="PHASE3", description="Trial phase")
    target_enrollment: int = Field(..., ge=1, le=1000000, description="Target enrollment")
    num_sites: int = Field(..., ge=1, le=10000, description="Number of planned sites")
    eligibility_criteria: Optional[str] = Field(default=None, max_length=50000, description="Eligibility criteria")
    start_date: Optional[str] = Field(default=None, description="Planned start date (YYYY-MM-DD)")

    @field_validator("phase")
    @classmethod
    def validate_phase(cls, v: str) -> str:
        if not validate_phase(v):
            raise ValueError("Invalid phase")
        return v.upper()


class EnrollmentForecastResponse(BaseModel):
    """Response model for enrollment forecast."""
    target_enrollment: int
    num_sites: int
    projected_days_to_target: float
    projected_completion_date: str
    confidence_level: str
    optimistic_days: float
    pessimistic_days: float
    projected_monthly_rate: float
    projected_rate_per_site: float
    enrollment_risk_score: float
    risk_factors: List[str]
    similar_trials_used: int
    benchmark_trials: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]


@app.post("/api/v1/enrollment/forecast", response_model=EnrollmentForecastResponse, tags=["Protocol Analysis"])
async def forecast_enrollment(request: EnrollmentForecastRequest):
    """
    Forecast enrollment timeline based on similar historical trials.

    Returns:
    - Projected days to target enrollment
    - Confidence intervals (optimistic/pessimistic)
    - Risk factors that may slow enrollment
    - Individual benchmark trials used for projection
    - Milestone dates (25%, 50%, 75%, 100%)
    """
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        from analysis.enrollment_forecasting import EnrollmentForecaster
        from analysis.trial_similarity import TrialSimilarityEngine

        # Create engines
        similarity_engine = TrialSimilarityEngine(db_manager)
        forecaster = EnrollmentForecaster(db_manager, similarity_engine)

        # Generate forecast
        projection = forecaster.forecast_enrollment(
            condition=request.condition,
            phase=request.phase,
            target_enrollment=request.target_enrollment,
            num_sites=request.num_sites,
            eligibility_criteria=request.eligibility_criteria,
            start_date=request.start_date,
        )

        return EnrollmentForecastResponse(
            target_enrollment=projection.target_enrollment,
            num_sites=projection.num_sites,
            projected_days_to_target=projection.projected_days_to_target,
            projected_completion_date=projection.projected_completion_date,
            confidence_level=projection.confidence_level,
            optimistic_days=projection.optimistic_days,
            pessimistic_days=projection.pessimistic_days,
            projected_monthly_rate=projection.projected_monthly_rate,
            projected_rate_per_site=projection.projected_rate_per_site,
            enrollment_risk_score=projection.enrollment_risk_score,
            risk_factors=projection.risk_factors,
            similar_trials_used=projection.similar_trials_used,
            benchmark_trials=projection.benchmark_trials,
            milestones=projection.projected_milestones,
        )

    except Exception as e:
        logger.error(f"Enrollment forecasting failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Enrollment forecasting failed")


class EnrollmentAlertRequest(BaseModel):
    """Request model for enrollment health check."""
    condition: str = Field(..., description="Trial condition")
    phase: str = Field(default="PHASE3", description="Trial phase")
    current_enrollment: int = Field(..., ge=0, description="Current enrolled patients")
    days_since_start: int = Field(..., ge=0, description="Days since first patient")
    target_enrollment: int = Field(..., ge=1, description="Target enrollment")
    num_sites: int = Field(..., ge=1, description="Number of active sites")


@app.post("/api/v1/enrollment/health-check", tags=["Protocol Analysis"])
async def check_enrollment_health(request: EnrollmentAlertRequest):
    """
    Check if current enrollment is on track and get early warning alerts.

    Returns alerts if:
    - Enrollment is lagging behind expected pace
    - Site productivity is below benchmark
    - Current trajectory will miss target
    """
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        from analysis.enrollment_forecasting import EnrollmentForecaster

        forecaster = EnrollmentForecaster(db_manager)
        alerts = forecaster.check_enrollment_health(
            current_enrollment=request.current_enrollment,
            days_since_start=request.days_since_start,
            target_enrollment=request.target_enrollment,
            num_sites=request.num_sites,
            condition=request.condition,
            phase=request.phase,
        )

        return {
            "status": "on_track" if not alerts else "at_risk",
            "alerts": [
                {
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "current_value": alert.current_value,
                    "expected_value": alert.expected_value,
                    "recommendation": alert.recommendation,
                }
                for alert in alerts
            ],
            "current_enrollment": request.current_enrollment,
            "target_enrollment": request.target_enrollment,
            "days_since_start": request.days_since_start,
        }

    except Exception as e:
        logger.error(f"Enrollment health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Enrollment health check failed")


# ============================================================
# SITE LEADERBOARD
# ============================================================

@app.get("/api/v1/sites/leaderboard", tags=["Site Intelligence"])
async def get_site_leaderboard(
    therapeutic_area: Optional[str] = Query(default=None, max_length=200, description="Filter by therapeutic area"),
    country: Optional[str] = Query(default="United States", max_length=100, description="Filter by country"),
    phase: Optional[str] = Query(default=None, description="Filter by trial phase"),
    metric: str = Query(default="overall", description="Sort by: overall, enrollment, completion, speed, experience"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of sites to return"),
):
    """
    Get ranked site leaderboard with performance metrics.

    Each site includes:
    - Performance scores (enrollment, completion, speed, experience)
    - Individual trial history
    - Availability status
    - Strengths and considerations
    """
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        from analysis.site_leaderboard import SiteLeaderboard

        leaderboard = SiteLeaderboard(db_manager)

        if metric == "overall":
            rankings = leaderboard.get_rankings(
                therapeutic_area=therapeutic_area,
                country=country,
                phase=phase,
                include_trial_history=True,
                limit=limit,
            )
        else:
            rankings = leaderboard.get_top_sites_by_metric(
                metric=metric,
                therapeutic_area=therapeutic_area,
                limit=limit,
            )

        return {
            "total_sites": len(rankings),
            "filters": {
                "therapeutic_area": therapeutic_area,
                "country": country,
                "phase": phase,
                "sorted_by": metric,
            },
            "rankings": [
                {
                    "rank": r.rank,
                    "site_id": r.site_id,
                    "facility_name": r.facility_name,
                    "city": r.city,
                    "state": r.state,
                    "country": r.country,
                    "scores": {
                        "overall": r.overall_score,
                        "enrollment": r.enrollment_score,
                        "completion": r.completion_score,
                        "speed": r.speed_score,
                        "experience": r.experience_score,
                    },
                    "metrics": {
                        "total_trials": r.total_trials,
                        "completed_trials": r.completed_trials,
                        "active_trials": r.active_trials,
                        "avg_enrollment": r.avg_enrollment_per_trial,
                        "enrollment_velocity": r.enrollment_velocity,
                        "completion_rate": r.completion_rate,
                    },
                    "availability": r.availability_status,
                    "diversity_score": r.diversity_score,
                    "strengths": r.strengths,
                    "considerations": r.considerations,
                    "trial_history_count": len(r.trial_history),
                }
                for r in rankings
            ],
        }

    except Exception as e:
        logger.error(f"Site leaderboard failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Site leaderboard failed")


@app.get("/api/v1/sites/{site_id}/compare/{site_id_2}", tags=["Site Intelligence"])
async def compare_sites(
    site_id: int,
    site_id_2: int,
    therapeutic_area: Optional[str] = Query(default=None, description="Therapeutic area for comparison"),
):
    """
    Compare two sites head-to-head.

    Returns:
    - Side-by-side metric comparison
    - Winner by each category
    - Recommendation
    """
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        from analysis.site_leaderboard import SiteLeaderboard

        leaderboard = SiteLeaderboard(db_manager)
        comparison = leaderboard.compare_sites(
            site_id_a=site_id,
            site_id_b=site_id_2,
            therapeutic_area=therapeutic_area,
        )

        return {
            "site_a": {
                "site_id": comparison.site_a.site_id,
                "facility_name": comparison.site_a.facility_name,
                "overall_score": comparison.site_a.overall_score,
            },
            "site_b": {
                "site_id": comparison.site_b.site_id,
                "facility_name": comparison.site_b.facility_name,
                "overall_score": comparison.site_b.overall_score,
            },
            "winners": {
                "enrollment": comparison.better_enrollment,
                "completion": comparison.better_completion,
                "speed": comparison.better_speed,
                "experience": comparison.better_experience,
            },
            "metric_comparisons": comparison.metric_comparisons,
            "recommendation": comparison.recommendation,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Site comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Site comparison failed")


# ============================================================
# ROI CALCULATOR
# ============================================================

class ROICalculationRequest(BaseModel):
    """Request model for ROI calculation."""
    therapeutic_area: str = Field(..., min_length=1, max_length=200, description="Therapeutic area")
    phase: str = Field(default="PHASE3", description="Trial phase")
    target_enrollment: int = Field(..., ge=1, le=1000000, description="Target enrollment")
    num_sites: int = Field(..., ge=1, le=10000, description="Number of sites")
    baseline_duration_months: float = Field(..., ge=1, le=240, description="Expected duration without optimization")
    protocol_risk_reduction: Optional[float] = Field(default=20, ge=0, le=100, description="Expected risk score reduction")
    enrollment_improvement_pct: Optional[float] = Field(default=15, ge=0, le=100, description="Expected enrollment improvement %")


class ROIResponse(BaseModel):
    """Response model for ROI calculation."""
    total_days_saved: int
    total_cost_saved: int
    total_revenue_impact: int
    roi_percentage: float
    confidence_level: str
    savings_breakdown: List[Dict[str, Any]]
    key_assumptions: List[str]
    timeline_impact: Dict[str, Any]


@app.post("/api/v1/roi/calculate", response_model=ROIResponse, tags=["Protocol Analysis"])
async def calculate_roi(request: ROICalculationRequest):
    """
    Calculate ROI from TrialIntel optimizations.

    Shows tangible value:
    - Days saved from better site selection
    - Cost savings from reduced amendments
    - Revenue acceleration from faster enrollment
    - Breakdown by optimization type
    """
    try:
        from analysis.roi_calculator import ROICalculator

        calculator = ROICalculator()
        roi = calculator.calculate_full_roi(
            therapeutic_area=request.therapeutic_area,
            phase=request.phase,
            target_enrollment=request.target_enrollment,
            num_sites=request.num_sites,
            baseline_duration_months=request.baseline_duration_months,
            optimizations_applied=[
                "site_selection",
                "protocol_risk",
                "enrollment_forecast",
            ],
            protocol_risk_reduction=request.protocol_risk_reduction,
            enrollment_improvement_pct=request.enrollment_improvement_pct,
            similar_trial_insights_used=True,
        )

        return ROIResponse(
            total_days_saved=roi.total_days_saved,
            total_cost_saved=roi.total_cost_saved,
            total_revenue_impact=roi.total_revenue_impact,
            roi_percentage=roi.roi_percentage,
            confidence_level=roi.confidence_level,
            savings_breakdown=[
                {
                    "optimization_type": s.optimization_type,
                    "description": s.description,
                    "days_saved": s.days_saved,
                    "cost_saved": s.cost_saved,
                    "confidence": s.confidence,
                    "calculation_basis": s.calculation_basis,
                }
                for s in roi.savings_breakdown
            ],
            key_assumptions=roi.key_assumptions,
            timeline_impact={
                "baseline_days": roi.baseline_duration_days,
                "optimized_days": roi.optimized_duration_days,
                "improvement_pct": round(
                    (roi.baseline_duration_days - roi.optimized_duration_days) /
                    roi.baseline_duration_days * 100, 1
                ) if roi.baseline_duration_days > 0 else 0,
            },
        )

    except Exception as e:
        logger.error(f"ROI calculation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="ROI calculation failed")


@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """Health check endpoint for monitoring."""
    db_status = "operational" if db_manager is not None else "not_initialized"

    # Check AI assistant availability
    ai_status = "not_available"
    if ANTHROPIC_AVAILABLE:
        if settings.ai.anthropic_api_key:
            ai_status = "operational"
        else:
            ai_status = "api_key_missing"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment,
        "components": {
            "database": db_status,
            "risk_scorer": "operational" if risk_scorer is not None else "not_initialized",
            "site_intel": "operational" if site_intel is not None else "not_initialized",
            "endpoint_benchmarker": "operational" if endpoint_benchmarker is not None else "not_initialized",
            "ai_assistant": ai_status,
        }
    }


@app.get("/api/v1/stats", tags=["System"])
async def get_database_stats():
    """Get database statistics."""
    if db_manager is None:
        return {
            "database": "not_initialized",
            "message": "Run the data pipeline to populate the database",
        }

    try:
        stats = db_manager.get_stats()
        return {
            "database": "operational",
            "total_trials": stats.get("total_trials", 0),
            "total_sites": stats.get("total_sites", 0),
            "total_endpoints": stats.get("total_endpoints", 0),
            "by_status": stats.get("by_status", {}),
            "by_phase": stats.get("by_phase", {}),
            "by_therapeutic_area": stats.get("by_therapeutic_area", {}),
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve database statistics")


# ============================================================
# WEBHOOK ENDPOINTS (for Jeeva integration)
# ============================================================

def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str,
) -> bool:
    """Verify webhook signature using HMAC-SHA256."""
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


async def send_webhook_callback(
    callback_url: str,
    payload: WebhookCallbackPayload,
    api_key: Optional[str] = None,
):
    """Send results back to the webhook caller."""
    import httpx

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                callback_url,
                json=payload.model_dump(mode="json"),
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            logger.info(f"Webhook callback sent successfully to {callback_url}")
    except httpx.HTTPError as e:
        logger.error(f"Webhook callback failed: {e}")


@app.post("/api/v1/webhooks/jeeva", response_model=WebhookResponse, tags=["Webhooks"])
async def jeeva_webhook(
    request: Request,
    payload: JeevaWebhookPayload,
    background_tasks: BackgroundTasks,
    x_webhook_signature: Optional[str] = Header(None, alias="X-Webhook-Signature"),
):
    """
    Webhook endpoint for Jeeva integration.

    Receives events from Jeeva when:
    - A new protocol is created (protocol_created)
    - Protocol is updated (protocol_updated)
    - Sites are being selected (site_selection_started)
    - Enrollment numbers change (enrollment_updated)
    - Trial completes (trial_completed)

    Triggers background analysis and sends results back to Jeeva.

    **Security**: Optionally verifies webhook signature using HMAC-SHA256.
    Set JEEVA_WEBHOOK_SECRET environment variable to enable signature verification.
    """
    # Verify webhook signature if secret is configured
    webhook_secret = os.getenv("JEEVA_WEBHOOK_SECRET")
    if webhook_secret and x_webhook_signature:
        body = await request.body()
        if not verify_webhook_signature(body, x_webhook_signature, webhook_secret):
            logger.warning(f"Invalid webhook signature for trial {payload.trial_id}")
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    logger.info(f"Received webhook: {payload.event_type} for trial {payload.trial_id}")

    try:
        task_id = f"{payload.event_type}_{payload.trial_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        if payload.event_type == "protocol_created":
            # Queue protocol analysis
            background_tasks.add_task(
                analyze_new_protocol,
                payload.trial_id,
                payload.data,
            )
            return WebhookResponse(
                status="accepted",
                message="Protocol analysis queued",
                task_id=task_id,
            )

        elif payload.event_type == "protocol_updated":
            # Queue protocol re-analysis
            background_tasks.add_task(
                analyze_updated_protocol,
                payload.trial_id,
                payload.data,
            )
            return WebhookResponse(
                status="accepted",
                message="Protocol re-analysis queued",
                task_id=task_id,
            )

        elif payload.event_type == "site_selection_started":
            # Queue site recommendations
            background_tasks.add_task(
                generate_site_recommendations,
                payload.trial_id,
                payload.data,
            )
            return WebhookResponse(
                status="accepted",
                message="Site recommendations queued",
                task_id=task_id,
            )

        elif payload.event_type == "enrollment_updated":
            # Queue enrollment analysis
            background_tasks.add_task(
                analyze_enrollment_status,
                payload.trial_id,
                payload.data,
            )
            return WebhookResponse(
                status="accepted",
                message="Enrollment analysis queued",
                task_id=task_id,
            )

        elif payload.event_type == "trial_completed":
            # Queue trial completion processing
            background_tasks.add_task(
                process_trial_completion,
                payload.trial_id,
                payload.data,
            )
            return WebhookResponse(
                status="accepted",
                message="Trial completion processing queued",
                task_id=task_id,
            )

        else:
            return WebhookResponse(
                status="ignored",
                message=f"Unknown event type: {payload.event_type}",
            )

    except Exception as e:
        logger.error(f"Webhook processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Webhook processing failed")


async def analyze_new_protocol(trial_id: str, data: Dict[str, Any]):
    """Background task to analyze a new protocol."""
    logger.info(f"Starting protocol analysis for trial {trial_id}")

    try:
        # Extract protocol details from webhook data
        condition = data.get("condition", "")
        phase = data.get("phase", "PHASE3")
        eligibility_criteria = data.get("eligibility_criteria", "")
        primary_endpoints = data.get("primary_endpoints", [])
        target_enrollment = data.get("target_enrollment", 100)
        planned_sites = data.get("planned_sites", 10)
        planned_duration = data.get("planned_duration_months", 24)

        # Run risk analysis
        assessment = risk_scorer.score_protocol(
            condition=condition,
            phase=phase,
            eligibility_criteria=eligibility_criteria,
            primary_endpoints=primary_endpoints,
            target_enrollment=target_enrollment,
            planned_sites=planned_sites,
            planned_duration_months=planned_duration,
        )

        # Prepare results
        results = {
            "trial_id": trial_id,
            "overall_risk_score": assessment.overall_risk_score,
            "amendment_probability": assessment.amendment_probability,
            "enrollment_delay_probability": assessment.enrollment_delay_probability,
            "termination_probability": assessment.termination_probability,
            "risk_factors": [
                {
                    "category": rf.category,
                    "description": rf.description,
                    "severity": rf.severity,
                    "recommendation": rf.recommendation,
                }
                for rf in assessment.risk_factors
            ],
            "recommendations": assessment.recommendations[:5],
        }

        # Send results back to Jeeva if callback URL configured
        callback_url = os.getenv("JEEVA_API_URL")
        if callback_url:
            callback_payload = WebhookCallbackPayload(
                event_type="protocol_analysis_complete",
                trial_id=trial_id,
                results=results,
            )
            await send_webhook_callback(
                f"{callback_url}/api/v1/webhooks/trialintel",
                callback_payload,
                os.getenv("JEEVA_API_KEY"),
            )

        logger.info(f"Protocol analysis completed for trial {trial_id}: risk_score={assessment.overall_risk_score}")

    except Exception as e:
        logger.error(f"Protocol analysis failed for trial {trial_id}: {e}", exc_info=True)


async def analyze_updated_protocol(trial_id: str, data: Dict[str, Any]):
    """Background task to re-analyze an updated protocol."""
    logger.info(f"Starting protocol re-analysis for trial {trial_id}")

    # Reuse the same analysis logic
    await analyze_new_protocol(trial_id, data)


async def generate_site_recommendations(trial_id: str, data: Dict[str, Any]):
    """Background task to generate site recommendations."""
    logger.info(f"Starting site recommendation generation for trial {trial_id}")

    try:
        # Extract requirements from webhook data
        therapeutic_area = data.get("therapeutic_area", data.get("condition", ""))
        target_enrollment = data.get("target_enrollment", 100)
        num_sites = data.get("num_sites", 20)
        country_filter = data.get("country")

        # Generate recommendations
        recommendations = site_intel.recommend_sites(
            therapeutic_area=therapeutic_area,
            target_enrollment=target_enrollment,
            num_sites=num_sites,
            country_filter=country_filter,
            prioritize_diversity=True,
        )

        # Prepare results
        results = {
            "trial_id": trial_id,
            "recommendations": [
                {
                    "facility_name": rec.site.facility_name,
                    "city": rec.site.city,
                    "state": rec.site.state,
                    "country": rec.site.country,
                    "match_score": rec.match_score,
                    "reasons": rec.reasons,
                    "projected_enrollment": rec.projected_enrollment,
                    "projected_timeline_months": rec.projected_timeline_months,
                }
                for rec in recommendations
            ],
        }

        # Send results back to Jeeva
        callback_url = os.getenv("JEEVA_API_URL")
        if callback_url:
            callback_payload = WebhookCallbackPayload(
                event_type="site_recommendations_ready",
                trial_id=trial_id,
                results=results,
            )
            await send_webhook_callback(
                f"{callback_url}/api/v1/webhooks/trialintel",
                callback_payload,
                os.getenv("JEEVA_API_KEY"),
            )

        logger.info(f"Site recommendations generated for trial {trial_id}: {len(recommendations)} sites")

    except Exception as e:
        logger.error(f"Site recommendation failed for trial {trial_id}: {e}", exc_info=True)


async def analyze_enrollment_status(trial_id: str, data: Dict[str, Any]):
    """Background task to analyze enrollment status and provide recommendations."""
    logger.info(f"Starting enrollment analysis for trial {trial_id}")

    try:
        current_enrollment = data.get("current_enrollment", 0)
        target_enrollment = data.get("target_enrollment", 100)
        elapsed_months = data.get("elapsed_months", 0)
        planned_duration = data.get("planned_duration_months", 24)
        site_performance = data.get("site_performance", [])

        # Calculate enrollment metrics
        enrollment_rate = current_enrollment / max(elapsed_months, 1)
        projected_total = enrollment_rate * planned_duration
        on_track = projected_total >= target_enrollment * 0.9

        # Identify underperforming sites
        underperforming_sites = [
            site for site in site_performance
            if site.get("actual", 0) < site.get("projected", 1) * 0.7
        ]

        results = {
            "trial_id": trial_id,
            "current_enrollment": current_enrollment,
            "target_enrollment": target_enrollment,
            "enrollment_rate_per_month": round(enrollment_rate, 2),
            "projected_final_enrollment": round(projected_total),
            "on_track": on_track,
            "underperforming_sites": underperforming_sites[:10],
            "recommendations": [],
        }

        # Add recommendations based on status
        if not on_track:
            results["recommendations"].extend([
                "Consider adding additional sites to meet enrollment targets",
                "Review eligibility criteria for potential modifications",
                "Evaluate underperforming sites for root cause analysis",
            ])

        # Send results back to Jeeva
        callback_url = os.getenv("JEEVA_API_URL")
        if callback_url:
            callback_payload = WebhookCallbackPayload(
                event_type="enrollment_analysis_complete",
                trial_id=trial_id,
                results=results,
            )
            await send_webhook_callback(
                f"{callback_url}/api/v1/webhooks/trialintel",
                callback_payload,
                os.getenv("JEEVA_API_KEY"),
            )

        logger.info(f"Enrollment analysis completed for trial {trial_id}: on_track={on_track}")

    except Exception as e:
        logger.error(f"Enrollment analysis failed for trial {trial_id}: {e}", exc_info=True)


async def process_trial_completion(trial_id: str, data: Dict[str, Any]):
    """Background task to process trial completion and update metrics."""
    logger.info(f"Processing trial completion for trial {trial_id}")

    try:
        # Extract completion data
        final_enrollment = data.get("final_enrollment", 0)
        actual_duration_months = data.get("actual_duration_months", 0)
        outcome_status = data.get("outcome_status", "unknown")  # completed, terminated, withdrawn
        site_metrics = data.get("site_metrics", [])

        # Store completion metrics for future analysis (would update database in production)
        results = {
            "trial_id": trial_id,
            "final_enrollment": final_enrollment,
            "actual_duration_months": actual_duration_months,
            "outcome_status": outcome_status,
            "sites_processed": len(site_metrics),
            "message": "Trial completion data processed successfully",
        }

        # Send acknowledgment back to Jeeva
        callback_url = os.getenv("JEEVA_API_URL")
        if callback_url:
            callback_payload = WebhookCallbackPayload(
                event_type="trial_completion_processed",
                trial_id=trial_id,
                results=results,
            )
            await send_webhook_callback(
                f"{callback_url}/api/v1/webhooks/trialintel",
                callback_payload,
                os.getenv("JEEVA_API_KEY"),
            )

        logger.info(f"Trial completion processed for trial {trial_id}: status={outcome_status}")

    except Exception as e:
        logger.error(f"Trial completion processing failed for trial {trial_id}: {e}", exc_info=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
    )
