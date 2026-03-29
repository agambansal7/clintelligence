"""
Enhanced Protocol Extractor

Extracts structured, multi-dimensional information from protocol text
for precise trial matching.
"""

import os
import json
import logging
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class PopulationCriteria:
    """Structured population/eligibility criteria."""
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    sex: str = "all"  # all, male, female

    # Conditions required
    required_conditions: List[str] = field(default_factory=list)
    # Conditions that exclude patients
    excluded_conditions: List[str] = field(default_factory=list)

    # Key lab requirements
    bmi_min: Optional[float] = None
    bmi_max: Optional[float] = None
    hba1c_min: Optional[float] = None
    hba1c_max: Optional[float] = None
    egfr_min: Optional[float] = None

    # Prior treatment requirements
    required_prior_treatments: List[str] = field(default_factory=list)
    excluded_prior_treatments: List[str] = field(default_factory=list)

    # Other key criteria
    performance_status_max: Optional[int] = None  # ECOG
    pregnant_allowed: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InterventionInfo:
    """Structured intervention information."""
    intervention_type: str = ""  # drug, biological, device, behavioral, procedure
    drug_name: Optional[str] = None
    drug_class: Optional[str] = None  # e.g., "GLP-1 receptor agonist"
    mechanism_of_action: Optional[str] = None

    # Administration
    route: Optional[str] = None  # oral, subcutaneous, intravenous
    frequency: Optional[str] = None  # daily, weekly, q2w, q3w
    dose_range: Optional[str] = None

    # Comparator
    comparator_type: Optional[str] = None  # placebo, active, standard of care
    comparator_name: Optional[str] = None

    # Inferred similar drugs (for unknown drugs)
    similar_known_drugs: List[str] = field(default_factory=list)
    search_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EndpointInfo:
    """Structured endpoint information."""
    primary_endpoint: str = ""
    primary_endpoint_type: str = ""  # efficacy, safety, pk, pro
    primary_endpoint_measure: str = ""  # change from baseline, time to event, response rate
    primary_timepoint: Optional[str] = None  # e.g., "Week 72"

    secondary_endpoints: List[str] = field(default_factory=list)

    # Categorized endpoint types for matching
    has_weight_endpoint: bool = False
    has_glycemic_endpoint: bool = False
    has_cv_endpoint: bool = False
    has_survival_endpoint: bool = False
    has_response_rate_endpoint: bool = False
    has_pro_endpoint: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StudyDesignInfo:
    """Structured study design information."""
    phase: str = ""  # PHASE1, PHASE2, PHASE3, PHASE4
    study_type: str = ""  # interventional, observational
    randomized: bool = True
    blinding: str = ""  # open, single, double, triple
    controlled: bool = True
    control_type: str = ""  # placebo, active, none

    # Size and duration
    target_enrollment: int = 0
    number_of_arms: int = 2
    duration_weeks: Optional[int] = None

    # Stratification factors
    stratification_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExtractedProtocolV2:
    """Complete structured protocol extraction."""
    # Core identification
    condition: str = ""
    condition_category: str = ""  # e.g., "obesity", "oncology", "cardiology"
    therapeutic_area: str = ""

    # Sponsor information
    sponsor: str = ""  # Primary sponsor/company name

    # Detailed components
    population: PopulationCriteria = field(default_factory=PopulationCriteria)
    intervention: InterventionInfo = field(default_factory=InterventionInfo)
    endpoints: EndpointInfo = field(default_factory=EndpointInfo)
    design: StudyDesignInfo = field(default_factory=StudyDesignInfo)

    # Search optimization
    condition_synonyms: List[str] = field(default_factory=list)
    intervention_search_terms: List[str] = field(default_factory=list)

    # Raw text for reference
    raw_protocol: str = ""

    def to_dict(self) -> Dict:
        return {
            "condition": self.condition,
            "condition_category": self.condition_category,
            "therapeutic_area": self.therapeutic_area,
            "sponsor": self.sponsor,
            "population": self.population.to_dict(),
            "intervention": self.intervention.to_dict(),
            "endpoints": self.endpoints.to_dict(),
            "design": self.design.to_dict(),
            "condition_synonyms": self.condition_synonyms,
            "intervention_search_terms": self.intervention_search_terms,
        }


class EnhancedProtocolExtractor:
    """
    Extract detailed structured information from protocol text using Claude.
    """

    EXTRACTION_PROMPT = '''You are an expert clinical trial protocol analyst. Extract detailed structured information from this protocol.

<protocol>
{protocol_text}
</protocol>

Analyze carefully and return a JSON object with these sections:

{{
    "sponsor": {{
        "name": "Sponsor/company name if mentioned (e.g., 'Mirati Therapeutics', 'Pfizer', 'Novartis'). Look for company names, pharma sponsors, or CRO names. Return empty string if not found.",
        "type": "industry, academic, government, or unknown"
    }},

    "condition": {{
        "primary_condition": "Main condition being studied (standardized medical term)",
        "condition_category": "Category: obesity, diabetes, oncology, cardiology, neurology, immunology, infectious, respiratory, other",
        "therapeutic_area": "Broad therapeutic area",
        "synonyms": ["List of condition synonyms and related terms for search"]
    }},

    "population": {{
        "min_age": <integer or null>,
        "max_age": <integer or null>,
        "sex": "all, male, or female",
        "required_conditions": ["Conditions patients MUST have to enroll"],
        "excluded_conditions": ["Conditions that EXCLUDE patients - BE THOROUGH HERE"],
        "bmi_min": <number or null>,
        "bmi_max": <number or null>,
        "hba1c_min": <number or null>,
        "hba1c_max": <number or null>,
        "egfr_min": <number or null>,
        "required_prior_treatments": ["Treatments patients must have tried"],
        "excluded_prior_treatments": ["Treatments that exclude patients"],
        "performance_status_max": <ECOG score or null>,
        "key_inclusion_summary": "Brief summary of key inclusion criteria",
        "key_exclusion_summary": "Brief summary of key exclusion criteria"
    }},

    "intervention": {{
        "type": "drug, biological, device, behavioral, procedure, or combination",
        "drug_name": "Name if known, or 'investigational' if unknown",
        "drug_class": "Drug class if identifiable (e.g., 'GLP-1 receptor agonist', 'PD-1 inhibitor')",
        "mechanism_of_action": "Mechanism if described or inferable",
        "route": "oral, subcutaneous, intravenous, topical, etc.",
        "frequency": "daily, twice daily, weekly, every 2 weeks, etc.",
        "dose_range": "Dose information if available",
        "comparator_type": "placebo, active comparator, standard of care, or none",
        "comparator_name": "Name of active comparator if applicable",
        "inferred_similar_drugs": ["If drug is unknown but class is clear, list similar approved drugs"],
        "search_terms": ["Terms to search for similar trials - include drug class, mechanism, brand names"]
    }},

    "endpoints": {{
        "primary_endpoint": "Exact primary endpoint",
        "primary_endpoint_type": "efficacy, safety, pharmacokinetic, patient-reported",
        "primary_endpoint_measure": "change from baseline, time to event, response rate, proportion achieving threshold",
        "primary_timepoint": "When primary endpoint is measured (e.g., Week 72)",
        "secondary_endpoints": ["List of secondary endpoints"],
        "endpoint_categories": {{
            "has_weight_endpoint": true/false,
            "has_glycemic_endpoint": true/false,
            "has_cv_endpoint": true/false,
            "has_survival_endpoint": true/false,
            "has_response_rate_endpoint": true/false,
            "has_pro_endpoint": true/false
        }}
    }},

    "design": {{
        "phase": "PHASE1, PHASE2, PHASE3, PHASE4, or NA",
        "study_type": "interventional or observational",
        "randomized": true/false,
        "blinding": "open-label, single-blind, double-blind, triple-blind",
        "controlled": true/false,
        "control_type": "placebo-controlled, active-controlled, or uncontrolled",
        "target_enrollment": <integer>,
        "number_of_arms": <integer>,
        "duration_weeks": <integer or null>,
        "stratification_factors": ["List of stratification factors"]
    }}
}}

IMPORTANT:
- For excluded_conditions: Be very thorough. If the protocol says "excludes diabetes" or "no history of cancer", capture these.
- For drug_class: Try to infer from route, frequency, dose patterns. Weekly subcutaneous with dose titration often = GLP-1. IV q3w in oncology often = checkpoint inhibitor.
- For inferred_similar_drugs: If the drug is novel/unknown, list approved drugs in the same class.
- Return ONLY valid JSON, no other text.'''

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def extract(self, protocol_text: str) -> ExtractedProtocolV2:
        """Extract structured protocol information."""

        prompt = self.EXTRACTION_PROMPT.format(protocol_text=protocol_text[:15000])

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text

        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        data = json.loads(response_text.strip())

        # Build structured objects
        protocol = ExtractedProtocolV2()
        protocol.raw_protocol = protocol_text

        # Sponsor
        sponsor_data = data.get("sponsor", {})
        protocol.sponsor = sponsor_data.get("name", "") if isinstance(sponsor_data, dict) else ""

        # Condition
        cond = data.get("condition", {})
        protocol.condition = cond.get("primary_condition", "")
        protocol.condition_category = cond.get("condition_category", "")
        protocol.therapeutic_area = cond.get("therapeutic_area", "")
        protocol.condition_synonyms = cond.get("synonyms", [])

        # Population
        pop = data.get("population", {})
        protocol.population = PopulationCriteria(
            min_age=pop.get("min_age"),
            max_age=pop.get("max_age"),
            sex=pop.get("sex", "all"),
            required_conditions=pop.get("required_conditions", []),
            excluded_conditions=pop.get("excluded_conditions", []),
            bmi_min=pop.get("bmi_min"),
            bmi_max=pop.get("bmi_max"),
            hba1c_min=pop.get("hba1c_min"),
            hba1c_max=pop.get("hba1c_max"),
            egfr_min=pop.get("egfr_min"),
            required_prior_treatments=pop.get("required_prior_treatments", []),
            excluded_prior_treatments=pop.get("excluded_prior_treatments", []),
            performance_status_max=pop.get("performance_status_max"),
            pregnant_allowed=pop.get("pregnant_allowed", False),
        )

        # Intervention
        intv = data.get("intervention", {})
        protocol.intervention = InterventionInfo(
            intervention_type=intv.get("type", ""),
            drug_name=intv.get("drug_name"),
            drug_class=intv.get("drug_class"),
            mechanism_of_action=intv.get("mechanism_of_action"),
            route=intv.get("route"),
            frequency=intv.get("frequency"),
            dose_range=intv.get("dose_range"),
            comparator_type=intv.get("comparator_type"),
            comparator_name=intv.get("comparator_name"),
            similar_known_drugs=intv.get("inferred_similar_drugs", []),
            search_terms=intv.get("search_terms", []),
        )
        protocol.intervention_search_terms = intv.get("search_terms", [])

        # Endpoints
        endp = data.get("endpoints", {})
        cats = endp.get("endpoint_categories", {})
        protocol.endpoints = EndpointInfo(
            primary_endpoint=endp.get("primary_endpoint", ""),
            primary_endpoint_type=endp.get("primary_endpoint_type", ""),
            primary_endpoint_measure=endp.get("primary_endpoint_measure", ""),
            primary_timepoint=endp.get("primary_timepoint"),
            secondary_endpoints=endp.get("secondary_endpoints", []),
            has_weight_endpoint=cats.get("has_weight_endpoint", False),
            has_glycemic_endpoint=cats.get("has_glycemic_endpoint", False),
            has_cv_endpoint=cats.get("has_cv_endpoint", False),
            has_survival_endpoint=cats.get("has_survival_endpoint", False),
            has_response_rate_endpoint=cats.get("has_response_rate_endpoint", False),
            has_pro_endpoint=cats.get("has_pro_endpoint", False),
        )

        # Design
        des = data.get("design", {})
        protocol.design = StudyDesignInfo(
            phase=des.get("phase", ""),
            study_type=des.get("study_type", "interventional"),
            randomized=des.get("randomized", True),
            blinding=des.get("blinding", ""),
            controlled=des.get("controlled", True),
            control_type=des.get("control_type", ""),
            target_enrollment=des.get("target_enrollment", 0),
            number_of_arms=des.get("number_of_arms", 2),
            duration_weeks=des.get("duration_weeks"),
            stratification_factors=des.get("stratification_factors", []),
        )

        logger.info(f"Extracted protocol: {protocol.condition} | {protocol.intervention.drug_class} | {protocol.design.phase}")

        return protocol


# Singleton
_extractor = None

def get_extractor() -> EnhancedProtocolExtractor:
    global _extractor
    if _extractor is None:
        _extractor = EnhancedProtocolExtractor()
    return _extractor
