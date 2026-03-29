"""
Enhanced Risk & Termination Analyzer

Provides comprehensive risk analysis including:
1. AI-powered termination reason classification
2. Protective factor analysis (what successful trials have)
3. Protocol-specific risk scoring with interaction terms
4. Temporal risk patterns (when do failures occur)
5. Root cause analysis for common failures
6. Risk mitigation recommendations
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class TerminationCategory:
    """A category of termination reason."""
    name: str
    count: int
    percentage: float
    risk_level: str  # high, medium, low
    description: str
    examples: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class RiskFactor:
    """An individual risk factor."""
    category: str  # enrollment, design, regulatory, operational, scientific, market
    name: str
    severity: str  # high, medium, low
    score: float  # 0-100
    description: str
    evidence: str
    mitigation: str
    confidence: str  # high, medium, low


@dataclass
class ProtectiveFactor:
    """A factor that correlates with trial success."""
    name: str
    impact: str  # strong, moderate, weak
    description: str
    your_protocol_status: str  # present, partial, absent
    recommendation: str


@dataclass
class TemporalRisk:
    """Risk pattern over trial lifecycle."""
    phase: str  # startup, enrollment, treatment, analysis
    risk_level: str
    common_issues: List[str]
    watchpoints: List[str]


@dataclass
class RiskScore:
    """Comprehensive risk score breakdown."""
    overall_score: float  # 0-100 (higher = riskier)
    risk_level: str  # high, medium, low
    confidence: str  # high, medium, low

    # Component scores
    enrollment_risk: float
    design_risk: float
    regulatory_risk: float
    operational_risk: float
    scientific_risk: float
    market_risk: float

    # Interaction effects
    interaction_effects: List[str]

    # Comparison to benchmarks
    vs_condition_average: str  # above, at, below
    vs_phase_average: str


@dataclass
class RootCauseAnalysis:
    """Deep analysis of a termination pattern."""
    category: str
    root_causes: List[str]
    contributing_factors: List[str]
    early_warning_signs: List[str]
    prevention_strategies: List[str]
    case_studies: List[Dict[str, str]]


@dataclass
class RiskAssessment:
    """Complete risk assessment report."""
    # Protocol info
    condition: str
    phase: str
    target_enrollment: int

    # Overall assessment
    risk_score: RiskScore
    success_probability: float  # 0-100%

    # Detailed analysis
    risk_factors: List[RiskFactor]
    protective_factors: List[ProtectiveFactor]
    termination_categories: List[TerminationCategory]
    root_cause_analyses: List[RootCauseAnalysis]
    temporal_risks: List[TemporalRisk]

    # Historical context
    similar_trials_analyzed: int
    completed_count: int
    terminated_count: int
    historical_success_rate: float

    # Recommendations
    top_risks: List[str]
    mitigation_priorities: List[str]
    key_insights: List[str]


class RiskAnalyzer:
    """
    Comprehensive risk and termination analysis.
    """

    TERMINATION_CLASSIFICATION_PROMPT = """Classify these trial termination reasons into detailed categories.

<termination_reasons>
{reasons_json}
</termination_reasons>

For each reason, classify into one of these categories:
1. **Enrollment Failure** - couldn't recruit enough patients
   - Sub-types: slow_accrual, competing_trials, restrictive_criteria, rare_population, site_issues
2. **Efficacy/Futility** - treatment didn't work
   - Sub-types: interim_futility, primary_endpoint_miss, lack_of_efficacy, no_difference
3. **Safety Concerns** - safety signals emerged
   - Sub-types: adverse_events, toxicity, deaths, risk_benefit
4. **Business/Strategic** - sponsor decision
   - Sub-types: funding, priority_change, merger, portfolio_decision, market_factors
5. **Regulatory** - regulatory issues
   - Sub-types: fda_hold, protocol_issues, compliance, data_integrity
6. **Operational** - execution problems
   - Sub-types: site_performance, data_quality, supply_issues, staffing
7. **Scientific** - scientific issues
   - Sub-types: biomarker_failure, wrong_population, mechanism_issue

Return JSON:
{{
    "classifications": [
        {{
            "nct_id": "NCT...",
            "original_reason": "...",
            "primary_category": "enrollment_failure|efficacy|safety|business|regulatory|operational|scientific",
            "sub_type": "specific sub-type",
            "confidence": "high|medium|low",
            "root_cause_hypothesis": "Brief hypothesis of underlying cause"
        }}
    ],
    "category_summary": {{
        "enrollment_failure": {{"count": N, "common_patterns": ["pattern1", "pattern2"]}},
        ...
    }}
}}"""

    RISK_ANALYSIS_PROMPT = """Analyze the risk profile for this clinical trial protocol based on historical data.

<user_protocol>
Condition: {condition}
Phase: {phase}
Target Enrollment: {target_enrollment}
Number of Sites: {num_sites}
Intervention: {intervention}
Primary Endpoint: {endpoint}
Key Eligibility: {eligibility}
</user_protocol>

<historical_context>
Similar Trials Analyzed: {similar_count}
Completion Rate: {completion_rate}%
Top Termination Reasons: {top_terminations}
Average Enrollment: {avg_enrollment}
Average Duration: {avg_duration} months
</historical_context>

<terminated_trials_sample>
{terminated_sample}
</terminated_trials_sample>

<completed_trials_sample>
{completed_sample}
</completed_trials_sample>

Analyze:
1. **Risk Factors**: What specific risks does this protocol face?
2. **Protective Factors**: What factors from successful trials should be adopted?
3. **Interaction Effects**: How do risks compound (e.g., rare disease + complex criteria)?
4. **Root Causes**: For common failures, what are the root causes?
5. **Mitigation Strategies**: How to reduce each major risk?

Return JSON:
{{
    "risk_factors": [
        {{
            "category": "enrollment|design|regulatory|operational|scientific|market",
            "name": "Specific risk name",
            "severity": "high|medium|low",
            "score": <0-100>,
            "description": "Why this is a risk",
            "evidence": "Evidence from historical data",
            "mitigation": "How to address this risk",
            "confidence": "high|medium|low"
        }}
    ],
    "protective_factors": [
        {{
            "name": "Factor name",
            "impact": "strong|moderate|weak",
            "description": "Why this helps",
            "your_protocol_status": "present|partial|absent",
            "recommendation": "What to do"
        }}
    ],
    "interaction_effects": [
        "Description of how risks compound"
    ],
    "root_cause_analyses": [
        {{
            "category": "Category of failure",
            "root_causes": ["cause1", "cause2"],
            "early_warning_signs": ["sign1", "sign2"],
            "prevention_strategies": ["strategy1", "strategy2"]
        }}
    ],
    "overall_risk_assessment": {{
        "enrollment_risk": <0-100>,
        "design_risk": <0-100>,
        "regulatory_risk": <0-100>,
        "operational_risk": <0-100>,
        "scientific_risk": <0-100>,
        "market_risk": <0-100>
    }},
    "success_probability": <0-100>,
    "top_risks": ["Most critical risk 1", "Most critical risk 2", "Most critical risk 3"],
    "mitigation_priorities": ["Priority 1", "Priority 2", "Priority 3"],
    "key_insights": ["Insight 1", "Insight 2"]
}}"""

    # Termination category keywords for initial classification
    TERMINATION_KEYWORDS = {
        "enrollment_failure": {
            "keywords": ["enroll", "recruit", "accrual", "patient", "subject", "participant",
                        "slow", "insufficient", "unable to recruit", "low enrollment"],
            "sub_types": {
                "slow_accrual": ["slow", "behind", "slower than"],
                "competing_trials": ["competing", "competition"],
                "restrictive_criteria": ["criteria", "eligibility", "restrictive"],
                "rare_population": ["rare", "uncommon", "difficult to find"],
                "site_issues": ["site", "investigator", "center"]
            }
        },
        "efficacy": {
            "keywords": ["efficacy", "futility", "interim", "endpoint", "ineffective",
                        "no benefit", "lack of efficacy", "did not meet", "negative"],
            "sub_types": {
                "interim_futility": ["interim", "futility", "dsmb"],
                "primary_endpoint_miss": ["primary", "endpoint", "did not meet"],
                "lack_of_efficacy": ["lack", "insufficient", "no efficacy"],
                "no_difference": ["no difference", "similar", "not superior"]
            }
        },
        "safety": {
            "keywords": ["safety", "adverse", "toxicity", "death", "serious", "sae",
                        "risk", "harmful", "side effect", "tolerability"],
            "sub_types": {
                "adverse_events": ["adverse", "event", "reaction"],
                "toxicity": ["toxic", "toxicity"],
                "deaths": ["death", "fatal", "mortality"],
                "risk_benefit": ["risk", "benefit", "ratio"]
            }
        },
        "business": {
            "keywords": ["business", "strategic", "sponsor", "funding", "financial",
                        "priority", "portfolio", "commercial", "merger", "acquisition"],
            "sub_types": {
                "funding": ["fund", "financial", "budget", "resource"],
                "priority_change": ["priority", "strategic", "focus"],
                "merger": ["merger", "acquisition", "reorgan"],
                "portfolio_decision": ["portfolio", "pipeline"],
                "market_factors": ["market", "commercial", "competitive landscape"]
            }
        },
        "regulatory": {
            "keywords": ["fda", "ema", "regulatory", "hold", "compliance", "gcp",
                        "protocol deviation", "audit", "inspection"],
            "sub_types": {
                "fda_hold": ["hold", "fda", "ema", "regulatory"],
                "protocol_issues": ["protocol", "amendment", "deviation"],
                "compliance": ["compliance", "gcp", "violation"],
                "data_integrity": ["data", "integrity", "quality"]
            }
        },
        "operational": {
            "keywords": ["operational", "logistics", "supply", "manufacture", "staff",
                        "site performance", "data management"],
            "sub_types": {
                "site_performance": ["site", "investigator", "performance"],
                "data_quality": ["data", "quality", "collection"],
                "supply_issues": ["supply", "drug", "manufacture", "shortage"],
                "staffing": ["staff", "personnel", "resource"]
            }
        },
        "scientific": {
            "keywords": ["scientific", "biomarker", "mechanism", "hypothesis", "target",
                        "biology", "preclinical"],
            "sub_types": {
                "biomarker_failure": ["biomarker", "marker"],
                "wrong_population": ["population", "patient selection"],
                "mechanism_issue": ["mechanism", "target", "pathway"]
            }
        }
    }

    def __init__(self, db_manager, api_key: Optional[str] = None):
        """Initialize analyzer."""
        self.db = db_manager
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _call_claude(self, prompt: str, max_tokens: int = 4000) -> dict:
        """Call Claude and parse JSON response."""
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text

        # Extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        return json.loads(response_text.strip())

    def _classify_termination_keyword(self, reason: str) -> Tuple[str, str]:
        """Quick keyword-based classification (fallback)."""
        reason_lower = reason.lower()

        for category, data in self.TERMINATION_KEYWORDS.items():
            for keyword in data["keywords"]:
                if keyword in reason_lower:
                    # Find sub-type
                    sub_type = "general"
                    for st_name, st_keywords in data["sub_types"].items():
                        for st_kw in st_keywords:
                            if st_kw in reason_lower:
                                sub_type = st_name
                                break
                    return category, sub_type

        return "other", "unclassified"

    def _get_trial_data(
        self,
        condition: str,
        phase: str,
        limit: int = 500
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """Get completed and terminated trials for analysis."""
        from sqlalchemy import text

        # Get completed trials
        completed_query = text("""
            SELECT nct_id, title, sponsor, enrollment, num_sites,
                   start_date, completion_date, primary_outcomes, eligibility_criteria,
                   julianday(completion_date) - julianday(start_date) as duration_days
            FROM trials
            WHERE (LOWER(conditions) LIKE :condition OR LOWER(therapeutic_area) LIKE :condition)
            AND phase = :phase
            AND status = 'COMPLETED'
            AND enrollment > 0
            ORDER BY completion_date DESC
            LIMIT :limit
        """)

        completed_results = self.db.execute_raw(completed_query.text, {
            "condition": f"%{condition.lower()}%",
            "phase": phase,
            "limit": limit
        })

        completed = []
        for r in completed_results:
            duration_months = r[9] / 30.44 if r[9] and r[9] > 0 else None
            completed.append({
                "nct_id": r[0],
                "title": r[1],
                "sponsor": r[2],
                "enrollment": r[3],
                "num_sites": r[4],
                "start_date": r[5],
                "completion_date": r[6],
                "primary_outcomes": r[7],
                "eligibility_criteria": r[8],
                "duration_months": duration_months
            })

        # Get terminated trials
        terminated_query = text("""
            SELECT nct_id, title, sponsor, enrollment, num_sites,
                   start_date, why_stopped, phase, primary_outcomes, eligibility_criteria
            FROM trials
            WHERE (LOWER(conditions) LIKE :condition OR LOWER(therapeutic_area) LIKE :condition)
            AND phase = :phase
            AND status IN ('TERMINATED', 'WITHDRAWN')
            ORDER BY enrollment DESC
            LIMIT :limit
        """)

        terminated_results = self.db.execute_raw(terminated_query.text, {
            "condition": f"%{condition.lower()}%",
            "phase": phase,
            "limit": limit
        })

        terminated = []
        for r in terminated_results:
            terminated.append({
                "nct_id": r[0],
                "title": r[1],
                "sponsor": r[2],
                "enrollment": r[3],
                "num_sites": r[4],
                "start_date": r[5],
                "why_stopped": r[6],
                "phase": r[7],
                "primary_outcomes": r[8],
                "eligibility_criteria": r[9]
            })

        # Calculate summary stats
        total = len(completed) + len(terminated)
        stats = {
            "total": total,
            "completed": len(completed),
            "terminated": len(terminated),
            "completion_rate": len(completed) / total * 100 if total > 0 else 0,
            "avg_enrollment": sum(t["enrollment"] or 0 for t in completed) / len(completed) if completed else 0,
            "avg_duration": sum(t["duration_months"] or 0 for t in completed if t["duration_months"]) / len([t for t in completed if t["duration_months"]]) if completed else 0
        }

        return completed, terminated, stats

    def _classify_terminations(
        self,
        terminated_trials: List[Dict],
        use_ai: bool = True
    ) -> Dict[str, TerminationCategory]:
        """Classify termination reasons into categories."""
        # First, do keyword-based classification
        categories = defaultdict(lambda: {"count": 0, "examples": [], "sub_types": defaultdict(int)})

        for trial in terminated_trials:
            why_stopped = trial.get("why_stopped", "")
            if not why_stopped:
                categories["unknown"]["count"] += 1
                continue

            category, sub_type = self._classify_termination_keyword(why_stopped)
            categories[category]["count"] += 1
            categories[category]["sub_types"][sub_type] += 1
            if len(categories[category]["examples"]) < 3:
                categories[category]["examples"].append({
                    "nct_id": trial["nct_id"],
                    "reason": why_stopped[:200]
                })

        # If using AI, refine classifications
        if use_ai and terminated_trials:
            try:
                # Sample trials with reasons for AI classification
                trials_with_reasons = [t for t in terminated_trials if t.get("why_stopped")][:30]
                if trials_with_reasons:
                    reasons_for_ai = [
                        {"nct_id": t["nct_id"], "reason": t["why_stopped"][:300]}
                        for t in trials_with_reasons
                    ]

                    prompt = self.TERMINATION_CLASSIFICATION_PROMPT.format(
                        reasons_json=json.dumps(reasons_for_ai, indent=2)
                    )

                    result = self._call_claude(prompt, max_tokens=3000)

                    # Update categories with AI insights
                    if "category_summary" in result:
                        for cat_name, cat_data in result["category_summary"].items():
                            if cat_name in categories:
                                # Add AI-identified patterns
                                categories[cat_name]["ai_patterns"] = cat_data.get("common_patterns", [])

            except Exception as e:
                logger.warning(f"AI classification failed, using keyword-only: {e}")

        # Convert to TerminationCategory objects
        total = sum(c["count"] for c in categories.values())
        result = {}

        mitigation_map = {
            "enrollment_failure": [
                "Expand eligibility criteria",
                "Add more sites in high-prevalence areas",
                "Implement patient identification tools",
                "Consider adaptive enrollment targets"
            ],
            "efficacy": [
                "Ensure robust preclinical data",
                "Consider biomarker-driven patient selection",
                "Plan for interim analyses",
                "Review endpoint selection"
            ],
            "safety": [
                "Implement robust safety monitoring",
                "Consider dose optimization studies first",
                "Establish clear stopping rules",
                "Plan for DSMB reviews"
            ],
            "business": [
                "Secure multi-year funding commitments",
                "Build partnerships for risk sharing",
                "Align with corporate strategy early"
            ],
            "regulatory": [
                "Early engagement with regulators",
                "Robust GCP compliance program",
                "Regular protocol reviews"
            ],
            "operational": [
                "Careful site selection and training",
                "Robust supply chain planning",
                "Experienced CRO partnership"
            ],
            "scientific": [
                "Validate biomarkers prospectively",
                "Ensure mechanism is well-understood",
                "Consider adaptive designs"
            ]
        }

        for cat_name, cat_data in categories.items():
            pct = cat_data["count"] / total * 100 if total > 0 else 0
            risk_level = "high" if pct > 25 else "medium" if pct > 15 else "low"

            result[cat_name] = TerminationCategory(
                name=cat_name.replace("_", " ").title(),
                count=cat_data["count"],
                percentage=pct,
                risk_level=risk_level,
                description=f"{cat_data['count']} trials ({pct:.1f}%)",
                examples=[e["reason"] for e in cat_data["examples"]],
                mitigation_strategies=mitigation_map.get(cat_name, ["Review historical patterns"])
            )

        return result

    def _calculate_risk_score(
        self,
        protocol: Dict,
        stats: Dict,
        termination_categories: Dict[str, TerminationCategory],
        risk_factors: List[RiskFactor]
    ) -> RiskScore:
        """Calculate comprehensive risk score."""
        # Base scores from risk factors
        scores = {
            "enrollment": [],
            "design": [],
            "regulatory": [],
            "operational": [],
            "scientific": [],
            "market": []
        }

        for rf in risk_factors:
            if rf.category in scores:
                scores[rf.category].append(rf.score)

        # Calculate averages
        def avg_or_default(lst, default=30):
            return sum(lst) / len(lst) if lst else default

        enrollment_risk = avg_or_default(scores["enrollment"])
        design_risk = avg_or_default(scores["design"])
        regulatory_risk = avg_or_default(scores["regulatory"])
        operational_risk = avg_or_default(scores["operational"])
        scientific_risk = avg_or_default(scores["scientific"])
        market_risk = avg_or_default(scores["market"])

        # Adjust based on termination patterns
        for cat_name, cat in termination_categories.items():
            if cat.risk_level == "high":
                if "enrollment" in cat_name:
                    enrollment_risk = min(100, enrollment_risk + 15)
                elif "efficacy" in cat_name or "scientific" in cat_name:
                    scientific_risk = min(100, scientific_risk + 15)
                elif "safety" in cat_name:
                    design_risk = min(100, design_risk + 10)
                elif "regulatory" in cat_name:
                    regulatory_risk = min(100, regulatory_risk + 15)

        # Interaction effects
        interactions = []

        # High enrollment risk + low historical success = compounded risk
        if enrollment_risk > 60 and stats["completion_rate"] < 50:
            enrollment_risk = min(100, enrollment_risk * 1.2)
            interactions.append("High enrollment risk compounds with low historical success rate")

        # Large enrollment target + few sites = risk
        if protocol.get("target_enrollment", 0) > 500 and protocol.get("num_sites", 100) < 30:
            enrollment_risk = min(100, enrollment_risk + 20)
            interactions.append("Large enrollment target with limited sites increases risk")

        # Calculate overall
        weights = {
            "enrollment": 0.30,
            "design": 0.15,
            "regulatory": 0.10,
            "operational": 0.15,
            "scientific": 0.20,
            "market": 0.10
        }

        overall = (
            enrollment_risk * weights["enrollment"] +
            design_risk * weights["design"] +
            regulatory_risk * weights["regulatory"] +
            operational_risk * weights["operational"] +
            scientific_risk * weights["scientific"] +
            market_risk * weights["market"]
        )

        # Determine risk level
        if overall > 65:
            risk_level = "high"
        elif overall > 40:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Compare to benchmarks
        avg_risk = 100 - stats["completion_rate"]
        vs_condition = "above" if overall > avg_risk + 10 else "below" if overall < avg_risk - 10 else "at"

        return RiskScore(
            overall_score=overall,
            risk_level=risk_level,
            confidence="medium",
            enrollment_risk=enrollment_risk,
            design_risk=design_risk,
            regulatory_risk=regulatory_risk,
            operational_risk=operational_risk,
            scientific_risk=scientific_risk,
            market_risk=market_risk,
            interaction_effects=interactions,
            vs_condition_average=vs_condition,
            vs_phase_average=vs_condition
        )

    def _get_temporal_risks(self, phase: str) -> List[TemporalRisk]:
        """Get risk patterns by trial phase."""
        return [
            TemporalRisk(
                phase="Startup (Months 1-3)",
                risk_level="medium",
                common_issues=["Site activation delays", "Regulatory amendments", "Supply chain setup"],
                watchpoints=["Site initiation rate", "Protocol amendments", "Drug supply readiness"]
            ),
            TemporalRisk(
                phase="Early Enrollment (Months 4-12)",
                risk_level="high",
                common_issues=["Slow recruitment", "Screen failure rate", "Eligibility challenges"],
                watchpoints=["Enrollment rate vs target", "Screen failure %", "Site performance variance"]
            ),
            TemporalRisk(
                phase="Peak Enrollment (Months 12-24)",
                risk_level="medium",
                common_issues=["Site fatigue", "Competition from other trials", "Protocol deviations"],
                watchpoints=["Enrollment velocity trend", "Major protocol deviations", "Dropout rate"]
            ),
            TemporalRisk(
                phase="Treatment & Follow-up",
                risk_level="medium",
                common_issues=["Patient retention", "Safety signals", "Data quality"],
                watchpoints=["Dropout rate", "SAE frequency", "Missing data %"]
            ),
            TemporalRisk(
                phase="Analysis & Closeout",
                risk_level="low",
                common_issues=["Database lock delays", "Statistical challenges", "Site closeout"],
                watchpoints=["Query resolution time", "Database lock timeline", "Site audit findings"]
            )
        ]

    def analyze(
        self,
        condition: str,
        phase: str,
        target_enrollment: int = 200,
        num_sites: int = 30,
        intervention: str = "",
        endpoint: str = "",
        eligibility_criteria: str = "",
        use_ai_analysis: bool = True
    ) -> RiskAssessment:
        """
        Generate comprehensive risk assessment.

        Args:
            condition: Medical condition
            phase: Trial phase
            target_enrollment: Target number of patients
            num_sites: Number of planned sites
            intervention: Intervention description
            endpoint: Primary endpoint
            eligibility_criteria: Eligibility criteria text
            use_ai_analysis: Whether to use AI for detailed analysis

        Returns:
            Complete RiskAssessment
        """
        logger.info(f"Analyzing risks for {condition} {phase} trial")

        # Get historical data
        completed, terminated, stats = self._get_trial_data(condition, phase)

        # Classify terminations
        termination_categories = self._classify_terminations(terminated, use_ai=use_ai_analysis)

        # Prepare protocol dict
        protocol = {
            "condition": condition,
            "phase": phase,
            "target_enrollment": target_enrollment,
            "num_sites": num_sites,
            "intervention": intervention,
            "endpoint": endpoint,
            "eligibility": eligibility_criteria
        }

        # AI-powered detailed analysis
        risk_factors = []
        protective_factors = []
        root_cause_analyses = []
        top_risks = []
        mitigation_priorities = []
        key_insights = []
        success_probability = stats["completion_rate"]

        if use_ai_analysis and (completed or terminated):
            try:
                # Prepare samples for Claude
                completed_sample = json.dumps([
                    {"nct_id": t["nct_id"], "enrollment": t["enrollment"],
                     "duration": t.get("duration_months"), "sponsor": t["sponsor"][:50] if t["sponsor"] else ""}
                    for t in completed[:10]
                ], indent=2)

                terminated_sample = json.dumps([
                    {"nct_id": t["nct_id"], "enrollment": t["enrollment"],
                     "why_stopped": t["why_stopped"][:150] if t["why_stopped"] else "", "sponsor": t["sponsor"][:50] if t["sponsor"] else ""}
                    for t in terminated[:10]
                ], indent=2)

                top_term_cats = sorted(termination_categories.values(), key=lambda x: -x.count)[:3]
                top_terminations = ", ".join([f"{c.name} ({c.count})" for c in top_term_cats])

                prompt = self.RISK_ANALYSIS_PROMPT.format(
                    condition=condition,
                    phase=phase,
                    target_enrollment=target_enrollment,
                    num_sites=num_sites,
                    intervention=intervention or "Not specified",
                    endpoint=endpoint or "Not specified",
                    eligibility=eligibility_criteria[:500] if eligibility_criteria else "Not specified",
                    similar_count=stats["total"],
                    completion_rate=stats["completion_rate"],
                    top_terminations=top_terminations,
                    avg_enrollment=stats["avg_enrollment"],
                    avg_duration=stats["avg_duration"],
                    terminated_sample=terminated_sample,
                    completed_sample=completed_sample
                )

                result = self._call_claude(prompt, max_tokens=4000)

                # Parse risk factors
                for rf in result.get("risk_factors", []):
                    risk_factors.append(RiskFactor(
                        category=rf.get("category", "operational"),
                        name=rf.get("name", ""),
                        severity=rf.get("severity", "medium"),
                        score=rf.get("score", 50),
                        description=rf.get("description", ""),
                        evidence=rf.get("evidence", ""),
                        mitigation=rf.get("mitigation", ""),
                        confidence=rf.get("confidence", "medium")
                    ))

                # Parse protective factors
                for pf in result.get("protective_factors", []):
                    protective_factors.append(ProtectiveFactor(
                        name=pf.get("name", ""),
                        impact=pf.get("impact", "moderate"),
                        description=pf.get("description", ""),
                        your_protocol_status=pf.get("your_protocol_status", "unknown"),
                        recommendation=pf.get("recommendation", "")
                    ))

                # Parse root cause analyses
                for rca in result.get("root_cause_analyses", []):
                    root_cause_analyses.append(RootCauseAnalysis(
                        category=rca.get("category", ""),
                        root_causes=rca.get("root_causes", []),
                        contributing_factors=[],
                        early_warning_signs=rca.get("early_warning_signs", []),
                        prevention_strategies=rca.get("prevention_strategies", []),
                        case_studies=[]
                    ))

                # Extract other insights
                top_risks = result.get("top_risks", [])
                mitigation_priorities = result.get("mitigation_priorities", [])
                key_insights = result.get("key_insights", [])

                # Update success probability from AI if provided
                if "success_probability" in result:
                    # Blend AI estimate with historical
                    ai_prob = result["success_probability"]
                    success_probability = (success_probability * 0.6 + ai_prob * 0.4)

            except Exception as e:
                logger.error(f"AI risk analysis failed: {e}")
                # Fall back to basic analysis
                key_insights.append("AI-powered analysis unavailable. Using historical patterns only.")

        # Calculate risk score
        risk_score = self._calculate_risk_score(protocol, stats, termination_categories, risk_factors)

        # Get temporal risks
        temporal_risks = self._get_temporal_risks(phase)

        # Build final assessment
        return RiskAssessment(
            condition=condition,
            phase=phase,
            target_enrollment=target_enrollment,
            risk_score=risk_score,
            success_probability=success_probability,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            termination_categories=list(termination_categories.values()),
            root_cause_analyses=root_cause_analyses,
            temporal_risks=temporal_risks,
            similar_trials_analyzed=stats["total"],
            completed_count=stats["completed"],
            terminated_count=stats["terminated"],
            historical_success_rate=stats["completion_rate"],
            top_risks=top_risks,
            mitigation_priorities=mitigation_priorities,
            key_insights=key_insights
        )
