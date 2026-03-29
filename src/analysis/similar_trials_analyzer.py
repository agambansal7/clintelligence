"""
Enhanced Similar Trials & Competitive Intelligence Analyzer

Provides deep analysis of similar trials including:
1. Multi-dimensional similarity breakdown
2. Lesson extraction from completed/terminated trials
3. Competitive threat analysis for active trials
4. Aggregate insights and benchmarks
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SimilarityBreakdown:
    """Detailed breakdown of why a trial is similar."""
    condition_score: float  # 0-100
    intervention_score: float  # 0-100
    endpoint_score: float  # 0-100
    eligibility_score: float  # 0-100
    phase_match: bool
    overall_score: float  # Weighted combination

    condition_details: str = ""
    intervention_details: str = ""
    endpoint_details: str = ""
    eligibility_details: str = ""


@dataclass
class TrialOutcome:
    """Structured outcome information from a trial."""
    primary_endpoint_met: Optional[bool] = None
    effect_size: Optional[str] = None
    p_value: Optional[str] = None
    enrollment_achieved: Optional[float] = None  # % of target
    actual_duration_months: Optional[float] = None
    key_findings: List[str] = field(default_factory=list)


@dataclass
class TrialLesson:
    """Extracted lesson from a trial."""
    category: str  # design, enrollment, endpoint, safety, regulatory
    lesson: str
    actionable_recommendation: str
    confidence: str  # high, medium, low
    source_nct_id: str


@dataclass
class EnhancedTrialMatch:
    """Comprehensive trial match with full intelligence."""
    # Basic info
    nct_id: str
    title: str
    status: str
    phase: str
    sponsor: str
    conditions: str
    interventions: str
    enrollment: int
    enrollment_type: str
    num_sites: int
    start_date: Optional[str]
    completion_date: Optional[str]
    primary_outcomes: str
    eligibility_criteria: str
    why_stopped: Optional[str]

    # Similarity analysis
    similarity: SimilarityBreakdown = None

    # Outcomes (for completed trials)
    outcome: TrialOutcome = None

    # Lessons learned
    lessons: List[TrialLesson] = field(default_factory=list)

    # Competitive analysis (for active trials)
    is_competitor: bool = False
    competition_level: str = ""  # direct, indirect, tangential
    patient_overlap_estimate: str = ""
    competitive_advantage: str = ""
    competitive_disadvantage: str = ""

    # Calculated metrics
    duration_months: Optional[float] = None
    enrollment_rate: Optional[float] = None  # patients per site per month
    recency_score: float = 0.0  # Higher = more recent


@dataclass
class CompetitiveIntel:
    """Competitive landscape intelligence."""
    total_active_competitors: int
    direct_competitors: List[EnhancedTrialMatch]
    indirect_competitors: List[EnhancedTrialMatch]
    total_competing_enrollment: int
    estimated_patient_pool_competition: str
    your_competitive_position: str
    timeline_analysis: str
    key_differentiators: List[str]
    competitive_risks: List[str]


@dataclass
class AggregateInsights:
    """Aggregate insights from all similar trials."""
    total_similar_trials: int
    completed_count: int
    terminated_count: int
    active_count: int

    # Success metrics
    success_rate: float
    avg_enrollment: float
    avg_duration_months: float
    avg_sites: float
    median_enrollment_rate: float

    # Patterns
    common_success_factors: List[str]
    common_failure_reasons: List[Dict[str, Any]]
    endpoint_patterns: Dict[str, int]
    phase_distribution: Dict[str, int]
    sponsor_distribution: Dict[str, int]

    # Benchmarks for user's protocol
    enrollment_benchmark: str
    duration_benchmark: str
    sites_benchmark: str

    # Top insights
    key_insights: List[str]


class SimilarTrialsAnalyzer:
    """
    Comprehensive analyzer for similar trials and competitive intelligence.
    """

    LESSON_EXTRACTION_PROMPT = """Analyze these clinical trials and extract key lessons for someone designing a similar trial.

<user_protocol>
Condition: {condition}
Phase: {phase}
Intervention: {intervention}
Primary Endpoint: {endpoint}
Target Enrollment: {enrollment}
</user_protocol>

<similar_trials>
{trials_json}
</similar_trials>

For each trial, extract:
1. **Key Lessons**: What can be learned from this trial's design, execution, or outcome?
2. **Actionable Recommendations**: Specific actions the user should take based on this trial
3. **Success Factors** (if completed): What contributed to success?
4. **Failure Analysis** (if terminated): Root cause and how to avoid

Return JSON:
{{
    "trial_lessons": [
        {{
            "nct_id": "NCT...",
            "lessons": [
                {{
                    "category": "design|enrollment|endpoint|safety|regulatory|operational",
                    "lesson": "What was learned",
                    "actionable_recommendation": "What to do about it",
                    "confidence": "high|medium|low"
                }}
            ],
            "success_factors": ["factor 1", "factor 2"],
            "failure_analysis": "Why it failed (if applicable)"
        }}
    ],
    "aggregate_insights": [
        "Cross-trial insight 1",
        "Cross-trial insight 2"
    ],
    "top_recommendations": [
        "Most important recommendation 1",
        "Most important recommendation 2",
        "Most important recommendation 3"
    ]
}}

Focus on actionable, specific insights. Avoid generic advice."""

    COMPETITIVE_ANALYSIS_PROMPT = """Analyze these active/recruiting trials as competitors to the user's planned trial.

<user_protocol>
Condition: {condition}
Phase: {phase}
Intervention: {intervention}
Target Enrollment: {enrollment}
Key Eligibility: {eligibility}
</user_protocol>

<active_competitors>
{competitors_json}
</active_competitors>

Analyze:
1. **Patient Overlap**: Will these trials compete for the same patients?
2. **Timeline Competition**: Who will finish first? Impact on recruitment?
3. **Competitive Advantages**: What advantages does user's trial have?
4. **Competitive Risks**: What threats do these competitors pose?

Return JSON:
{{
    "competitor_analysis": [
        {{
            "nct_id": "NCT...",
            "competition_level": "direct|indirect|tangential",
            "patient_overlap_estimate": "high (>70%)|medium (30-70%)|low (<30%)",
            "competitive_advantage": "User's advantage vs this trial",
            "competitive_disadvantage": "This trial's advantage vs user",
            "threat_level": "high|medium|low"
        }}
    ],
    "overall_competitive_position": "favorable|neutral|challenging",
    "timeline_analysis": "Analysis of enrollment race",
    "key_differentiators": ["What makes user's trial unique"],
    "strategic_recommendations": ["How to compete effectively"]
}}"""

    SIMILARITY_ANALYSIS_PROMPT = """Analyze the similarity between the user's protocol and this trial.

<user_protocol>
Condition: {user_condition}
Intervention: {user_intervention}
Phase: {user_phase}
Primary Endpoint: {user_endpoint}
Key Eligibility: {user_eligibility}
</user_protocol>

<trial>
NCT ID: {trial_nct_id}
Condition: {trial_conditions}
Intervention: {trial_interventions}
Phase: {trial_phase}
Primary Endpoint: {trial_endpoints}
Eligibility: {trial_eligibility}
</trial>

Score similarity (0-100) for each dimension:

Return JSON:
{{
    "condition_score": <0-100>,
    "condition_details": "Why this score - what matches/differs",
    "intervention_score": <0-100>,
    "intervention_details": "Why this score - mechanism similarity",
    "endpoint_score": <0-100>,
    "endpoint_details": "Why this score - endpoint alignment",
    "eligibility_score": <0-100>,
    "eligibility_details": "Why this score - population overlap"
}}

Scoring guide:
- 90-100: Near identical
- 70-89: Very similar (same disease, similar approach)
- 50-69: Moderately similar (related but different)
- 30-49: Loosely related
- 0-29: Different"""

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

    def _calculate_duration(self, start_date: str, end_date: str) -> Optional[float]:
        """Calculate duration in months."""
        if not start_date or not end_date:
            return None
        try:
            def parse_date(d):
                for fmt in ["%Y-%m-%d", "%Y-%m", "%B %Y", "%B %d, %Y", "%Y"]:
                    try:
                        return datetime.strptime(d, fmt)
                    except ValueError:
                        continue
                return None

            start = parse_date(start_date)
            end = parse_date(end_date)

            if start and end and end > start:
                return (end - start).days / 30.44
        except Exception:
            pass
        return None

    def _calculate_recency(self, completion_date: str) -> float:
        """Calculate recency score (0-1, higher = more recent)."""
        if not completion_date:
            return 0.5
        try:
            def parse_date(d):
                for fmt in ["%Y-%m-%d", "%Y-%m", "%B %Y", "%Y"]:
                    try:
                        return datetime.strptime(d, fmt)
                    except ValueError:
                        continue
                return None

            end = parse_date(completion_date)
            if end:
                days_ago = (datetime.now() - end).days
                # Trials in last 2 years get high score, older trials lower
                if days_ago < 365:
                    return 1.0
                elif days_ago < 730:
                    return 0.8
                elif days_ago < 1095:
                    return 0.6
                elif days_ago < 1825:
                    return 0.4
                else:
                    return 0.2
        except Exception:
            pass
        return 0.5

    def analyze_similarity_breakdown(
        self,
        user_protocol: Dict[str, Any],
        trial: Dict[str, Any]
    ) -> SimilarityBreakdown:
        """Get detailed similarity breakdown for a single trial using Claude."""
        try:
            prompt = self.SIMILARITY_ANALYSIS_PROMPT.format(
                user_condition=user_protocol.get("condition", ""),
                user_intervention=user_protocol.get("intervention", ""),
                user_phase=user_protocol.get("phase", ""),
                user_endpoint=user_protocol.get("primary_endpoint", ""),
                user_eligibility=user_protocol.get("eligibility", "")[:500],
                trial_nct_id=trial.get("nct_id", ""),
                trial_conditions=trial.get("conditions", ""),
                trial_interventions=trial.get("interventions", ""),
                trial_phase=trial.get("phase", ""),
                trial_endpoints=trial.get("primary_outcomes", "")[:500],
                trial_eligibility=trial.get("eligibility_criteria", "")[:500]
            )

            result = self._call_claude(prompt, max_tokens=1000)

            # Calculate weighted overall score
            condition_score = result.get("condition_score", 50)
            intervention_score = result.get("intervention_score", 50)
            endpoint_score = result.get("endpoint_score", 50)
            eligibility_score = result.get("eligibility_score", 50)

            overall = (
                condition_score * 0.35 +
                intervention_score * 0.30 +
                endpoint_score * 0.20 +
                eligibility_score * 0.15
            )

            phase_match = trial.get("phase") == user_protocol.get("phase")
            if phase_match:
                overall = min(100, overall + 5)

            return SimilarityBreakdown(
                condition_score=condition_score,
                intervention_score=intervention_score,
                endpoint_score=endpoint_score,
                eligibility_score=eligibility_score,
                phase_match=phase_match,
                overall_score=overall,
                condition_details=result.get("condition_details", ""),
                intervention_details=result.get("intervention_details", ""),
                endpoint_details=result.get("endpoint_details", ""),
                eligibility_details=result.get("eligibility_details", "")
            )
        except Exception as e:
            logger.error(f"Similarity breakdown failed: {e}")
            return SimilarityBreakdown(
                condition_score=50, intervention_score=50,
                endpoint_score=50, eligibility_score=50,
                phase_match=False, overall_score=50
            )

    def extract_lessons(
        self,
        user_protocol: Dict[str, Any],
        trials: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, List[TrialLesson]], List[str], List[str]]:
        """Extract lessons from similar trials."""
        if not trials:
            return {}, [], []

        # Prepare trials for Claude
        trials_for_analysis = []
        for t in trials[:15]:  # Limit to 15 for API
            trials_for_analysis.append({
                "nct_id": t.get("nct_id"),
                "title": (t.get("title") or "")[:150],
                "status": t.get("status"),
                "phase": t.get("phase"),
                "conditions": (t.get("conditions") or "")[:200],
                "interventions": (t.get("interventions") or "")[:200],
                "enrollment": t.get("enrollment"),
                "duration_months": t.get("duration_months"),
                "why_stopped": t.get("why_stopped"),
                "primary_outcomes": (t.get("primary_outcomes") or "")[:300],
            })

        prompt = self.LESSON_EXTRACTION_PROMPT.format(
            condition=user_protocol.get("condition", ""),
            phase=user_protocol.get("phase", ""),
            intervention=user_protocol.get("intervention", ""),
            endpoint=user_protocol.get("primary_endpoint", ""),
            enrollment=user_protocol.get("target_enrollment", ""),
            trials_json=json.dumps(trials_for_analysis, indent=2)
        )

        try:
            result = self._call_claude(prompt, max_tokens=4000)

            # Parse lessons per trial
            trial_lessons = {}
            for tl in result.get("trial_lessons", []):
                nct_id = tl.get("nct_id")
                lessons = []
                for l in tl.get("lessons", []):
                    lessons.append(TrialLesson(
                        category=l.get("category", "design"),
                        lesson=l.get("lesson", ""),
                        actionable_recommendation=l.get("actionable_recommendation", ""),
                        confidence=l.get("confidence", "medium"),
                        source_nct_id=nct_id
                    ))
                trial_lessons[nct_id] = lessons

            aggregate_insights = result.get("aggregate_insights", [])
            top_recommendations = result.get("top_recommendations", [])

            return trial_lessons, aggregate_insights, top_recommendations

        except Exception as e:
            logger.error(f"Lesson extraction failed: {e}")
            return {}, [], []

    def analyze_competition(
        self,
        user_protocol: Dict[str, Any],
        active_trials: List[Dict[str, Any]]
    ) -> CompetitiveIntel:
        """Analyze competitive landscape."""
        if not active_trials:
            return CompetitiveIntel(
                total_active_competitors=0,
                direct_competitors=[],
                indirect_competitors=[],
                total_competing_enrollment=0,
                estimated_patient_pool_competition="No active competitors found",
                your_competitive_position="favorable",
                timeline_analysis="No timeline competition",
                key_differentiators=[],
                competitive_risks=[]
            )

        # Prepare for Claude
        competitors_for_analysis = []
        for t in active_trials[:20]:
            competitors_for_analysis.append({
                "nct_id": t.get("nct_id"),
                "title": (t.get("title") or "")[:150],
                "sponsor": t.get("sponsor"),
                "phase": t.get("phase"),
                "conditions": (t.get("conditions") or "")[:200],
                "interventions": (t.get("interventions") or "")[:200],
                "enrollment": t.get("enrollment"),
                "num_sites": t.get("num_sites"),
                "start_date": t.get("start_date"),
                "eligibility": (t.get("eligibility_criteria") or "")[:300],
            })

        prompt = self.COMPETITIVE_ANALYSIS_PROMPT.format(
            condition=user_protocol.get("condition", ""),
            phase=user_protocol.get("phase", ""),
            intervention=user_protocol.get("intervention", ""),
            enrollment=user_protocol.get("target_enrollment", ""),
            eligibility=user_protocol.get("eligibility", "")[:300],
            competitors_json=json.dumps(competitors_for_analysis, indent=2)
        )

        try:
            result = self._call_claude(prompt, max_tokens=3000)

            # Process competitor analysis
            direct = []
            indirect = []
            total_enrollment = 0

            for ca in result.get("competitor_analysis", []):
                nct_id = ca.get("nct_id")
                trial = next((t for t in active_trials if t.get("nct_id") == nct_id), None)
                if not trial:
                    continue

                # Create enhanced match with competitive info
                match = self._create_enhanced_match(trial)
                match.is_competitor = True
                match.competition_level = ca.get("competition_level", "indirect")
                match.patient_overlap_estimate = ca.get("patient_overlap_estimate", "unknown")
                match.competitive_advantage = ca.get("competitive_advantage", "")
                match.competitive_disadvantage = ca.get("competitive_disadvantage", "")

                if ca.get("competition_level") == "direct":
                    direct.append(match)
                else:
                    indirect.append(match)

                total_enrollment += trial.get("enrollment") or 0

            return CompetitiveIntel(
                total_active_competitors=len(active_trials),
                direct_competitors=direct,
                indirect_competitors=indirect,
                total_competing_enrollment=total_enrollment,
                estimated_patient_pool_competition=f"{len(direct)} direct, {len(indirect)} indirect competitors",
                your_competitive_position=result.get("overall_competitive_position", "neutral"),
                timeline_analysis=result.get("timeline_analysis", ""),
                key_differentiators=result.get("key_differentiators", []),
                competitive_risks=result.get("strategic_recommendations", [])
            )

        except Exception as e:
            logger.error(f"Competitive analysis failed: {e}")
            return CompetitiveIntel(
                total_active_competitors=len(active_trials),
                direct_competitors=[],
                indirect_competitors=[],
                total_competing_enrollment=sum(t.get("enrollment") or 0 for t in active_trials),
                estimated_patient_pool_competition="Analysis unavailable",
                your_competitive_position="unknown",
                timeline_analysis="",
                key_differentiators=[],
                competitive_risks=[]
            )

    def _create_enhanced_match(self, trial: Dict[str, Any]) -> EnhancedTrialMatch:
        """Create EnhancedTrialMatch from trial dict."""
        duration = self._calculate_duration(
            trial.get("start_date"),
            trial.get("completion_date")
        )

        enrollment = trial.get("enrollment") or 0
        num_sites = trial.get("num_sites") or 1

        # Calculate enrollment rate if we have the data
        enrollment_rate = None
        if duration and duration > 0 and enrollment > 0:
            enrollment_rate = enrollment / num_sites / duration

        return EnhancedTrialMatch(
            nct_id=trial.get("nct_id", ""),
            title=trial.get("title", ""),
            status=trial.get("status", ""),
            phase=trial.get("phase", ""),
            sponsor=trial.get("sponsor", ""),
            conditions=trial.get("conditions", ""),
            interventions=trial.get("interventions", ""),
            enrollment=enrollment,
            enrollment_type=trial.get("enrollment_type", ""),
            num_sites=num_sites,
            start_date=trial.get("start_date"),
            completion_date=trial.get("completion_date"),
            primary_outcomes=trial.get("primary_outcomes", ""),
            eligibility_criteria=trial.get("eligibility_criteria", ""),
            why_stopped=trial.get("why_stopped"),
            duration_months=duration,
            enrollment_rate=enrollment_rate,
            recency_score=self._calculate_recency(trial.get("completion_date"))
        )

    def generate_aggregate_insights(
        self,
        user_protocol: Dict[str, Any],
        all_matches: List[EnhancedTrialMatch]
    ) -> AggregateInsights:
        """Generate aggregate insights from all similar trials."""
        if not all_matches:
            return AggregateInsights(
                total_similar_trials=0,
                completed_count=0,
                terminated_count=0,
                active_count=0,
                success_rate=0,
                avg_enrollment=0,
                avg_duration_months=0,
                avg_sites=0,
                median_enrollment_rate=0,
                common_success_factors=[],
                common_failure_reasons=[],
                endpoint_patterns={},
                phase_distribution={},
                sponsor_distribution={},
                enrollment_benchmark="",
                duration_benchmark="",
                sites_benchmark="",
                key_insights=[]
            )

        # Calculate statistics
        completed = [m for m in all_matches if m.status == "COMPLETED"]
        terminated = [m for m in all_matches if m.status in ["TERMINATED", "WITHDRAWN"]]
        active = [m for m in all_matches if m.status in ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]]

        total = len(all_matches)
        completed_count = len(completed)
        terminated_count = len(terminated)
        active_count = len(active)

        success_rate = (completed_count / (completed_count + terminated_count) * 100) if (completed_count + terminated_count) > 0 else 0

        enrollments = [m.enrollment for m in all_matches if m.enrollment and m.enrollment > 0]
        durations = [m.duration_months for m in all_matches if m.duration_months and m.duration_months > 0]
        sites = [m.num_sites for m in all_matches if m.num_sites and m.num_sites > 0]
        rates = [m.enrollment_rate for m in all_matches if m.enrollment_rate and m.enrollment_rate > 0]

        avg_enrollment = sum(enrollments) / len(enrollments) if enrollments else 0
        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_sites = sum(sites) / len(sites) if sites else 0

        # Median enrollment rate
        median_rate = 0
        if rates:
            sorted_rates = sorted(rates)
            mid = len(sorted_rates) // 2
            median_rate = sorted_rates[mid]

        # Phase distribution
        phase_dist = defaultdict(int)
        for m in all_matches:
            if m.phase:
                phase_dist[m.phase] += 1

        # Sponsor distribution (top 10)
        sponsor_dist = defaultdict(int)
        for m in all_matches:
            if m.sponsor:
                sponsor_dist[m.sponsor] += 1
        sponsor_dist = dict(sorted(sponsor_dist.items(), key=lambda x: x[1], reverse=True)[:10])

        # Failure reasons
        failure_reasons = defaultdict(int)
        for m in terminated:
            if m.why_stopped:
                why = m.why_stopped.lower()
                if any(w in why for w in ["enroll", "recruit", "accrual"]):
                    failure_reasons["Enrollment Issues"] += 1
                elif any(w in why for w in ["efficacy", "futility", "endpoint"]):
                    failure_reasons["Efficacy/Futility"] += 1
                elif any(w in why for w in ["safety", "adverse", "toxicity"]):
                    failure_reasons["Safety Concerns"] += 1
                elif any(w in why for w in ["business", "strategic", "funding", "sponsor"]):
                    failure_reasons["Business/Strategic"] += 1
                else:
                    failure_reasons["Other"] += 1

        # Benchmarks
        user_enrollment = user_protocol.get("target_enrollment", 0)
        enrollment_benchmark = ""
        if user_enrollment and avg_enrollment > 0:
            ratio = user_enrollment / avg_enrollment
            if ratio < 0.5:
                enrollment_benchmark = f"Your target ({user_enrollment}) is well below average ({avg_enrollment:.0f}) - lower risk"
            elif ratio < 0.8:
                enrollment_benchmark = f"Your target ({user_enrollment}) is below average ({avg_enrollment:.0f}) - manageable"
            elif ratio < 1.2:
                enrollment_benchmark = f"Your target ({user_enrollment}) is near average ({avg_enrollment:.0f}) - typical"
            elif ratio < 1.5:
                enrollment_benchmark = f"Your target ({user_enrollment}) is above average ({avg_enrollment:.0f}) - challenging"
            else:
                enrollment_benchmark = f"Your target ({user_enrollment}) is well above average ({avg_enrollment:.0f}) - high risk"

        duration_benchmark = f"Similar trials average {avg_duration:.1f} months" if avg_duration else ""
        sites_benchmark = f"Similar trials average {avg_sites:.0f} sites" if avg_sites else ""

        # Key insights
        key_insights = []
        if success_rate > 70:
            key_insights.append(f"High success rate ({success_rate:.0f}%) in similar trials - favorable indication")
        elif success_rate < 50:
            key_insights.append(f"Low success rate ({success_rate:.0f}%) in similar trials - careful planning needed")

        if failure_reasons:
            top_failure = max(failure_reasons.items(), key=lambda x: x[1])
            key_insights.append(f"Most common failure reason: {top_failure[0]} ({top_failure[1]} trials)")

        if median_rate > 0:
            key_insights.append(f"Typical enrollment rate: {median_rate:.2f} patients/site/month")

        return AggregateInsights(
            total_similar_trials=total,
            completed_count=completed_count,
            terminated_count=terminated_count,
            active_count=active_count,
            success_rate=success_rate,
            avg_enrollment=avg_enrollment,
            avg_duration_months=avg_duration,
            avg_sites=avg_sites,
            median_enrollment_rate=median_rate,
            common_success_factors=[],  # Populated by lesson extraction
            common_failure_reasons=[{"reason": k, "count": v} for k, v in failure_reasons.items()],
            endpoint_patterns={},
            phase_distribution=dict(phase_dist),
            sponsor_distribution=sponsor_dist,
            enrollment_benchmark=enrollment_benchmark,
            duration_benchmark=duration_benchmark,
            sites_benchmark=sites_benchmark,
            key_insights=key_insights
        )

    def analyze(
        self,
        user_protocol: Dict[str, Any],
        similar_trials: List[Dict[str, Any]],
        analyze_top_n_similarity: int = 10,
        extract_lessons: bool = True,
        analyze_competition: bool = True
    ) -> Dict[str, Any]:
        """
        Main entry point: Comprehensive analysis of similar trials.

        Args:
            user_protocol: Extracted protocol with condition, intervention, phase, etc.
            similar_trials: List of similar trials from hybrid matcher
            analyze_top_n_similarity: Number of trials to get detailed similarity breakdown
            extract_lessons: Whether to extract lessons using Claude
            analyze_competition: Whether to analyze competitive landscape

        Returns:
            Dict with enhanced_matches, competitive_intel, aggregate_insights,
            trial_lessons, top_recommendations
        """
        logger.info(f"Analyzing {len(similar_trials)} similar trials...")

        # Convert to enhanced matches
        enhanced_matches = []
        for trial in similar_trials:
            match = self._create_enhanced_match(trial)
            enhanced_matches.append(match)

        # Sort by recency + any existing similarity score
        enhanced_matches.sort(
            key=lambda x: (x.recency_score * 0.3 + 0.7),  # Weight recency
            reverse=True
        )

        # Get detailed similarity breakdown for top N
        logger.info(f"Getting detailed similarity for top {analyze_top_n_similarity} trials...")
        for match in enhanced_matches[:analyze_top_n_similarity]:
            trial_dict = next(
                (t for t in similar_trials if t.get("nct_id") == match.nct_id),
                {}
            )
            match.similarity = self.analyze_similarity_breakdown(user_protocol, trial_dict)

        # Re-sort by detailed similarity if available
        def sort_key(m):
            if m.similarity:
                return m.similarity.overall_score
            return 50  # Default

        enhanced_matches.sort(key=sort_key, reverse=True)

        # Separate completed, terminated, and active
        completed = [m for m in enhanced_matches if m.status == "COMPLETED"]
        terminated = [m for m in enhanced_matches if m.status in ["TERMINATED", "WITHDRAWN"]]
        active = [m for m in enhanced_matches if m.status in ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]]

        # Extract lessons
        trial_lessons = {}
        aggregate_lesson_insights = []
        top_recommendations = []

        if extract_lessons and (completed or terminated):
            logger.info("Extracting lessons from trials...")
            trials_for_lessons = [
                next((t for t in similar_trials if t.get("nct_id") == m.nct_id), {})
                for m in (completed[:10] + terminated[:5])
            ]
            trial_lessons, aggregate_lesson_insights, top_recommendations = self.extract_lessons(
                user_protocol, trials_for_lessons
            )

            # Attach lessons to matches
            for match in enhanced_matches:
                if match.nct_id in trial_lessons:
                    match.lessons = trial_lessons[match.nct_id]

        # Competitive analysis
        competitive_intel = None
        if analyze_competition and active:
            logger.info("Analyzing competitive landscape...")
            active_dicts = [
                next((t for t in similar_trials if t.get("nct_id") == m.nct_id), {})
                for m in active[:20]
            ]
            competitive_intel = self.analyze_competition(user_protocol, active_dicts)

        # Generate aggregate insights
        logger.info("Generating aggregate insights...")
        aggregate_insights = self.generate_aggregate_insights(user_protocol, enhanced_matches)

        # Add lesson insights to aggregate
        aggregate_insights.common_success_factors = aggregate_lesson_insights

        return {
            "enhanced_matches": enhanced_matches,
            "completed_trials": completed,
            "terminated_trials": terminated,
            "active_trials": active,
            "competitive_intel": competitive_intel,
            "aggregate_insights": aggregate_insights,
            "trial_lessons": trial_lessons,
            "top_recommendations": top_recommendations
        }
