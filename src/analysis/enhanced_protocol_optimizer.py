"""
Enhanced Protocol Optimization Engine

Provides comprehensive protocol analysis including:
1. Design element scoring and benchmarking
2. Phase-specific optimization guidance
3. Regulatory alignment checks
4. Statistical power assessment
5. Timeline feasibility analysis
6. Competitive positioning
7. AI-powered recommendations with evidence
"""

import os
import json
import logging
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DesignElement:
    """Analysis of a single protocol design element."""
    name: str
    category: str  # enrollment, endpoints, eligibility, duration, sites, design
    current_value: str
    benchmark_value: str
    score: float  # 0-100
    assessment: str  # optimal, acceptable, needs_improvement, concerning
    recommendation: Optional[str] = None
    impact_if_changed: Optional[str] = None


@dataclass
class PhaseGuidance:
    """Phase-specific guidance."""
    phase: str
    typical_enrollment: Tuple[int, int]  # (min, max) range
    typical_duration_months: Tuple[int, int]
    typical_sites: Tuple[int, int]
    key_focus_areas: List[str]
    common_pitfalls: List[str]
    regulatory_expectations: List[str]
    success_rate_benchmark: float


@dataclass
class RegulatoryCheck:
    """Regulatory alignment check result."""
    requirement: str
    category: str  # safety, efficacy, documentation, design
    status: str  # aligned, needs_attention, missing
    details: str
    recommendation: Optional[str] = None


@dataclass
class PowerAnalysis:
    """Statistical power assessment."""
    estimated_power: float  # 0-1
    sample_size_adequate: bool
    recommended_sample_size: int
    effect_size_assumption: str
    notes: List[str]


@dataclass
class TimelineAssessment:
    """Timeline feasibility assessment."""
    proposed_duration_months: float
    benchmark_duration_months: float
    feasibility: str  # aggressive, realistic, conservative
    risk_factors: List[str]
    recommendations: List[str]
    milestone_estimates: Dict[str, float]  # milestone -> months


@dataclass
class CompetitivePosition:
    """Competitive positioning analysis."""
    active_competitors: int
    recently_completed: int
    enrollment_competition_level: str  # low, medium, high
    differentiation_factors: List[str]
    competitive_advantages: List[str]
    competitive_risks: List[str]


@dataclass
class Recommendation:
    """A single optimization recommendation."""
    category: str
    priority: str  # high, medium, low
    title: str
    current_state: str
    recommendation: str
    expected_impact: str
    evidence: List[str]
    confidence: float
    implementation_complexity: str  # low, medium, high


@dataclass
class EnhancedOptimizationReport:
    """Comprehensive optimization report."""
    # Overall assessment
    overall_score: int  # 0-100
    readiness_level: str  # ready, needs_minor_changes, needs_major_revision

    # Design analysis
    design_elements: List[DesignElement]
    design_score_by_category: Dict[str, float]

    # Phase-specific
    phase_guidance: PhaseGuidance
    phase_alignment_score: float

    # Regulatory
    regulatory_checks: List[RegulatoryCheck]
    regulatory_score: float

    # Statistical
    power_analysis: PowerAnalysis

    # Timeline
    timeline_assessment: TimelineAssessment

    # Competitive
    competitive_position: CompetitivePosition

    # Recommendations
    recommendations: List[Recommendation]
    total_recommendations: int
    high_priority_count: int

    # Summary
    summary: str
    key_strengths: List[str]
    critical_gaps: List[str]
    estimated_improvement: Dict[str, str]


class EnhancedProtocolOptimizer:
    """Enhanced protocol optimization engine."""

    # Phase-specific benchmarks
    PHASE_BENCHMARKS = {
        "Phase 1": PhaseGuidance(
            phase="Phase 1",
            typical_enrollment=(20, 80),
            typical_duration_months=(6, 18),
            typical_sites=(1, 10),
            key_focus_areas=["Safety", "Tolerability", "PK/PD", "Dose finding"],
            common_pitfalls=["Overly restrictive eligibility", "Too many dose levels", "Insufficient safety monitoring"],
            regulatory_expectations=["IND safety requirements", "Dose escalation rules", "Stopping criteria"],
            success_rate_benchmark=0.63
        ),
        "Phase 2": PhaseGuidance(
            phase="Phase 2",
            typical_enrollment=(100, 300),
            typical_duration_months=(12, 36),
            typical_sites=(10, 50),
            key_focus_areas=["Efficacy signals", "Dose selection", "Safety profile", "Patient selection"],
            common_pitfalls=["Underpowered studies", "Wrong dose selection", "Heterogeneous population"],
            regulatory_expectations=["Preliminary efficacy", "Safety database", "Dose rationale for Phase 3"],
            success_rate_benchmark=0.31
        ),
        "Phase 3": PhaseGuidance(
            phase="Phase 3",
            typical_enrollment=(300, 3000),
            typical_duration_months=(24, 60),
            typical_sites=(50, 300),
            key_focus_areas=["Confirmatory efficacy", "Safety", "Regulatory endpoints", "Subgroup analysis"],
            common_pitfalls=["Enrollment delays", "Protocol amendments", "Endpoint selection issues"],
            regulatory_expectations=["Statistically significant efficacy", "Comprehensive safety", "Benefit-risk profile"],
            success_rate_benchmark=0.58
        ),
        "Phase 4": PhaseGuidance(
            phase="Phase 4",
            typical_enrollment=(500, 10000),
            typical_duration_months=(24, 72),
            typical_sites=(50, 500),
            key_focus_areas=["Real-world evidence", "Long-term safety", "New indications", "Comparative effectiveness"],
            common_pitfalls=["Low enrollment motivation", "Data quality", "Lost to follow-up"],
            regulatory_expectations=["Post-marketing commitments", "Safety surveillance", "Label updates"],
            success_rate_benchmark=0.75
        ),
    }

    # Regulatory requirements by category
    REGULATORY_REQUIREMENTS = {
        "safety": [
            ("Adverse event monitoring plan", "Safety monitoring"),
            ("Data Safety Monitoring Board", "DSMB for phase 2/3"),
            ("Stopping rules defined", "Clear stopping criteria"),
            ("SAE reporting procedures", "Serious AE handling"),
        ],
        "efficacy": [
            ("Primary endpoint clearly defined", "Primary endpoint"),
            ("Statistical analysis plan", "SAP requirement"),
            ("Sample size justification", "Power calculation"),
            ("Control group specified", "Comparator arm"),
        ],
        "documentation": [
            ("Informed consent process", "ICF requirements"),
            ("Protocol version control", "Amendment tracking"),
            ("Data management plan", "eCRF/data handling"),
        ],
        "design": [
            ("Randomization method", "Randomization scheme"),
            ("Blinding procedures", "Blinding approach"),
            ("Visit schedule", "Assessment timing"),
        ],
    }

    def __init__(self, db=None, api_key: Optional[str] = None):
        """Initialize optimizer."""
        self.db = db
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

    def optimize(self, extracted_protocol, similar_trials: List,
                 metrics: Dict, matching_context=None) -> EnhancedOptimizationReport:
        """Generate comprehensive optimization report."""
        logger.info(f"Generating enhanced optimization for {extracted_protocol.condition} {extracted_protocol.phase}")

        # Analyze design elements
        design_elements = self._analyze_design_elements(extracted_protocol, similar_trials, metrics)

        # Get phase guidance
        phase_guidance = self._get_phase_guidance(extracted_protocol.phase)
        phase_alignment = self._assess_phase_alignment(extracted_protocol, phase_guidance)

        # Regulatory checks
        regulatory_checks = self._check_regulatory_alignment(extracted_protocol)
        regulatory_score = self._calculate_regulatory_score(regulatory_checks)

        # Power analysis
        power_analysis = self._estimate_power(extracted_protocol, metrics)

        # Timeline assessment
        timeline = self._assess_timeline(extracted_protocol, metrics, similar_trials)

        # Competitive position
        competitive = self._analyze_competitive_position(similar_trials, metrics)

        # Calculate scores
        design_scores = self._calculate_design_scores(design_elements)
        overall_score = self._calculate_overall_score(
            design_scores, phase_alignment, regulatory_score, power_analysis, timeline
        )

        # Generate AI recommendations
        recommendations = self._generate_recommendations(
            extracted_protocol, similar_trials, metrics, matching_context,
            design_elements, phase_guidance, regulatory_checks, power_analysis, timeline, competitive
        )

        # Identify strengths and gaps
        strengths, gaps = self._identify_strengths_and_gaps(
            design_elements, regulatory_checks, phase_alignment, power_analysis
        )

        # Determine readiness
        readiness = self._determine_readiness(overall_score, gaps)

        # Summary
        summary = self._generate_summary(
            extracted_protocol, overall_score, readiness, len(recommendations),
            sum(1 for r in recommendations if r.priority == "high")
        )

        return EnhancedOptimizationReport(
            overall_score=overall_score,
            readiness_level=readiness,
            design_elements=design_elements,
            design_score_by_category=design_scores,
            phase_guidance=phase_guidance,
            phase_alignment_score=phase_alignment,
            regulatory_checks=regulatory_checks,
            regulatory_score=regulatory_score,
            power_analysis=power_analysis,
            timeline_assessment=timeline,
            competitive_position=competitive,
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            high_priority_count=sum(1 for r in recommendations if r.priority == "high"),
            summary=summary,
            key_strengths=strengths,
            critical_gaps=gaps,
            estimated_improvement=self._estimate_improvements(recommendations)
        )

    def _analyze_design_elements(self, protocol, similar_trials: List,
                                  metrics: Dict) -> List[DesignElement]:
        """Analyze individual design elements against benchmarks."""
        elements = []

        # Enrollment
        avg_enrollment = metrics.get("avg_enrollment", 0)
        if avg_enrollment > 0:
            enrollment_ratio = protocol.target_enrollment / avg_enrollment
            if 0.7 <= enrollment_ratio <= 1.3:
                assessment = "optimal"
                score = 90
            elif 0.5 <= enrollment_ratio <= 1.5:
                assessment = "acceptable"
                score = 70
            else:
                assessment = "needs_improvement"
                score = 50

            elements.append(DesignElement(
                name="Target Enrollment",
                category="enrollment",
                current_value=f"{protocol.target_enrollment:,} patients",
                benchmark_value=f"{avg_enrollment:.0f} (similar trials avg)",
                score=score,
                assessment=assessment,
                recommendation=f"Consider {avg_enrollment:.0f} patients" if score < 70 else None,
                impact_if_changed="+15% enrollment feasibility" if score < 70 else None
            ))

        # Primary Endpoint
        endpoint_score = 75  # Default
        if protocol.primary_endpoint:
            endpoint_lower = protocol.primary_endpoint.lower()
            # Check if using validated/common endpoints
            if any(term in endpoint_lower for term in ["survival", "progression", "response", "remission", "hba1c"]):
                endpoint_score = 90
                endpoint_assessment = "optimal"
            elif any(term in endpoint_lower for term in ["improvement", "reduction", "change"]):
                endpoint_score = 75
                endpoint_assessment = "acceptable"
            else:
                endpoint_score = 60
                endpoint_assessment = "needs_improvement"

            elements.append(DesignElement(
                name="Primary Endpoint",
                category="endpoints",
                current_value=protocol.primary_endpoint[:80] + "..." if len(protocol.primary_endpoint) > 80 else protocol.primary_endpoint,
                benchmark_value="Validated regulatory endpoint recommended",
                score=endpoint_score,
                assessment=endpoint_assessment,
                recommendation="Consider using established regulatory endpoint" if endpoint_score < 75 else None
            ))

        # Eligibility complexity
        total_criteria = len(protocol.key_inclusion or []) + len(protocol.key_exclusion or [])
        if total_criteria <= 10:
            elig_score = 90
            elig_assessment = "optimal"
        elif total_criteria <= 15:
            elig_score = 75
            elig_assessment = "acceptable"
        elif total_criteria <= 20:
            elig_score = 60
            elig_assessment = "needs_improvement"
        else:
            elig_score = 40
            elig_assessment = "concerning"

        elements.append(DesignElement(
            name="Eligibility Complexity",
            category="eligibility",
            current_value=f"{total_criteria} total criteria",
            benchmark_value="10-15 criteria typical",
            score=elig_score,
            assessment=elig_assessment,
            recommendation="Simplify eligibility criteria" if elig_score < 70 else None,
            impact_if_changed="+20% enrollment speed" if elig_score < 70 else None
        ))

        # Study design
        design_score = 80
        design_assessment = "acceptable"
        if protocol.study_type:
            if "randomized" in protocol.study_type.lower():
                design_score = 90
                design_assessment = "optimal"
            elif "single" in protocol.study_type.lower():
                design_score = 65
                design_assessment = "needs_improvement"

        elements.append(DesignElement(
            name="Study Design",
            category="design",
            current_value=protocol.study_type or "Not specified",
            benchmark_value="Randomized controlled trial (gold standard)",
            score=design_score,
            assessment=design_assessment
        ))

        # Comparator
        comparator_score = 70
        if protocol.comparator:
            if "placebo" in protocol.comparator.lower() or "standard" in protocol.comparator.lower():
                comparator_score = 90
                comparator_assessment = "optimal"
            else:
                comparator_score = 75
                comparator_assessment = "acceptable"
        else:
            comparator_score = 50
            comparator_assessment = "needs_improvement"

        elements.append(DesignElement(
            name="Comparator",
            category="design",
            current_value=protocol.comparator or "Not specified",
            benchmark_value="Placebo or standard of care",
            score=comparator_score,
            assessment=comparator_assessment,
            recommendation="Define clear comparator arm" if comparator_score < 70 else None
        ))

        # Duration (estimated from phase)
        phase_guidance = self.PHASE_BENCHMARKS.get(protocol.phase)
        if phase_guidance:
            duration_min, duration_max = phase_guidance.typical_duration_months
            elements.append(DesignElement(
                name="Expected Duration",
                category="duration",
                current_value=f"{metrics.get('avg_duration_months', duration_min + (duration_max - duration_min) / 2):.0f} months (estimated)",
                benchmark_value=f"{duration_min}-{duration_max} months typical for {protocol.phase}",
                score=75,
                assessment="acceptable"
            ))

        return elements

    def _get_phase_guidance(self, phase: str) -> PhaseGuidance:
        """Get phase-specific guidance."""
        # Normalize phase string
        phase_normalized = phase
        for key in self.PHASE_BENCHMARKS:
            if key.lower() in phase.lower() or phase.lower() in key.lower():
                phase_normalized = key
                break

        return self.PHASE_BENCHMARKS.get(phase_normalized, self.PHASE_BENCHMARKS["Phase 2"])

    def _assess_phase_alignment(self, protocol, guidance: PhaseGuidance) -> float:
        """Assess how well protocol aligns with phase expectations."""
        scores = []

        # Enrollment alignment
        target = protocol.target_enrollment or 0
        min_enroll, max_enroll = guidance.typical_enrollment
        if min_enroll <= target <= max_enroll:
            scores.append(100)
        elif target < min_enroll * 0.5 or target > max_enroll * 2:
            scores.append(40)
        else:
            scores.append(70)

        return sum(scores) / len(scores) if scores else 70

    def _check_regulatory_alignment(self, protocol) -> List[RegulatoryCheck]:
        """Check regulatory requirements alignment."""
        checks = []

        # Safety checks
        checks.append(RegulatoryCheck(
            requirement="Adverse event monitoring",
            category="safety",
            status="needs_attention" if not protocol.eligibility_criteria else "aligned",
            details="AE monitoring should be clearly defined in protocol",
            recommendation="Ensure comprehensive AE monitoring plan" if not protocol.eligibility_criteria else None
        ))

        # DSMB for Phase 2/3
        if "2" in protocol.phase or "3" in protocol.phase:
            checks.append(RegulatoryCheck(
                requirement="Data Safety Monitoring Board",
                category="safety",
                status="needs_attention",
                details="DSMB recommended for Phase 2/3 trials",
                recommendation="Consider establishing DSMB"
            ))

        # Primary endpoint
        checks.append(RegulatoryCheck(
            requirement="Primary endpoint definition",
            category="efficacy",
            status="aligned" if protocol.primary_endpoint else "missing",
            details=f"Primary endpoint: {protocol.primary_endpoint[:50] if protocol.primary_endpoint else 'Not defined'}",
            recommendation="Define clear primary endpoint" if not protocol.primary_endpoint else None
        ))

        # Sample size
        checks.append(RegulatoryCheck(
            requirement="Sample size justification",
            category="efficacy",
            status="needs_attention",
            details=f"Target: {protocol.target_enrollment} patients",
            recommendation="Provide power calculation and sample size rationale"
        ))

        # Control group
        checks.append(RegulatoryCheck(
            requirement="Control group specification",
            category="design",
            status="aligned" if protocol.comparator else "missing",
            details=f"Comparator: {protocol.comparator or 'Not specified'}",
            recommendation="Specify control arm" if not protocol.comparator else None
        ))

        # Randomization
        study_type = (protocol.study_type or "").lower()
        checks.append(RegulatoryCheck(
            requirement="Randomization method",
            category="design",
            status="aligned" if "random" in study_type else "needs_attention",
            details=f"Study type: {protocol.study_type or 'Not specified'}",
            recommendation="Consider randomized design" if "random" not in study_type else None
        ))

        return checks

    def _calculate_regulatory_score(self, checks: List[RegulatoryCheck]) -> float:
        """Calculate overall regulatory alignment score."""
        if not checks:
            return 70

        score_map = {"aligned": 100, "needs_attention": 60, "missing": 20}
        scores = [score_map.get(c.status, 50) for c in checks]
        return sum(scores) / len(scores)

    def _estimate_power(self, protocol, metrics: Dict) -> PowerAnalysis:
        """Estimate statistical power based on sample size and design."""
        target = protocol.target_enrollment or 100
        avg_completed = metrics.get("avg_enrollment", 150)

        # Rough power estimation
        if target >= avg_completed:
            power = 0.80
            adequate = True
        elif target >= avg_completed * 0.7:
            power = 0.70
            adequate = True
        else:
            power = 0.60
            adequate = False

        # Recommend based on phase
        phase_targets = {
            "Phase 1": 50,
            "Phase 2": 200,
            "Phase 3": 500,
            "Phase 4": 1000,
        }

        recommended = phase_targets.get(protocol.phase, 200)
        if target < recommended * 0.5:
            power -= 0.1

        notes = []
        if not adequate:
            notes.append("Consider increasing sample size for adequate power")
        if target < avg_completed * 0.7:
            notes.append(f"Historical trials averaged {avg_completed:.0f} patients")

        return PowerAnalysis(
            estimated_power=max(0.5, min(0.95, power)),
            sample_size_adequate=adequate,
            recommended_sample_size=max(target, recommended),
            effect_size_assumption="Moderate effect size assumed",
            notes=notes
        )

    def _assess_timeline(self, protocol, metrics: Dict, similar_trials: List) -> TimelineAssessment:
        """Assess timeline feasibility."""
        avg_duration = metrics.get("avg_duration_months", 24)

        # Get phase benchmark
        guidance = self._get_phase_guidance(protocol.phase)
        min_dur, max_dur = guidance.typical_duration_months

        if avg_duration < min_dur:
            feasibility = "aggressive"
            risk_factors = ["Timeline shorter than typical", "May face enrollment pressure"]
        elif avg_duration > max_dur:
            feasibility = "conservative"
            risk_factors = ["Extended timeline may increase costs", "Patient retention challenges"]
        else:
            feasibility = "realistic"
            risk_factors = []

        recommendations = []
        if feasibility == "aggressive":
            recommendations.append("Consider buffer time for enrollment")
            recommendations.append("Ensure adequate site capacity")

        # Milestone estimates
        milestones = {
            "First patient in": avg_duration * 0.1,
            "50% enrollment": avg_duration * 0.4,
            "Last patient in": avg_duration * 0.7,
            "Database lock": avg_duration * 0.9,
            "Primary analysis": avg_duration,
        }

        return TimelineAssessment(
            proposed_duration_months=avg_duration,
            benchmark_duration_months=(min_dur + max_dur) / 2,
            feasibility=feasibility,
            risk_factors=risk_factors,
            recommendations=recommendations,
            milestone_estimates=milestones
        )

    def _analyze_competitive_position(self, similar_trials: List, metrics: Dict) -> CompetitivePosition:
        """Analyze competitive landscape."""
        active = [t for t in similar_trials if t.status in ["RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]]
        completed = [t for t in similar_trials if t.status == "COMPLETED"]

        active_count = len(active)
        if active_count > 20:
            competition_level = "high"
        elif active_count > 10:
            competition_level = "medium"
        else:
            competition_level = "low"

        advantages = []
        risks = []

        if active_count < 5:
            advantages.append("Limited direct competition for patients")
        else:
            risks.append(f"{active_count} competing trials may affect enrollment")

        if len(completed) > 20:
            advantages.append("Strong historical precedent for success")

        termination_rate = metrics.get("termination_rate", 0)
        if termination_rate > 30:
            risks.append(f"High termination rate ({termination_rate:.0f}%) in similar trials")

        return CompetitivePosition(
            active_competitors=active_count,
            recently_completed=len(completed),
            enrollment_competition_level=competition_level,
            differentiation_factors=["Novel mechanism", "Improved dosing"],  # Placeholder
            competitive_advantages=advantages,
            competitive_risks=risks
        )

    def _calculate_design_scores(self, elements: List[DesignElement]) -> Dict[str, float]:
        """Calculate design scores by category."""
        by_category = defaultdict(list)
        for e in elements:
            by_category[e.category].append(e.score)

        return {cat: sum(scores) / len(scores) for cat, scores in by_category.items()}

    def _calculate_overall_score(self, design_scores: Dict[str, float],
                                  phase_alignment: float,
                                  regulatory_score: float,
                                  power_analysis: PowerAnalysis,
                                  timeline: TimelineAssessment) -> int:
        """Calculate overall protocol score."""
        # Weighted average
        design_avg = sum(design_scores.values()) / len(design_scores) if design_scores else 70

        power_score = power_analysis.estimated_power * 100

        timeline_scores = {"realistic": 90, "conservative": 75, "aggressive": 60}
        timeline_score = timeline_scores.get(timeline.feasibility, 70)

        overall = (
            design_avg * 0.35 +
            phase_alignment * 0.15 +
            regulatory_score * 0.25 +
            power_score * 0.15 +
            timeline_score * 0.10
        )

        return int(min(100, max(0, overall)))

    def _generate_recommendations(self, protocol, similar_trials, metrics, matching_context,
                                   design_elements, phase_guidance, regulatory_checks,
                                   power_analysis, timeline, competitive) -> List[Recommendation]:
        """Generate AI-powered recommendations."""
        recommendations = []

        # Rule-based recommendations first
        # From design elements
        for element in design_elements:
            if element.assessment in ["needs_improvement", "concerning"] and element.recommendation:
                recommendations.append(Recommendation(
                    category=element.category,
                    priority="high" if element.assessment == "concerning" else "medium",
                    title=f"Improve {element.name}",
                    current_state=element.current_value,
                    recommendation=element.recommendation,
                    expected_impact=element.impact_if_changed or "Improved alignment with benchmarks",
                    evidence=[f"Benchmark: {element.benchmark_value}"],
                    confidence=0.8,
                    implementation_complexity="medium"
                ))

        # From regulatory checks
        for check in regulatory_checks:
            if check.status in ["missing", "needs_attention"] and check.recommendation:
                recommendations.append(Recommendation(
                    category="regulatory",
                    priority="high" if check.status == "missing" else "medium",
                    title=check.requirement,
                    current_state=check.details,
                    recommendation=check.recommendation,
                    expected_impact="Improved regulatory alignment",
                    evidence=[f"Regulatory requirement: {check.category}"],
                    confidence=0.9,
                    implementation_complexity="low"
                ))

        # From power analysis
        if not power_analysis.sample_size_adequate:
            recommendations.append(Recommendation(
                category="enrollment",
                priority="high",
                title="Increase Sample Size",
                current_state=f"Target: {protocol.target_enrollment} patients",
                recommendation=f"Consider increasing to {power_analysis.recommended_sample_size} patients",
                expected_impact="Improved statistical power",
                evidence=power_analysis.notes,
                confidence=0.85,
                implementation_complexity="medium"
            ))

        # From timeline
        for rec in timeline.recommendations:
            recommendations.append(Recommendation(
                category="timeline",
                priority="medium",
                title="Timeline Optimization",
                current_state=f"Timeline: {timeline.feasibility}",
                recommendation=rec,
                expected_impact="Reduced timeline risk",
                evidence=[f"Benchmark: {timeline.benchmark_duration_months:.0f} months"],
                confidence=0.7,
                implementation_complexity="low"
            ))

        # From competitive analysis
        for risk in competitive.competitive_risks:
            recommendations.append(Recommendation(
                category="competitive",
                priority="medium",
                title="Address Competitive Risk",
                current_state=risk,
                recommendation="Differentiate trial or optimize enrollment strategy",
                expected_impact="Improved competitive positioning",
                evidence=[f"{competitive.active_competitors} active competitors"],
                confidence=0.6,
                implementation_complexity="medium"
            ))

        # AI-powered detailed recommendations
        ai_recs = self._get_ai_recommendations(protocol, similar_trials, metrics, phase_guidance)
        recommendations.extend(ai_recs)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 3))

        # Deduplicate by title
        seen_titles = set()
        unique_recs = []
        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique_recs.append(rec)

        return unique_recs[:12]  # Limit to top 12

    def _get_ai_recommendations(self, protocol, similar_trials, metrics, phase_guidance) -> List[Recommendation]:
        """Get AI-generated recommendations from Claude."""
        if not self._client and not self.api_key:
            return []

        completed = [t for t in similar_trials if t.status == "COMPLETED"][:5]
        terminated = [t for t in similar_trials if t.status in ["TERMINATED", "WITHDRAWN"]][:5]

        prompt = f"""Analyze this clinical trial protocol and provide 3-5 specific optimization recommendations.

PROTOCOL:
- Condition: {protocol.condition}
- Phase: {protocol.phase}
- Target Enrollment: {protocol.target_enrollment}
- Primary Endpoint: {protocol.primary_endpoint}
- Intervention: {protocol.intervention_type} - {protocol.intervention_name}

BENCHMARKS:
- Similar trials completion rate: {metrics.get('completion_rate', 0):.0f}%
- Average enrollment: {metrics.get('avg_enrollment', 0):.0f}
- Phase typical enrollment: {phase_guidance.typical_enrollment[0]}-{phase_guidance.typical_enrollment[1]}
- Phase success rate: {phase_guidance.success_rate_benchmark*100:.0f}%

HISTORICAL CONTEXT:
- Completed similar trials: {len(completed)}
- Terminated similar trials: {len(terminated)}
- Common pitfalls for {protocol.phase}: {', '.join(phase_guidance.common_pitfalls[:3])}

Provide recommendations in JSON format:
[
  {{
    "category": "eligibility|endpoint|enrollment|design|timeline|sites",
    "priority": "high|medium|low",
    "title": "Short title",
    "current_state": "Current protocol state",
    "recommendation": "Specific actionable recommendation",
    "expected_impact": "Quantified impact estimate",
    "evidence": ["Supporting evidence 1", "Supporting evidence 2"]
  }}
]

Return ONLY the JSON array."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text.strip()

            # Parse JSON
            if response_text.startswith("["):
                data = json.loads(response_text)
            else:
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    return []

            recommendations = []
            for item in data[:5]:
                recommendations.append(Recommendation(
                    category=item.get("category", "general"),
                    priority=item.get("priority", "medium"),
                    title=item.get("title", ""),
                    current_state=item.get("current_state", ""),
                    recommendation=item.get("recommendation", ""),
                    expected_impact=item.get("expected_impact", ""),
                    evidence=item.get("evidence", []),
                    confidence=0.75,
                    implementation_complexity="medium"
                ))

            return recommendations

        except Exception as e:
            logger.error(f"AI recommendations failed: {e}")
            return []

    def _identify_strengths_and_gaps(self, design_elements, regulatory_checks,
                                      phase_alignment, power_analysis) -> Tuple[List[str], List[str]]:
        """Identify key strengths and critical gaps."""
        strengths = []
        gaps = []

        # From design elements
        for element in design_elements:
            if element.assessment == "optimal":
                strengths.append(f"{element.name}: {element.current_value}")
            elif element.assessment in ["concerning", "needs_improvement"]:
                gaps.append(f"{element.name}: {element.recommendation or 'Needs improvement'}")

        # From regulatory
        missing = [c for c in regulatory_checks if c.status == "missing"]
        if missing:
            gaps.append(f"{len(missing)} regulatory requirements need attention")

        aligned = [c for c in regulatory_checks if c.status == "aligned"]
        if len(aligned) >= 4:
            strengths.append("Good regulatory alignment")

        # From power
        if power_analysis.sample_size_adequate:
            strengths.append("Adequate sample size for statistical power")
        else:
            gaps.append("Sample size may be insufficient")

        # Phase alignment
        if phase_alignment >= 80:
            strengths.append(f"Well-aligned with phase requirements")

        return strengths[:5], gaps[:5]

    def _determine_readiness(self, score: int, gaps: List[str]) -> str:
        """Determine protocol readiness level."""
        if score >= 75 and len(gaps) <= 2:
            return "ready"
        elif score >= 60 or len(gaps) <= 4:
            return "needs_minor_changes"
        else:
            return "needs_major_revision"

    def _generate_summary(self, protocol, score: int, readiness: str,
                          total_recs: int, high_priority: int) -> str:
        """Generate executive summary."""
        readiness_text = {
            "ready": "is well-optimized and ready for execution",
            "needs_minor_changes": "is solid but could benefit from minor improvements",
            "needs_major_revision": "needs significant revisions before proceeding"
        }

        return (f"Your {protocol.phase} {protocol.condition} protocol scores {score}/100 and "
                f"{readiness_text.get(readiness, 'needs review')}. "
                f"We identified {total_recs} recommendations, including {high_priority} high-priority items.")

    def _estimate_improvements(self, recommendations: List[Recommendation]) -> Dict[str, str]:
        """Estimate potential improvements from recommendations."""
        improvements = {}

        high_priority = [r for r in recommendations if r.priority == "high"]

        if any(r.category == "enrollment" for r in high_priority):
            improvements["enrollment_speed"] = "+20-30%"

        if any(r.category in ["design", "endpoint"] for r in high_priority):
            improvements["success_probability"] = "+10-15%"

        if any(r.category == "timeline" for r in high_priority):
            improvements["timeline_reduction"] = "2-4 months"

        if any(r.category == "eligibility" for r in high_priority):
            improvements["patient_pool"] = "+15-25%"

        return improvements
