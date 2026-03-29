"""
Protocol Optimization Recommendations Engine

Uses Claude AI + historical trial data to generate specific, evidence-backed
recommendations for improving clinical trial protocols.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single optimization recommendation."""
    category: str  # eligibility, endpoint, enrollment, design, timeline, sites
    priority: str  # high, medium, low
    title: str
    current_state: str
    recommendation: str
    expected_impact: str
    evidence: List[str]  # NCT IDs or stats supporting this
    confidence: float  # 0-1


@dataclass
class OptimizationReport:
    """Complete optimization report for a protocol."""
    overall_score: int  # 0-100, how optimized the protocol already is
    total_recommendations: int
    high_priority_count: int
    recommendations: List[Recommendation]
    summary: str
    estimated_improvement: Dict[str, str]  # e.g., {"enrollment_speed": "+25%", "success_probability": "+15%"}


class ProtocolOptimizer:
    """
    Generates evidence-backed protocol optimization recommendations.
    """

    OPTIMIZATION_PROMPT = """You are an expert clinical trial consultant. Analyze this protocol against historical trial data and provide specific, actionable optimization recommendations.

<user_protocol>
Condition: {condition}
Phase: {phase}
Target Enrollment: {target_enrollment}
Study Design: {study_design}
Intervention: {intervention_type} - {intervention_name}
Mechanism: {intervention_mechanism}
Comparator: {comparator}
Primary Endpoint: {primary_endpoint}
Key Inclusion Criteria: {key_inclusion}
Key Exclusion Criteria: {key_exclusion}
</user_protocol>

<historical_analysis>
Similar Trials Analyzed: {total_similar}
Completion Rate: {completion_rate}%
Termination Rate: {termination_rate}%
Average Enrollment: {avg_enrollment}
Average Duration: {avg_duration_months} months
High-Similarity Trials: {high_similarity_count}

Top Completed Trials (to learn from):
{completed_trials_summary}

Top Terminated Trials (to avoid):
{terminated_trials_summary}

Common Termination Reasons:
{termination_reasons}

Successful Trial Patterns:
{success_patterns}
</historical_analysis>

Generate 6-10 specific optimization recommendations. For EACH recommendation, provide:

1. **Category**: One of: eligibility, endpoint, enrollment, design, timeline, sites
2. **Priority**: high (likely significant impact), medium (moderate impact), low (minor improvement)
3. **Title**: Short descriptive title (5-10 words)
4. **Current State**: What the protocol currently has/does
5. **Recommendation**: Specific change to make
6. **Expected Impact**: Quantified if possible (e.g., "+20% enrollment speed", "reduce termination risk by 30%")
7. **Evidence**: Reference specific NCT IDs or statistics that support this recommendation

Focus on:
- Eligibility criteria that may be too restrictive (compare to successful trials)
- Endpoint selection (what worked in completed trials)
- Enrollment targets (realistic vs. historical averages)
- Study design elements that correlate with success
- Timeline optimization
- Geographic/site considerations

Return as JSON:
{{
    "overall_score": <0-100, how optimized the protocol already is>,
    "summary": "2-3 sentence executive summary",
    "estimated_improvement": {{
        "enrollment_speed": "+X%",
        "success_probability": "+X%",
        "timeline_reduction": "X months"
    }},
    "recommendations": [
        {{
            "category": "eligibility|endpoint|enrollment|design|timeline|sites",
            "priority": "high|medium|low",
            "title": "Short title",
            "current_state": "What protocol currently does",
            "recommendation": "Specific actionable change",
            "expected_impact": "Quantified impact",
            "evidence": ["NCT12345678 achieved X with this approach", "85% of successful trials used Y"]
        }}
    ]
}}

Return ONLY valid JSON."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Anthropic API key."""
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

    def _analyze_termination_reasons(self, terminated_trials: List) -> str:
        """Analyze and categorize termination reasons."""
        if not terminated_trials:
            return "No terminated trials in dataset"

        categories = {
            "Enrollment/Recruitment": 0,
            "Efficacy/Futility": 0,
            "Safety Concerns": 0,
            "Business/Strategic": 0,
            "Other": 0
        }

        for t in terminated_trials:
            reason = (getattr(t, 'why_stopped', '') or '').lower()
            if any(w in reason for w in ["enroll", "recruit", "accrual", "patient"]):
                categories["Enrollment/Recruitment"] += 1
            elif any(w in reason for w in ["efficacy", "futility", "endpoint", "interim"]):
                categories["Efficacy/Futility"] += 1
            elif any(w in reason for w in ["safety", "adverse", "toxicity", "risk"]):
                categories["Safety Concerns"] += 1
            elif any(w in reason for w in ["business", "strategic", "funding", "sponsor"]):
                categories["Business/Strategic"] += 1
            else:
                categories["Other"] += 1

        result = []
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / len(terminated_trials) * 100
                result.append(f"- {cat}: {count} trials ({pct:.0f}%)")

        return "\n".join(result) if result else "Unable to categorize"

    def _extract_success_patterns(self, completed_trials: List) -> str:
        """Extract patterns from successful trials."""
        if not completed_trials:
            return "Insufficient completed trials for pattern analysis"

        patterns = []

        # Enrollment patterns
        enrollments = [t.enrollment for t in completed_trials if t.enrollment]
        if enrollments:
            avg = sum(enrollments) / len(enrollments)
            median = sorted(enrollments)[len(enrollments)//2]
            patterns.append(f"- Enrollment: avg {avg:.0f}, median {median:.0f}")

        # Duration patterns
        durations = [t.duration_months for t in completed_trials if t.duration_months]
        if durations:
            avg = sum(durations) / len(durations)
            patterns.append(f"- Duration: avg {avg:.1f} months")

        # Sites patterns
        sites = [t.num_sites for t in completed_trials if t.num_sites]
        if sites:
            avg = sum(sites) / len(sites)
            patterns.append(f"- Sites: avg {avg:.0f} sites")

        # High similarity patterns
        high_sim = [t for t in completed_trials if t.overall_similarity >= 70]
        if high_sim:
            patterns.append(f"- {len(high_sim)} highly similar trials completed successfully")

        return "\n".join(patterns) if patterns else "Limited pattern data"

    def _format_trials_summary(self, trials: List, limit: int = 5) -> str:
        """Format trial summaries for the prompt."""
        if not trials:
            return "None available"

        summaries = []
        for t in trials[:limit]:
            summary = f"- {t.nct_id}: "
            if t.enrollment:
                summary += f"enrolled {t.enrollment}, "
            if t.duration_months:
                summary += f"{t.duration_months:.0f} months, "
            if t.overall_similarity:
                summary += f"{t.overall_similarity:.0f}% similar"
            if hasattr(t, 'why_stopped') and t.why_stopped:
                summary += f" | Stopped: {t.why_stopped[:50]}"
            summaries.append(summary)

        return "\n".join(summaries)

    def generate_recommendations(
        self,
        extracted_protocol,
        similar_trials: List,
        metrics: Dict,
        matching_context=None
    ) -> OptimizationReport:
        """
        Generate optimization recommendations for a protocol.

        Args:
            extracted_protocol: ExtractedProtocol from protocol_analyzer
            similar_trials: List of SemanticMatch objects
            metrics: Metrics dict from protocol analysis
            matching_context: Optional MatchingContext for additional info

        Returns:
            OptimizationReport with all recommendations
        """
        # Separate completed and terminated trials
        completed = [t for t in similar_trials if t.status == "COMPLETED"]
        terminated = [t for t in similar_trials if t.status in ["TERMINATED", "WITHDRAWN"]]

        # Sort by similarity for best examples
        completed_sorted = sorted(completed, key=lambda x: x.overall_similarity, reverse=True)
        terminated_sorted = sorted(terminated, key=lambda x: x.overall_similarity, reverse=True)

        # Build the prompt
        prompt = self.OPTIMIZATION_PROMPT.format(
            condition=extracted_protocol.condition,
            phase=extracted_protocol.phase,
            target_enrollment=extracted_protocol.target_enrollment,
            study_design=getattr(extracted_protocol, 'study_type', 'interventional'),
            intervention_type=extracted_protocol.intervention_type,
            intervention_name=extracted_protocol.intervention_name or "Not specified",
            intervention_mechanism=getattr(matching_context, 'intervention_mechanism', 'Not specified') if matching_context else "Not specified",
            comparator=extracted_protocol.comparator or "Not specified",
            primary_endpoint=extracted_protocol.primary_endpoint,
            key_inclusion="; ".join(extracted_protocol.key_inclusion[:5]) if extracted_protocol.key_inclusion else "Not specified",
            key_exclusion="; ".join(extracted_protocol.key_exclusion[:5]) if extracted_protocol.key_exclusion else "Not specified",
            total_similar=metrics.get("total_similar", 0),
            completion_rate=metrics.get("completion_rate", 0),
            termination_rate=metrics.get("termination_rate", 0),
            avg_enrollment=metrics.get("avg_enrollment", 0),
            avg_duration_months=metrics.get("avg_duration_months", 0),
            high_similarity_count=metrics.get("high_similarity_count", 0),
            completed_trials_summary=self._format_trials_summary(completed_sorted, 5),
            terminated_trials_summary=self._format_trials_summary(terminated_sorted, 5),
            termination_reasons=self._analyze_termination_reasons(terminated_sorted),
            success_patterns=self._extract_success_patterns(completed_sorted),
        )

        try:
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

            # Build recommendations
            recommendations = []
            for rec in data.get("recommendations", []):
                recommendations.append(Recommendation(
                    category=rec.get("category", "general"),
                    priority=rec.get("priority", "medium"),
                    title=rec.get("title", ""),
                    current_state=rec.get("current_state", ""),
                    recommendation=rec.get("recommendation", ""),
                    expected_impact=rec.get("expected_impact", ""),
                    evidence=rec.get("evidence", []),
                    confidence=0.8  # Default confidence
                ))

            high_priority = len([r for r in recommendations if r.priority == "high"])

            return OptimizationReport(
                overall_score=data.get("overall_score", 50),
                total_recommendations=len(recommendations),
                high_priority_count=high_priority,
                recommendations=recommendations,
                summary=data.get("summary", ""),
                estimated_improvement=data.get("estimated_improvement", {})
            )

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            # Return basic report on error
            return OptimizationReport(
                overall_score=50,
                total_recommendations=0,
                high_priority_count=0,
                recommendations=[],
                summary=f"Unable to generate recommendations: {str(e)}",
                estimated_improvement={}
            )

    def generate_quick_recommendations(
        self,
        extracted_protocol,
        metrics: Dict,
        db_manager
    ) -> List[Recommendation]:
        """
        Generate quick rule-based recommendations without full Claude analysis.
        Useful for immediate feedback while full analysis runs.
        """
        recommendations = []

        # Enrollment vs average
        if metrics.get("avg_enrollment", 0) > 0:
            if extracted_protocol.target_enrollment > metrics["avg_enrollment"] * 1.5:
                recommendations.append(Recommendation(
                    category="enrollment",
                    priority="high",
                    title="High Enrollment Target",
                    current_state=f"Target: {extracted_protocol.target_enrollment} patients",
                    recommendation=f"Consider reducing to ~{int(metrics['avg_enrollment'])} (historical average) or adding more sites",
                    expected_impact="Reduced timeline risk, higher completion probability",
                    evidence=[f"Similar trials averaged {metrics['avg_enrollment']:.0f} patients"],
                    confidence=0.9
                ))
            elif extracted_protocol.target_enrollment < metrics["avg_enrollment"] * 0.5:
                recommendations.append(Recommendation(
                    category="enrollment",
                    priority="medium",
                    title="Low Enrollment Target",
                    current_state=f"Target: {extracted_protocol.target_enrollment} patients",
                    recommendation=f"Consider if sample size provides adequate statistical power",
                    expected_impact="May need larger sample for regulatory requirements",
                    evidence=[f"Similar trials averaged {metrics['avg_enrollment']:.0f} patients"],
                    confidence=0.7
                ))

        # Termination rate check
        if metrics.get("termination_rate", 0) > 30:
            recommendations.append(Recommendation(
                category="design",
                priority="high",
                title="High Termination Risk Area",
                current_state=f"{metrics['termination_rate']:.0f}% of similar trials terminated",
                recommendation="Review terminated trial reasons and adjust protocol accordingly",
                expected_impact="Reduce termination risk by addressing common failure modes",
                evidence=[f"{metrics.get('terminated_count', 0)} similar trials terminated"],
                confidence=0.85
            ))

        # Exclusion criteria count
        if len(extracted_protocol.key_exclusion) > 5:
            recommendations.append(Recommendation(
                category="eligibility",
                priority="medium",
                title="Complex Exclusion Criteria",
                current_state=f"{len(extracted_protocol.key_exclusion)} exclusion criteria",
                recommendation="Review each exclusion criterion for necessity; consider relaxing non-essential criteria",
                expected_impact="+10-20% enrollment speed",
                evidence=["Studies with simpler criteria typically enroll faster"],
                confidence=0.7
            ))

        # Novel design check
        if metrics.get("high_similarity_count", 0) < 5:
            recommendations.append(Recommendation(
                category="design",
                priority="medium",
                title="Novel Study Design",
                current_state=f"Only {metrics.get('high_similarity_count', 0)} highly similar historical trials",
                recommendation="Consider adaptive design or pilot study to mitigate novelty risk",
                expected_impact="Risk mitigation for uncharted territory",
                evidence=["Limited historical precedent increases uncertainty"],
                confidence=0.6
            ))

        return recommendations
