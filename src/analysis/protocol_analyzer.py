"""
Protocol Analyzer using Claude API

Uses Anthropic's Claude to understand clinical trial protocols and extract
structured information for matching against historical trials.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedProtocol:
    """Structured data extracted from a protocol."""
    # Core trial info
    condition: str
    phase: str
    target_enrollment: int
    study_duration_months: Optional[int]

    # Design details
    study_type: str  # interventional, observational
    intervention_type: str  # drug, device, biological, behavioral
    intervention_name: Optional[str]
    comparator: Optional[str]  # placebo, active comparator, etc.

    # Endpoints
    primary_endpoint: str
    primary_endpoint_timeframe: Optional[str]
    secondary_endpoints: List[str] = field(default_factory=list)

    # Eligibility
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    sex: str = "All"
    key_inclusion: List[str] = field(default_factory=list)
    key_exclusion: List[str] = field(default_factory=list)

    # Risk factors identified by Claude
    identified_risks: List[str] = field(default_factory=list)
    complexity_score: int = 50  # 0-100

    # Original text for reference
    original_text: str = ""
    extraction_confidence: float = 0.0


class ProtocolAnalyzer:
    """Analyzes clinical trial protocols using Claude API."""

    EXTRACTION_PROMPT = """You are an expert clinical trial analyst. Analyze the following clinical trial protocol and extract structured information.

<protocol>
{protocol_text}
</protocol>

Extract the following information and return as JSON:

{{
    "condition": "Primary condition/indication being studied (e.g., 'Type 2 Diabetes', 'Non-Small Cell Lung Cancer')",
    "phase": "Trial phase (PHASE1, PHASE2, PHASE3, PHASE4, or NA)",
    "target_enrollment": <integer>,
    "study_duration_months": <integer or null>,
    "study_type": "interventional or observational",
    "intervention_type": "drug, device, biological, behavioral, or other",
    "intervention_name": "Name of drug/device being tested or null",
    "comparator": "placebo, active comparator, standard of care, or null",
    "primary_endpoint": "Primary efficacy endpoint",
    "primary_endpoint_timeframe": "Timeframe for primary endpoint or null",
    "secondary_endpoints": ["list of secondary endpoints"],
    "min_age": <integer or null>,
    "max_age": <integer or null>,
    "sex": "All, Male, or Female",
    "key_inclusion": ["top 3-5 key inclusion criteria"],
    "key_exclusion": ["top 3-5 key exclusion criteria"],
    "identified_risks": ["List of potential risks/challenges you identify in this protocol design"],
    "complexity_score": <0-100, where 100 is most complex>,
    "extraction_confidence": <0.0-1.0, your confidence in the extraction>
}}

Be precise and extract only what is explicitly stated or can be reasonably inferred. If information is not available, use null for optional fields.

Return ONLY valid JSON, no other text."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Anthropic API key."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set. Please set the environment variable or pass api_key to ProtocolAnalyzer.")

            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install the anthropic package: pip install anthropic")

        return self._client

    def extract_protocol_info(self, protocol_text: str) -> ExtractedProtocol:
        """
        Extract structured information from protocol text using Claude.

        Args:
            protocol_text: Raw protocol text (synopsis, eligibility, etc.)

        Returns:
            ExtractedProtocol with structured information
        """
        if not protocol_text or len(protocol_text.strip()) < 50:
            raise ValueError("Protocol text is too short. Please provide more details.")

        prompt = self.EXTRACTION_PROMPT.format(protocol_text=protocol_text)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = message.content[0].text

            # Parse JSON response
            # Handle potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            data = json.loads(response_text.strip())

            return ExtractedProtocol(
                condition=data.get("condition", "Unknown"),
                phase=self._normalize_phase(data.get("phase", "NA")),
                target_enrollment=data.get("target_enrollment", 100),
                study_duration_months=data.get("study_duration_months"),
                study_type=data.get("study_type", "interventional"),
                intervention_type=data.get("intervention_type", "drug"),
                intervention_name=data.get("intervention_name"),
                comparator=data.get("comparator"),
                primary_endpoint=data.get("primary_endpoint", ""),
                primary_endpoint_timeframe=data.get("primary_endpoint_timeframe"),
                secondary_endpoints=data.get("secondary_endpoints", []),
                min_age=data.get("min_age"),
                max_age=data.get("max_age"),
                sex=data.get("sex", "All"),
                key_inclusion=data.get("key_inclusion", []),
                key_exclusion=data.get("key_exclusion", []),
                identified_risks=data.get("identified_risks", []),
                complexity_score=data.get("complexity_score", 50),
                original_text=protocol_text,
                extraction_confidence=data.get("extraction_confidence", 0.8)
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            raise ValueError(f"Failed to parse protocol analysis: {e}")
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            raise

    def _normalize_phase(self, phase: str) -> str:
        """Normalize phase to database format."""
        if not phase:
            return "NA"

        phase = phase.upper().replace(" ", "").replace("-", "")

        mappings = {
            "PHASE1": "PHASE1",
            "PHASEI": "PHASE1",
            "1": "PHASE1",
            "PHASE2": "PHASE2",
            "PHASEII": "PHASE2",
            "2": "PHASE2",
            "PHASE3": "PHASE3",
            "PHASEIII": "PHASE3",
            "3": "PHASE3",
            "PHASE4": "PHASE4",
            "PHASEIV": "PHASE4",
            "4": "PHASE4",
            "PHASE1/PHASE2": "PHASE1/PHASE2",
            "PHASE2/PHASE3": "PHASE2/PHASE3",
        }

        return mappings.get(phase, "NA")

    def analyze_and_match(
        self,
        protocol_text: str,
        db_manager,
        include_site_recommendations: bool = True,
        min_similarity: int = 40,
        max_candidates: int = 500,
        semantic_rank_top_n: int = 100,
        use_hybrid_matching: bool = True
    ) -> Dict[str, Any]:
        """
        Full analysis: extract protocol info, match trials using hybrid approach.

        Uses hybrid matching (vector + keyword + ontology + eligibility + Claude)
        for best results. Falls back to semantic-only if vector store not ready.

        Args:
            protocol_text: Raw protocol text
            db_manager: Database manager instance
            include_site_recommendations: Whether to include site recommendations
            min_similarity: Minimum similarity score (0-100) to include trials
            max_candidates: Maximum candidates to retrieve for matching
            semantic_rank_top_n: Number of top candidates to send to Claude for ranking
            use_hybrid_matching: Whether to use hybrid matching (vector + keyword + Claude)

        Returns:
            Comprehensive analysis results
        """
        # Step 1: Extract structured info (for display purposes)
        extracted = self.extract_protocol_info(protocol_text)

        # Step 2: Try hybrid matching first (vector + keyword + ontology + Claude)
        matching_context = None
        similar_trials = []

        if use_hybrid_matching:
            try:
                from src.analysis.hybrid_matcher import HybridTrialMatcher
                from src.analysis.vector_store import get_vector_store

                vector_store = get_vector_store()
                use_vector = vector_store.is_initialized()

                logger.info(f"Using hybrid matcher (vector_search={use_vector})")

                hybrid_matcher = HybridTrialMatcher(db_manager, self.api_key)
                query, hybrid_matches = hybrid_matcher.find_similar_trials(
                    protocol_text=protocol_text,
                    min_similarity=min_similarity,
                    max_results=max_candidates,
                    use_vector_search=use_vector,
                    use_claude_reranking=True
                )

                # Convert HybridMatch to SemanticMatch-like objects for compatibility
                from src.analysis.semantic_matcher import SemanticMatch

                similar_trials = []
                for m in hybrid_matches:
                    similar_trials.append(SemanticMatch(
                        nct_id=m.nct_id,
                        title=m.title,
                        status=m.status,
                        phase=m.phase,
                        sponsor=m.sponsor,
                        enrollment=m.enrollment,
                        num_sites=m.num_sites,
                        duration_months=m.duration_months,
                        completion_date=m.completion_date,
                        why_stopped=m.why_stopped,
                        overall_similarity=m.overall_similarity,
                        condition_similarity=m.keyword_score,
                        endpoint_similarity=m.vector_score * 100,
                        eligibility_similarity=m.eligibility_score,
                        design_similarity=m.keyword_score,
                        intervention_similarity=m.vector_score * 100,
                        relevance_explanation=m.relevance_explanation,
                        key_similarities=m.key_similarities,
                        key_differences=m.key_differences,
                        lessons=m.lessons,
                    ))

                # Create matching context from query
                from src.analysis.semantic_matcher import MatchingContext
                matching_context = MatchingContext(
                    condition=query.condition,
                    condition_synonyms=query.condition_synonyms,
                    phase=query.phase,
                    target_enrollment=query.target_enrollment,
                    primary_endpoint=query.primary_endpoint,
                    endpoint_type="efficacy",
                    key_inclusion=[],
                    key_exclusion=[],
                    intervention_type=query.intervention_type,
                    intervention_name=query.intervention_name,
                    intervention_mechanism=query.intervention_mechanism,
                    comparator=None,
                    study_design="",
                    therapeutic_area=query.therapeutic_area,
                )

                logger.info(f"Hybrid matching found {len(similar_trials)} trials")

            except Exception as e:
                logger.warning(f"Hybrid matching failed, falling back to semantic matcher: {e}")
                use_hybrid_matching = False

        # Fallback to original semantic matcher
        if not use_hybrid_matching or not similar_trials:
            from src.analysis.semantic_matcher import SemanticTrialMatcher

            matcher = SemanticTrialMatcher(db_manager, self.api_key)
            matching_context, similar_trials = matcher.find_similar_trials(
                protocol_text=protocol_text,
                min_similarity=min_similarity,
                max_candidates=max_candidates,
                semantic_rank_top_n=semantic_rank_top_n
            )

        # Step 3: Calculate metrics from semantic matches
        completed = [t for t in similar_trials if t.status == "COMPLETED"]
        terminated = [t for t in similar_trials if t.status in ["TERMINATED", "WITHDRAWN"]]
        recruiting = [t for t in similar_trials if t.status == "RECRUITING"]

        metrics = {
            "total_similar": len(similar_trials),
            "completed_count": len(completed),
            "terminated_count": len(terminated),
            "recruiting_count": len(recruiting),
            "completion_rate": len(completed) / len(similar_trials) * 100 if similar_trials else 0,
            "termination_rate": len(terminated) / len(similar_trials) * 100 if similar_trials else 0,
        }

        # Enrollment metrics
        if similar_trials:
            enrollments = [t.enrollment for t in similar_trials if t.enrollment]
            metrics["avg_enrollment"] = sum(enrollments) / len(enrollments) if enrollments else 0
            metrics["median_enrollment"] = sorted(enrollments)[len(enrollments)//2] if enrollments else 0

        # Duration metrics
        if completed:
            durations = [t.duration_months for t in completed if t.duration_months]
            metrics["avg_duration_months"] = sum(durations) / len(durations) if durations else 0
            metrics["median_duration_months"] = sorted(durations)[len(durations)//2] if durations else 0

        # Similarity metrics
        if similar_trials:
            metrics["avg_similarity"] = sum(t.overall_similarity for t in similar_trials) / len(similar_trials)
            metrics["high_similarity_count"] = len([t for t in similar_trials if t.overall_similarity >= 70])

        # Step 4: Risk assessment (using semantic match data)
        risk_assessment = self._assess_risks_semantic(extracted, metrics, terminated, similar_trials)

        # Step 5: Enrollment projection
        enrollment_projection = self._project_enrollment_semantic(extracted, completed)

        # Step 6: Site recommendations (optional)
        site_recommendations = []
        if include_site_recommendations:
            site_recommendations = self._get_site_recommendations(extracted, db_manager)

        # Step 7: Extract lessons from terminated trials (with Claude insights)
        termination_lessons = self._extract_termination_lessons_semantic(terminated)

        return {
            "extracted_protocol": extracted,
            "matching_context": matching_context,
            "similar_trials": similar_trials,
            "metrics": metrics,
            "risk_assessment": risk_assessment,
            "enrollment_projection": enrollment_projection,
            "site_recommendations": site_recommendations,
            "terminated_trial_lessons": termination_lessons,
        }

    def _assess_risks_semantic(
        self,
        extracted: ExtractedProtocol,
        metrics: Dict,
        terminated: List,
        all_matches: List
    ) -> Dict[str, Any]:
        """Assess risks based on semantic matching results."""
        risks = []
        risk_score = 0

        # High termination rate
        if metrics.get("termination_rate", 0) > 25:
            risks.append({
                "category": "Historical Performance",
                "risk": f"High termination rate ({metrics['termination_rate']:.0f}%) in similar trials",
                "severity": "high",
                "recommendation": "Review terminated trial reasons carefully"
            })
            risk_score += 25

        # Low similarity matches (if few high-similarity trials found)
        high_sim_count = metrics.get("high_similarity_count", 0)
        if high_sim_count < 5 and len(all_matches) > 0:
            risks.append({
                "category": "Novel Design",
                "risk": f"Only {high_sim_count} highly similar trials found - this may be a novel design",
                "severity": "medium",
                "recommendation": "Limited historical precedent - consider pilot study or adaptive design"
            })
            risk_score += 15

        # Enrollment challenges
        if extracted.target_enrollment > metrics.get("avg_enrollment", 0) * 1.5 and metrics.get("avg_enrollment", 0) > 0:
            risks.append({
                "category": "Enrollment",
                "risk": f"Target enrollment ({extracted.target_enrollment}) is significantly above average ({metrics.get('avg_enrollment', 0):.0f})",
                "severity": "medium",
                "recommendation": "Consider adding more sites or extending timeline"
            })
            risk_score += 15

        # Complex eligibility
        if len(extracted.key_exclusion) > 5:
            risks.append({
                "category": "Eligibility",
                "risk": "Many exclusion criteria may limit patient pool",
                "severity": "medium",
                "recommendation": "Review if all exclusion criteria are necessary"
            })
            risk_score += 10

        # Claude-identified risks from protocol
        for risk in extracted.identified_risks[:3]:
            risks.append({
                "category": "Protocol Design",
                "risk": risk,
                "severity": "medium",
                "recommendation": "Review protocol design"
            })
            risk_score += 10

        # Add insights from similar terminated trials
        if terminated:
            common_reasons = {}
            for t in terminated:
                if t.why_stopped:
                    reason = t.why_stopped.lower()
                    if "enrollment" in reason or "recruit" in reason:
                        common_reasons["enrollment"] = common_reasons.get("enrollment", 0) + 1
                    elif "efficacy" in reason or "futility" in reason:
                        common_reasons["efficacy"] = common_reasons.get("efficacy", 0) + 1
                    elif "safety" in reason or "adverse" in reason:
                        common_reasons["safety"] = common_reasons.get("safety", 0) + 1

            if common_reasons:
                top_reason = max(common_reasons, key=common_reasons.get)
                risks.append({
                    "category": "Historical Patterns",
                    "risk": f"Similar trials commonly terminated due to {top_reason} issues ({common_reasons[top_reason]} trials)",
                    "severity": "high" if common_reasons[top_reason] >= 3 else "medium",
                    "recommendation": f"Pay special attention to {top_reason} planning"
                })
                risk_score += 15

        return {
            "overall_risk_score": min(risk_score, 100),
            "risk_level": "high" if risk_score > 60 else "medium" if risk_score > 30 else "low",
            "risks": risks,
        }

    def _project_enrollment_semantic(
        self,
        extracted: ExtractedProtocol,
        completed_trials: List
    ) -> Dict[str, Any]:
        """Project enrollment timeline based on semantically matched trials."""
        if not completed_trials:
            return {
                "available": False,
                "message": "Not enough historical data for projection"
            }

        # Calculate enrollment rates from completed trials
        rates = []
        for t in completed_trials:
            if t.duration_months and t.duration_months > 0 and t.enrollment and t.num_sites and t.num_sites > 0:
                rate_per_site_month = t.enrollment / t.duration_months / t.num_sites
                rates.append(rate_per_site_month)

        if not rates:
            return {
                "available": False,
                "message": "Could not calculate enrollment rates"
            }

        avg_rate = sum(rates) / len(rates)
        min_rate = min(rates)
        max_rate = max(rates)

        assumed_sites = 50  # Default assumption

        return {
            "available": True,
            "assumed_sites": assumed_sites,
            "rate_per_site_month": avg_rate,
            "optimistic_months": extracted.target_enrollment / (max_rate * assumed_sites) if max_rate > 0 else 0,
            "expected_months": extracted.target_enrollment / (avg_rate * assumed_sites) if avg_rate > 0 else 0,
            "pessimistic_months": extracted.target_enrollment / (min_rate * assumed_sites) if min_rate > 0 else 0,
            "historical_trials_used": len(rates),
        }

    def _extract_termination_lessons_semantic(self, terminated_trials: List) -> List[Dict]:
        """Extract lessons from terminated trials with semantic insights."""
        lessons = []
        for t in terminated_trials[:10]:  # More lessons since no limit
            lesson = {
                "nct_id": t.nct_id,
                "sponsor": t.sponsor,
                "reason": t.why_stopped,
                "enrollment": t.enrollment,
                "similarity": t.overall_similarity,
                "relevance": t.relevance_explanation,
            }
            # Add any lessons identified by Claude during matching
            if t.lessons:
                lesson["insights"] = t.lessons
            lessons.append(lesson)
        return lessons

    def _assess_risks(
        self,
        extracted: ExtractedProtocol,
        metrics: Dict,
        terminated: List
    ) -> Dict[str, Any]:
        """Assess risks based on extracted protocol and historical data."""
        risks = []
        risk_score = 0

        # High termination rate
        if metrics.get("termination_rate", 0) > 25:
            risks.append({
                "category": "Historical Performance",
                "risk": f"High termination rate ({metrics['termination_rate']:.0f}%) in similar trials",
                "severity": "high",
                "recommendation": "Review terminated trial reasons carefully"
            })
            risk_score += 25

        # Enrollment challenges
        if extracted.target_enrollment > metrics.get("avg_enrollment", 0) * 1.5:
            risks.append({
                "category": "Enrollment",
                "risk": f"Target enrollment ({extracted.target_enrollment}) is significantly above average ({metrics.get('avg_enrollment', 0):.0f})",
                "severity": "medium",
                "recommendation": "Consider adding more sites or extending timeline"
            })
            risk_score += 15

        # Complex eligibility
        if len(extracted.key_exclusion) > 5:
            risks.append({
                "category": "Eligibility",
                "risk": "Many exclusion criteria may limit patient pool",
                "severity": "medium",
                "recommendation": "Review if all exclusion criteria are necessary"
            })
            risk_score += 10

        # Age restrictions
        if extracted.min_age and extracted.min_age > 18:
            if extracted.max_age and extracted.max_age < 65:
                risks.append({
                    "category": "Eligibility",
                    "risk": f"Narrow age range ({extracted.min_age}-{extracted.max_age}) may limit recruitment",
                    "severity": "low",
                    "recommendation": "Consider if age range can be expanded"
                })
                risk_score += 5

        # Claude-identified risks
        for risk in extracted.identified_risks[:3]:  # Top 3 AI-identified risks
            risks.append({
                "category": "Protocol Design",
                "risk": risk,
                "severity": "medium",
                "recommendation": "Review protocol design"
            })
            risk_score += 10

        # Complexity score
        if extracted.complexity_score > 70:
            risks.append({
                "category": "Complexity",
                "risk": f"High protocol complexity score ({extracted.complexity_score}/100)",
                "severity": "medium",
                "recommendation": "Consider simplifying study design where possible"
            })
            risk_score += 15

        return {
            "overall_risk_score": min(risk_score, 100),
            "risk_level": "high" if risk_score > 60 else "medium" if risk_score > 30 else "low",
            "risks": risks,
        }

    def _project_enrollment(
        self,
        extracted: ExtractedProtocol,
        completed_trials: List
    ) -> Dict[str, Any]:
        """Project enrollment timeline based on historical data."""
        if not completed_trials:
            return {
                "available": False,
                "message": "Not enough historical data for projection"
            }

        # Calculate enrollment rates from completed trials
        rates = []
        for t in completed_trials:
            if t.duration_months and t.duration_months > 0 and t.enrollment and t.num_sites:
                rate_per_site_month = t.enrollment / t.duration_months / t.num_sites
                rates.append(rate_per_site_month)

        if not rates:
            return {
                "available": False,
                "message": "Could not calculate enrollment rates"
            }

        avg_rate = sum(rates) / len(rates)
        min_rate = min(rates)
        max_rate = max(rates)

        # Assume 50 sites as default if not specified
        assumed_sites = 50

        return {
            "available": True,
            "assumed_sites": assumed_sites,
            "rate_per_site_month": avg_rate,
            "optimistic_months": extracted.target_enrollment / (max_rate * assumed_sites),
            "expected_months": extracted.target_enrollment / (avg_rate * assumed_sites),
            "pessimistic_months": extracted.target_enrollment / (min_rate * assumed_sites),
            "historical_trials_used": len(rates),
        }

    def _get_site_recommendations(
        self,
        extracted: ExtractedProtocol,
        db_manager
    ) -> List[Dict]:
        """Get site recommendations based on extracted protocol using direct SQL."""
        try:
            from sqlalchemy import text

            condition = extracted.condition.lower()

            query = text("""
                SELECT
                    facility_name, city, state, country,
                    total_trials, completed_trials,
                    CASE WHEN total_trials > 0 THEN CAST(completed_trials AS FLOAT) / total_trials ELSE 0 END as completion_rate
                FROM sites
                WHERE total_trials >= 3
                AND (
                    LOWER(therapeutic_areas) LIKE :condition
                    OR LOWER(facility_name) LIKE :condition
                )
                ORDER BY
                    CASE WHEN total_trials > 0 THEN CAST(completed_trials AS FLOAT) / total_trials ELSE 0 END DESC,
                    total_trials DESC
                LIMIT 10
            """)

            results = db_manager.execute_raw(query.text, {"condition": f"%{condition}%"})

            recommendations = []
            for r in results[:5]:
                recommendations.append({
                    "facility_name": r[0],
                    "city": r[1],
                    "country": r[3],
                    "total_trials": r[4],
                    "completion_rate": (r[6] or 0) * 100,
                })

            return recommendations

        except Exception as e:
            logger.warning(f"Could not get site recommendations: {e}")
            return []

    def _extract_termination_lessons(self, terminated_trials: List) -> List[Dict]:
        """Extract lessons from terminated trials."""
        lessons = []
        for t in terminated_trials[:5]:
            if t.why_stopped:
                lessons.append({
                    "nct_id": t.nct_id,
                    "sponsor": t.sponsor,
                    "reason": t.why_stopped,
                    "enrollment": t.enrollment,
                })
        return lessons
