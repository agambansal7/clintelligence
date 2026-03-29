"""
Improved Trial Matcher

New architecture for protocol-to-trial matching:
1. Enhanced multi-dimensional extraction
2. Parallel specialized searches (condition, intervention, endpoint)
3. Multi-dimensional scoring with hard filters
4. Claude-powered intelligent reranking
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import time

from src.analysis.enhanced_extractor import ExtractedProtocolV2, get_extractor
from src.analysis.trial_scorer import MultiDimensionalScore, get_scorer
from src.analysis.medical_ontology import get_ontology

logger = logging.getLogger(__name__)


@dataclass
class MatchedTrial:
    """A trial matched with full scoring details."""
    nct_id: str
    title: str
    conditions: str
    interventions: str
    phase: str
    status: str
    enrollment: int
    primary_outcomes: str
    eligibility_criteria: str

    # Scores
    overall_score: float
    condition_score: float
    intervention_score: float
    endpoint_score: float
    population_score: float
    design_score: float

    # Conflict info
    has_exclusion_conflict: bool = False
    exclusion_reasons: List[str] = field(default_factory=list)

    # Claude analysis (after reranking)
    relevance_explanation: str = ""
    key_similarities: List[str] = field(default_factory=list)
    key_differences: List[str] = field(default_factory=list)
    strategic_insights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "conditions": self.conditions,
            "interventions": self.interventions,
            "phase": self.phase,
            "status": self.status,
            "enrollment": self.enrollment,
            "primary_outcomes": self.primary_outcomes,
            "overall_similarity": self.overall_score,
            "condition_score": self.condition_score,
            "intervention_score": self.intervention_score,
            "endpoint_score": self.endpoint_score,
            "population_score": self.population_score,
            "design_score": self.design_score,
            "has_exclusion_conflict": self.has_exclusion_conflict,
            "exclusion_reasons": self.exclusion_reasons,
            "relevance_explanation": self.relevance_explanation,
            "key_similarities": self.key_similarities,
            "key_differences": self.key_differences,
            "strategic_insights": self.strategic_insights,
        }


class ImprovedTrialMatcher:
    """
    New improved trial matching system.
    """

    RERANKING_PROMPT = '''You are an expert clinical trial analyst helping identify the most relevant precedent trials.

<user_protocol>
Condition: {condition} ({condition_category})
Phase: {phase}
Target Enrollment: {enrollment}
Intervention: {intervention_type} - {drug_class}
  Similar drugs: {similar_drugs}
  Route: {route}, Frequency: {frequency}
Primary Endpoint: {primary_endpoint}
Population: {population_summary}
Key Exclusions: {exclusions}
</user_protocol>

<candidate_trials>
{trials_json}
</candidate_trials>

For each trial, provide:
1. **Relevance Score** (50-100): How useful is this trial for planning the user's study?
2. **Relevance Explanation**: 1-2 sentences on why this trial is relevant
3. **Key Similarities**: What makes this trial comparable?
4. **Key Differences**: Important differences to note
5. **Strategic Insights**: What can be learned from this trial?

IMPORTANT SCORING GUIDANCE:
- 90-100: Directly comparable (same condition, same drug class, same phase, same endpoints)
- 80-89: Highly relevant (same condition, similar mechanism, compatible population)
- 70-79: Very relevant (same condition, different mechanism OR same mechanism, related condition)
- 60-69: Moderately relevant (related condition and mechanism)
- 50-59: Loosely relevant (same therapeutic area only)
- Below 50: Not relevant (conflicts with protocol requirements)

CRITICAL: If a trial has an exclusion conflict (e.g., trial requires diabetes but protocol excludes diabetes), score it below 50.

Return JSON array:
[
    {{
        "nct_id": "NCT...",
        "relevance_score": <50-100>,
        "relevance_explanation": "...",
        "key_similarities": ["...", "..."],
        "key_differences": ["...", "..."],
        "strategic_insights": ["..."]
    }}
]

Return ONLY valid JSON.'''

    def __init__(self, db_manager, api_key: Optional[str] = None):
        """Initialize the improved matcher."""
        self.db = db_manager
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        self._vector_store = None
        self.extractor = get_extractor()
        self.scorer = get_scorer()
        self.ontology = get_ontology()

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @property
    def vector_store(self):
        if self._vector_store is None:
            from src.analysis.vector_store import get_vector_store
            self._vector_store = get_vector_store()
        return self._vector_store

    def find_similar_trials(
        self,
        protocol_text: str,
        min_score: int = 40,
        max_results: int = 50,
        use_reranking: bool = True,
        weights: Dict[str, float] = None
    ) -> Tuple[ExtractedProtocolV2, List[MatchedTrial]]:
        """
        Find similar trials using improved matching.

        Args:
            protocol_text: Raw protocol text
            min_score: Minimum overall score (0-100)
            max_results: Maximum results to return
            use_reranking: Whether to use Claude for final reranking
            weights: Custom dimension weights

        Returns:
            Tuple of (extracted protocol, list of matched trials)
        """
        logger.info("=" * 60)
        logger.info("Starting Improved Trial Matching")
        logger.info("=" * 60)

        # Step 1: Enhanced extraction
        logger.info("Step 1: Extracting structured protocol information...")
        protocol = self.extractor.extract(protocol_text)
        logger.info(f"  Condition: {protocol.condition} ({protocol.condition_category})")
        logger.info(f"  Intervention: {protocol.intervention.drug_class or protocol.intervention.drug_name}")
        logger.info(f"  Phase: {protocol.design.phase}")
        logger.info(f"  Excluded conditions: {protocol.population.excluded_conditions}")

        # Step 2: Multi-query vector search
        logger.info("Step 2: Running specialized vector searches...")
        vector_candidates = self._multi_vector_search(protocol, n_results=400)
        logger.info(f"  Vector search found {len(vector_candidates)} candidates")

        # Step 3: Keyword/SQL search with ontology expansion
        logger.info("Step 3: Running keyword search with ontology expansion...")
        keyword_candidates = self._keyword_search(protocol, max_results=400)
        logger.info(f"  Keyword search found {len(keyword_candidates)} candidates")

        # Step 4: Merge candidates
        all_candidates = self._merge_candidates(vector_candidates, keyword_candidates)
        logger.info(f"Step 4: Merged to {len(all_candidates)} unique candidates")

        # Step 5: Multi-dimensional scoring
        logger.info("Step 5: Computing multi-dimensional scores...")
        scored_trials = []
        for nct_id, trial_data in all_candidates.items():
            vector_score = trial_data.get("_vector_score", 0.5)
            score = self.scorer.score_trial(protocol, trial_data, vector_score)

            if score.overall_score >= min_score * 0.6:  # Pre-filter
                matched = MatchedTrial(
                    nct_id=nct_id,
                    title=trial_data.get("title", ""),
                    conditions=trial_data.get("conditions", ""),
                    interventions=trial_data.get("interventions", ""),
                    phase=trial_data.get("phase", ""),
                    status=trial_data.get("status", ""),
                    enrollment=trial_data.get("enrollment") or 0,
                    primary_outcomes=trial_data.get("primary_outcomes", ""),
                    eligibility_criteria=trial_data.get("eligibility_criteria", ""),
                    overall_score=score.overall_score,
                    condition_score=score.condition_score.score,
                    intervention_score=score.intervention_score.score,
                    endpoint_score=score.endpoint_score.score,
                    population_score=score.population_score.score,
                    design_score=score.design_score.score,
                    has_exclusion_conflict=score.has_exclusion_conflict,
                    exclusion_reasons=score.exclusion_reasons,
                )
                scored_trials.append(matched)

        # Sort by score
        scored_trials.sort(key=lambda x: x.overall_score, reverse=True)
        logger.info(f"  Scored {len(scored_trials)} trials above pre-filter threshold")

        # Step 6: Claude reranking (top candidates only)
        if use_reranking and scored_trials:
            logger.info("Step 6: Claude reranking top candidates...")
            top_for_reranking = scored_trials[:min(80, max_results * 2)]
            scored_trials = self._rerank_with_claude(top_for_reranking, protocol)

        # Final filter
        final_trials = [t for t in scored_trials if t.overall_score >= min_score]
        final_trials = final_trials[:max_results]

        logger.info(f"Final: {len(final_trials)} trials with score >= {min_score}")
        logger.info("=" * 60)

        return protocol, final_trials

    def _multi_vector_search(
        self,
        protocol: ExtractedProtocolV2,
        n_results: int = 400
    ) -> Dict[str, Dict]:
        """Run multiple specialized vector searches."""
        candidates = {}

        if not self.vector_store.is_initialized():
            logger.warning("Vector store not initialized")
            return candidates

        # Search 1: Condition-focused (50% weight)
        condition_query = self._build_condition_query(protocol)
        try:
            results = self.vector_store.search(condition_query, n_results=n_results)
            for r in results:
                if r.nct_id not in candidates:
                    candidates[r.nct_id] = {"_vector_score": 0}
                candidates[r.nct_id]["_vector_score"] += r.score * 0.50
                candidates[r.nct_id]["_condition_vector"] = r.score
            logger.info(f"    Condition search: {len(results)} results")
        except Exception as e:
            logger.warning(f"Condition vector search failed: {e}")

        # Search 2: Intervention-focused (30% weight)
        intervention_query = self._build_intervention_query(protocol)
        if intervention_query:
            try:
                results = self.vector_store.search(intervention_query, n_results=n_results // 2)
                for r in results:
                    if r.nct_id not in candidates:
                        candidates[r.nct_id] = {"_vector_score": 0}
                    candidates[r.nct_id]["_vector_score"] += r.score * 0.30
                    candidates[r.nct_id]["_intervention_vector"] = r.score
                logger.info(f"    Intervention search: {len(results)} results")
            except Exception as e:
                logger.warning(f"Intervention vector search failed: {e}")

        # Search 3: Endpoint-focused (20% weight)
        endpoint_query = self._build_endpoint_query(protocol)
        if endpoint_query:
            try:
                results = self.vector_store.search(endpoint_query, n_results=n_results // 3)
                for r in results:
                    if r.nct_id not in candidates:
                        candidates[r.nct_id] = {"_vector_score": 0}
                    candidates[r.nct_id]["_vector_score"] += r.score * 0.20
                    candidates[r.nct_id]["_endpoint_vector"] = r.score
                logger.info(f"    Endpoint search: {len(results)} results")
            except Exception as e:
                logger.warning(f"Endpoint vector search failed: {e}")

        return candidates

    def _build_condition_query(self, protocol: ExtractedProtocolV2) -> str:
        """Build condition-focused search query."""
        parts = [protocol.condition]
        parts.extend(protocol.condition_synonyms[:3])
        if protocol.therapeutic_area:
            parts.append(protocol.therapeutic_area)
        parts.append("clinical trial")
        return " ".join(parts)

    def _build_intervention_query(self, protocol: ExtractedProtocolV2) -> str:
        """Build intervention-focused search query."""
        intv = protocol.intervention
        parts = []

        if intv.drug_class:
            parts.append(intv.drug_class)
        if intv.drug_name and intv.drug_name.lower() != "investigational":
            parts.append(intv.drug_name)
        if intv.mechanism_of_action:
            parts.append(intv.mechanism_of_action)

        # Add similar known drugs
        parts.extend(intv.similar_known_drugs[:3])

        # Add search terms
        parts.extend(intv.search_terms[:3])

        if parts:
            parts.append("treatment therapy")
            return " ".join(parts)
        return ""

    def _build_endpoint_query(self, protocol: ExtractedProtocolV2) -> str:
        """Build endpoint-focused search query."""
        endp = protocol.endpoints
        parts = []

        if endp.primary_endpoint:
            # Extract key words from endpoint
            words = [w for w in endp.primary_endpoint.split() if len(w) >= 4]
            parts.extend(words[:5])

        if endp.has_weight_endpoint:
            parts.extend(["weight loss", "body weight", "BMI"])
        if endp.has_glycemic_endpoint:
            parts.extend(["HbA1c", "glycemic control"])
        if endp.has_survival_endpoint:
            parts.extend(["overall survival", "progression-free survival"])
        if endp.has_cv_endpoint:
            parts.extend(["cardiovascular", "MACE"])

        if parts:
            parts.append("primary endpoint outcome")
            return " ".join(parts)
        return ""

    def _keyword_search(
        self,
        protocol: ExtractedProtocolV2,
        max_results: int = 400
    ) -> List[Dict]:
        """SQL keyword search with ontology expansion."""
        from sqlalchemy import text

        candidates = []
        seen_ids = set()

        # Build search terms
        condition_terms = [protocol.condition.lower()]
        condition_terms.extend([s.lower() for s in protocol.condition_synonyms[:8]])

        # Add ontology expansion
        canonical, expanded = self.ontology.normalize_condition(protocol.condition)
        condition_terms.extend([t.lower() for t in expanded[:5]])
        condition_terms = list(set(condition_terms))[:12]

        # Search by condition terms
        for term in condition_terms[:6]:
            if len(term) < 3:
                continue

            sql = text("""
                SELECT
                    nct_id, title, status, phase, sponsor, enrollment,
                    conditions, interventions, eligibility_criteria,
                    primary_outcomes, therapeutic_area, min_age, max_age, sex
                FROM trials
                WHERE (
                    LOWER(conditions) LIKE :term
                    OR LOWER(therapeutic_area) LIKE :term
                    OR LOWER(title) LIKE :term
                )
                AND status IN ('COMPLETED', 'TERMINATED', 'WITHDRAWN', 'RECRUITING', 'ACTIVE_NOT_RECRUITING')
                LIMIT :limit
            """)

            with self.db.engine.connect() as conn:
                results = conn.execute(sql, {
                    "term": f"%{term}%",
                    "limit": max_results // 4
                }).fetchall()

            for row in results:
                if row[0] not in seen_ids:
                    seen_ids.add(row[0])
                    candidates.append(self._row_to_dict(row))

        # Search by intervention terms
        intv = protocol.intervention
        intervention_terms = []
        if intv.drug_class:
            intervention_terms.append(intv.drug_class.lower())
        intervention_terms.extend([d.lower() for d in intv.similar_known_drugs[:5]])
        intervention_terms.extend([t.lower() for t in intv.search_terms[:5]])
        intervention_terms = list(set(intervention_terms))

        for term in intervention_terms[:5]:
            if len(term) < 4:
                continue

            sql = text("""
                SELECT
                    nct_id, title, status, phase, sponsor, enrollment,
                    conditions, interventions, eligibility_criteria,
                    primary_outcomes, therapeutic_area, min_age, max_age, sex
                FROM trials
                WHERE LOWER(interventions) LIKE :term
                AND status IN ('COMPLETED', 'TERMINATED', 'RECRUITING')
                LIMIT :limit
            """)

            with self.db.engine.connect() as conn:
                results = conn.execute(sql, {
                    "term": f"%{term}%",
                    "limit": 100
                }).fetchall()

            for row in results:
                if row[0] not in seen_ids:
                    seen_ids.add(row[0])
                    candidates.append(self._row_to_dict(row))

        return candidates[:max_results]

    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary."""
        return {
            "nct_id": row[0],
            "title": row[1],
            "status": row[2],
            "phase": row[3],
            "sponsor": row[4],
            "enrollment": row[5],
            "conditions": row[6],
            "interventions": row[7],
            "eligibility_criteria": row[8],
            "primary_outcomes": row[9],
            "therapeutic_area": row[10],
            "min_age": row[11],
            "max_age": row[12],
            "sex": row[13],
        }

    def _merge_candidates(
        self,
        vector_candidates: Dict[str, Dict],
        keyword_candidates: List[Dict]
    ) -> Dict[str, Dict]:
        """Merge vector and keyword candidates."""
        merged = {}

        # Add vector candidates
        for nct_id, data in vector_candidates.items():
            merged[nct_id] = data

        # Add/merge keyword candidates
        for trial in keyword_candidates:
            nct_id = trial["nct_id"]
            if nct_id in merged:
                # Merge - keep vector score, add trial data
                merged[nct_id].update(trial)
            else:
                trial["_vector_score"] = 0.3  # Default score for keyword-only
                merged[nct_id] = trial

        # Fetch full data for vector-only candidates
        vector_only = [nct_id for nct_id in merged if "title" not in merged[nct_id]]
        if vector_only:
            self._fetch_trial_data(vector_only, merged)

        return merged

    def _fetch_trial_data(self, nct_ids: List[str], merged: Dict[str, Dict]):
        """Fetch full trial data for given NCT IDs."""
        from sqlalchemy import text

        if not nct_ids:
            return

        # Batch fetch
        placeholders = ",".join([f"'{nct_id}'" for nct_id in nct_ids[:200]])
        sql = text(f"""
            SELECT
                nct_id, title, status, phase, sponsor, enrollment,
                conditions, interventions, eligibility_criteria,
                primary_outcomes, therapeutic_area, min_age, max_age, sex
            FROM trials
            WHERE nct_id IN ({placeholders})
        """)

        with self.db.engine.connect() as conn:
            results = conn.execute(sql).fetchall()
        for row in results:
            nct_id = row[0]
            if nct_id in merged:
                trial_data = self._row_to_dict(row)
                merged[nct_id].update(trial_data)

    def _rerank_with_claude(
        self,
        candidates: List[MatchedTrial],
        protocol: ExtractedProtocolV2
    ) -> List[MatchedTrial]:
        """Rerank candidates using Claude."""
        if not candidates:
            return candidates

        # Prepare trials for Claude (batch processing)
        batch_size = 20
        all_reranked = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            trials_for_claude = []
            for t in batch:
                trials_for_claude.append({
                    "nct_id": t.nct_id,
                    "title": (t.title or "")[:150],
                    "status": t.status,
                    "phase": t.phase,
                    "conditions": (t.conditions or "")[:200],
                    "interventions": (t.interventions or "")[:200],
                    "primary_outcomes": (t.primary_outcomes or "")[:200],
                    "enrollment": t.enrollment,
                    "pre_scores": {
                        "overall": round(t.overall_score, 1),
                        "condition": round(t.condition_score, 1),
                        "intervention": round(t.intervention_score, 1),
                    },
                    "has_exclusion_conflict": t.has_exclusion_conflict,
                    "exclusion_reasons": t.exclusion_reasons[:2] if t.exclusion_reasons else [],
                })

            prompt = self.RERANKING_PROMPT.format(
                condition=protocol.condition,
                condition_category=protocol.condition_category,
                phase=protocol.design.phase,
                enrollment=protocol.design.target_enrollment,
                intervention_type=protocol.intervention.intervention_type,
                drug_class=protocol.intervention.drug_class or "Unknown",
                similar_drugs=", ".join(protocol.intervention.similar_known_drugs[:3]) or "N/A",
                route=protocol.intervention.route or "N/A",
                frequency=protocol.intervention.frequency or "N/A",
                primary_endpoint=protocol.endpoints.primary_endpoint,
                population_summary=f"Ages {protocol.population.min_age}-{protocol.population.max_age}",
                exclusions=", ".join(protocol.population.excluded_conditions[:3]) or "None specified",
                trials_json=json.dumps(trials_for_claude, indent=2),
            )

            try:
                message = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = message.content[0].text
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                # Try to parse JSON, with fallback for truncated responses
                response_text = response_text.strip()
                try:
                    rankings = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to repair truncated JSON by finding last complete object
                    import re
                    # Find array of objects pattern
                    if response_text.startswith('['):
                        # Find last complete object (ends with })
                        last_complete = response_text.rfind('}')
                        if last_complete > 0:
                            repaired = response_text[:last_complete+1] + ']'
                            try:
                                rankings = json.loads(repaired)
                            except:
                                # Fall back to empty rankings
                                logger.warning("Could not repair truncated JSON, using pre-computed scores")
                                rankings = []
                    else:
                        rankings = []

                # Update candidates with Claude's analysis
                for ranking in rankings:
                    match = next((t for t in batch if t.nct_id == ranking["nct_id"]), None)
                    if match:
                        # Blend scores: 40% pre-computed, 60% Claude
                        pre_score = match.overall_score
                        claude_score = ranking.get("relevance_score", pre_score)
                        match.overall_score = pre_score * 0.4 + claude_score * 0.6

                        match.relevance_explanation = ranking.get("relevance_explanation", "")
                        match.key_similarities = ranking.get("key_similarities", [])
                        match.key_differences = ranking.get("key_differences", [])
                        match.strategic_insights = ranking.get("strategic_insights", [])

                all_reranked.extend(batch)

            except Exception as e:
                logger.error(f"Claude reranking failed for batch: {e}")
                all_reranked.extend(batch)

            time.sleep(0.1)

        # Sort by final score
        all_reranked.sort(key=lambda x: x.overall_score, reverse=True)
        return all_reranked

    def get_summary(self, matches: List[MatchedTrial]) -> Dict:
        """Generate summary statistics."""
        if not matches:
            return {}

        completed = [m for m in matches if m.status == "COMPLETED"]
        with_conflicts = [m for m in matches if m.has_exclusion_conflict]

        avg_scores = {
            "overall": sum(m.overall_score for m in matches) / len(matches),
            "condition": sum(m.condition_score for m in matches) / len(matches),
            "intervention": sum(m.intervention_score for m in matches) / len(matches),
            "endpoint": sum(m.endpoint_score for m in matches) / len(matches),
        }

        return {
            "total_matches": len(matches),
            "completed_trials": len(completed),
            "trials_with_conflicts": len(with_conflicts),
            "average_scores": avg_scores,
            "top_trial": matches[0].nct_id if matches else None,
        }
