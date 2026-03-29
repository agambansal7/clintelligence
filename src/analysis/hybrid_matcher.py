"""
Hybrid Trial Matcher

Combines multiple matching strategies for best results:
1. Vector semantic search (find conceptually similar trials)
2. Keyword/SQL search (find exact matches)
3. Medical ontology expansion (find synonym matches)
4. Structured eligibility comparison (numerical comparison)
5. Claude AI re-ranking (intelligent final ranking)
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class HybridMatch:
    """A trial matched using hybrid approach."""
    nct_id: str
    title: str
    status: str
    phase: str
    sponsor: str
    conditions: str
    interventions: str
    enrollment: int
    num_sites: int
    primary_outcomes: str
    eligibility_criteria: str
    duration_months: Optional[float]
    completion_date: Optional[str]
    why_stopped: Optional[str]

    # Matching scores
    overall_similarity: float  # Final combined score (0-100)
    vector_score: float  # Semantic similarity (0-1)
    keyword_score: float  # Keyword match score (0-100)
    eligibility_score: float  # Eligibility overlap (0-100)
    ontology_boost: float  # Boost from medical ontology match

    # Claude analysis (filled after re-ranking)
    relevance_explanation: str = ""
    key_similarities: List[str] = field(default_factory=list)
    key_differences: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)


@dataclass
class MatchingQuery:
    """Structured query for hybrid matching."""
    condition: str
    condition_synonyms: List[str]
    phase: str
    target_enrollment: int
    primary_endpoint: str
    intervention_type: str
    intervention_name: Optional[str]
    intervention_mechanism: Optional[str]
    eligibility_criteria: str
    therapeutic_area: str

    # Parsed eligibility for comparison
    parsed_eligibility: Any = None  # ParsedEligibility object


class HybridTrialMatcher:
    """
    Advanced trial matching using multiple strategies.
    """

    RERANKING_PROMPT = """You are an expert clinical trial analyst. Re-rank these candidate trials based on their relevance to the user's protocol.

<user_protocol>
Condition: {condition}
Phase: {phase}
Target Enrollment: {target_enrollment}
Primary Endpoint: {primary_endpoint}
Intervention: {intervention_type} - {intervention_name} ({intervention_mechanism})
Therapeutic Area: {therapeutic_area}
Key Eligibility: {eligibility_summary}
</user_protocol>

<candidate_trials>
{trials_json}
</candidate_trials>

For each trial, analyze:
1. **Condition relevance** - Is it the same disease/indication?
2. **Intervention similarity** - Same drug class, mechanism, or approach?
3. **Endpoint alignment** - Similar outcome measures?
4. **Design compatibility** - Same phase, similar enrollment?
5. **Learnings potential** - What can be learned from this trial?

Return a JSON array with your analysis for each trial:
[
    {{
        "nct_id": "NCT...",
        "final_score": <50-100, your assessment of relevance>,
        "relevance_explanation": "2-3 sentences explaining relevance",
        "key_similarities": ["similarity 1", "similarity 2"],
        "key_differences": ["difference 1"],
        "lessons": ["What can be learned from this trial"]
    }}
]

Scoring:
- 85-100: Highly relevant - same condition, similar mechanism/endpoint
- 70-84: Very relevant - same condition, related approach
- 55-69: Moderately relevant - related condition or same drug class
- 50-54: Loosely relevant - same therapeutic area only

Return ONLY valid JSON array."""

    def __init__(self, db_manager, api_key: Optional[str] = None):
        """Initialize hybrid matcher."""
        self.db = db_manager
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        self._vector_store = None
        self._ontology = None
        self._eligibility_parser = None

    @property
    def client(self):
        """Lazy-load Anthropic client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @property
    def vector_store(self):
        """Lazy-load vector store."""
        if self._vector_store is None:
            from src.analysis.vector_store import get_vector_store
            self._vector_store = get_vector_store()
        return self._vector_store

    @property
    def ontology(self):
        """Lazy-load medical ontology."""
        if self._ontology is None:
            from src.analysis.medical_ontology import get_ontology
            self._ontology = get_ontology()
        return self._ontology

    @property
    def eligibility_parser(self):
        """Lazy-load eligibility parser."""
        if self._eligibility_parser is None:
            from src.analysis.eligibility_parser import get_parser
            self._eligibility_parser = get_parser()
        return self._eligibility_parser

    def _build_query(self, protocol_text: str) -> MatchingQuery:
        """Extract structured query from protocol using Claude."""
        prompt = """Extract matching criteria from this clinical trial protocol. Return JSON:

<protocol>
{protocol_text}
</protocol>

{{
    "condition": "Primary condition (standardized medical term)",
    "condition_synonyms": ["synonyms and alternative names"],
    "phase": "PHASE1, PHASE2, PHASE3, PHASE4, or NA",
    "target_enrollment": <integer>,
    "primary_endpoint": "Primary endpoint",
    "intervention_type": "drug, biological, device, behavioral, or other",
    "intervention_name": "Specific drug/device name or null",
    "intervention_mechanism": "Mechanism of action or null",
    "eligibility_summary": "Key eligibility criteria summary",
    "therapeutic_area": "Broad therapeutic area"
}}

Return ONLY JSON.""".format(protocol_text=protocol_text)

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        data = json.loads(response_text.strip())

        # Expand synonyms using ontology
        _, expanded_terms = self.ontology.normalize_condition(data.get("condition", ""))
        all_synonyms = list(set(data.get("condition_synonyms", []) + expanded_terms))

        # Parse eligibility
        parsed_elig = self.eligibility_parser.parse(data.get("eligibility_summary", ""))

        return MatchingQuery(
            condition=data.get("condition", ""),
            condition_synonyms=all_synonyms,
            phase=data.get("phase", "NA"),
            target_enrollment=data.get("target_enrollment", 100),
            primary_endpoint=data.get("primary_endpoint", ""),
            intervention_type=data.get("intervention_type", "drug"),
            intervention_name=data.get("intervention_name"),
            intervention_mechanism=data.get("intervention_mechanism"),
            eligibility_criteria=data.get("eligibility_summary", ""),
            therapeutic_area=data.get("therapeutic_area", ""),
            parsed_eligibility=parsed_elig
        )

    def _vector_search(self, query: MatchingQuery, n_results: int = 300) -> Dict[str, float]:
        """Phase 1: Multi-query vector semantic search for better precision."""
        if not self.vector_store.is_initialized():
            logger.warning("Vector store not initialized, skipping vector search")
            return {}

        all_scores = {}

        # Search 1: Condition-focused (highest weight - 50%)
        condition_query = f"{query.condition} {' '.join(query.condition_synonyms[:3])} {query.therapeutic_area} clinical trial"
        try:
            results = self.vector_store.search(condition_query, n_results=n_results)
            for r in results:
                all_scores[r.nct_id] = all_scores.get(r.nct_id, 0) + r.score * 0.50
            logger.info(f"Condition search found {len(results)} results")
        except Exception as e:
            logger.warning(f"Condition vector search failed: {e}")

        # Search 2: Intervention-focused (30% weight)
        if query.intervention_mechanism or query.intervention_name or query.intervention_type:
            intervention_query = f"{query.intervention_type} {query.intervention_name or ''} {query.intervention_mechanism or ''} treatment therapy drug"
            try:
                results = self.vector_store.search(intervention_query, n_results=n_results // 2)
                for r in results:
                    all_scores[r.nct_id] = all_scores.get(r.nct_id, 0) + r.score * 0.30
                logger.info(f"Intervention search found {len(results)} results")
            except Exception as e:
                logger.warning(f"Intervention vector search failed: {e}")

        # Search 3: Endpoint-focused (20% weight)
        if query.primary_endpoint:
            endpoint_query = f"{query.primary_endpoint} efficacy outcome measure endpoint"
            try:
                results = self.vector_store.search(endpoint_query, n_results=n_results // 3)
                for r in results:
                    all_scores[r.nct_id] = all_scores.get(r.nct_id, 0) + r.score * 0.20
                logger.info(f"Endpoint search found {len(results)} results")
            except Exception as e:
                logger.warning(f"Endpoint vector search failed: {e}")

        logger.info(f"Multi-query vector search found {len(all_scores)} unique candidates")
        return all_scores

    def _keyword_search(self, query: MatchingQuery, max_results: int = 500) -> List[Dict[str, Any]]:
        """Phase 2: SQL keyword search with ontology expansion."""
        from sqlalchemy import text

        candidates = []
        seen_ids = set()

        # Build search terms
        condition_terms = [query.condition.lower()] + [s.lower() for s in query.condition_synonyms[:10]]

        # Add ontology-expanded terms
        expanded = self.ontology.expand_search_terms(query.condition)
        condition_terms.extend(expanded)
        condition_terms = list(set(condition_terms))[:15]  # Limit to avoid query explosion

        # Search by condition terms
        for term in condition_terms[:5]:
            if len(term) < 3:
                continue

            sql = text("""
                SELECT
                    nct_id, title, status, phase, sponsor, enrollment,
                    num_sites, start_date, completion_date, why_stopped,
                    conditions, interventions, eligibility_criteria,
                    primary_outcomes, therapeutic_area
                FROM trials
                WHERE (
                    LOWER(conditions) LIKE :term
                    OR LOWER(therapeutic_area) LIKE :term
                    OR LOWER(title) LIKE :term
                )
                AND status IN ('COMPLETED', 'TERMINATED', 'WITHDRAWN', 'RECRUITING', 'ACTIVE_NOT_RECRUITING')
                LIMIT :limit
            """)

            results = self.db.execute_raw(sql.text, {
                "term": f"%{term}%",
                "limit": max_results // 5
            })

            for row in results:
                if row[0] not in seen_ids:
                    seen_ids.add(row[0])
                    candidates.append(self._row_to_dict(row))

        # Search by intervention mechanism if specified
        if query.intervention_mechanism:
            mechanism_terms = self.ontology.expand_search_terms(query.intervention_mechanism)
            for term in mechanism_terms[:3]:
                if len(term) < 4:
                    continue

                sql = text("""
                    SELECT
                        nct_id, title, status, phase, sponsor, enrollment,
                        num_sites, start_date, completion_date, why_stopped,
                        conditions, interventions, eligibility_criteria,
                        primary_outcomes, therapeutic_area
                    FROM trials
                    WHERE LOWER(interventions) LIKE :term
                    AND status IN ('COMPLETED', 'TERMINATED', 'RECRUITING')
                    LIMIT :limit
                """)

                results = self.db.execute_raw(sql.text, {
                    "term": f"%{term}%",
                    "limit": 100
                })

                for row in results:
                    if row[0] not in seen_ids:
                        seen_ids.add(row[0])
                        candidates.append(self._row_to_dict(row))

        return candidates[:max_results]

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        return {
            "nct_id": row[0],
            "title": row[1],
            "status": row[2],
            "phase": row[3],
            "sponsor": row[4],
            "enrollment": row[5],
            "num_sites": row[6],
            "start_date": row[7],
            "completion_date": row[8],
            "why_stopped": row[9],
            "conditions": row[10],
            "interventions": row[11],
            "eligibility_criteria": row[12],
            "primary_outcomes": row[13],
            "therapeutic_area": row[14],
        }

    def _calculate_keyword_score(self, trial: Dict[str, Any], query: MatchingQuery) -> float:
        """Calculate keyword-based match score."""
        score = 0.0
        max_score = 100.0

        trial_text = f"{trial.get('conditions', '')} {trial.get('interventions', '')} {trial.get('primary_outcomes', '')}".lower()
        trial_title = (trial.get('title') or '').lower()

        # Condition match (40 points max)
        condition_lower = query.condition.lower()
        if condition_lower in trial_text:
            score += 40
        else:
            for syn in query.condition_synonyms[:10]:
                if syn.lower() in trial_text:
                    score += 30
                    break
            else:
                # Partial word match
                words = [w for w in condition_lower.split() if len(w) >= 4]
                word_matches = sum(1 for w in words if w in trial_text)
                score += min(20, word_matches * 8)

        # Phase match (15 points)
        if trial.get('phase') == query.phase:
            score += 15
        elif trial.get('phase') and query.phase:
            # Adjacent phase
            phases = ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]
            try:
                diff = abs(phases.index(trial['phase']) - phases.index(query.phase))
                if diff == 1:
                    score += 8
            except ValueError:
                pass

        # Intervention match (25 points max)
        if query.intervention_mechanism:
            mech_lower = query.intervention_mechanism.lower()
            if mech_lower in trial_text:
                score += 25
            else:
                mech_words = [w for w in mech_lower.split() if len(w) >= 4]
                matches = sum(1 for w in mech_words if w in trial_text)
                score += min(15, matches * 5)

        if query.intervention_name:
            if query.intervention_name.lower() in trial_text:
                score += 10

        # Endpoint match (10 points max)
        if query.primary_endpoint:
            endpoint_words = [w for w in query.primary_endpoint.lower().split() if len(w) >= 4]
            matches = sum(1 for w in endpoint_words if w in trial_text)
            score += min(10, matches * 3)

        # Status bonus (10 points)
        if trial.get('status') == 'COMPLETED':
            score += 10
        elif trial.get('status') in ['TERMINATED', 'WITHDRAWN']:
            score += 7

        return min(score, max_score)

    def _calculate_eligibility_score(self, trial: Dict[str, Any], query: MatchingQuery) -> float:
        """Calculate eligibility overlap score."""
        if not query.parsed_eligibility or not trial.get('eligibility_criteria'):
            return 50.0  # Unknown, neutral score

        trial_eligibility = self.eligibility_parser.parse(trial['eligibility_criteria'])
        return query.parsed_eligibility.calculate_similarity(trial_eligibility)

    def _calculate_duration(self, start_date: str, completion_date: str) -> Optional[float]:
        """Calculate trial duration in months."""
        if not start_date or not completion_date:
            return None
        try:
            from datetime import datetime

            def parse_date(d):
                for fmt in ["%Y-%m-%d", "%Y-%m", "%B %Y", "%Y"]:
                    try:
                        return datetime.strptime(d, fmt)
                    except ValueError:
                        continue
                return None

            start = parse_date(start_date)
            end = parse_date(completion_date)

            if start and end:
                return (end - start).days / 30.44
        except Exception:
            pass
        return None

    def _combine_scores(
        self,
        trial: Dict[str, Any],
        vector_scores: Dict[str, float],
        keyword_score: float,
        eligibility_score: float,
        query: MatchingQuery
    ) -> float:
        """Combine all scores into final similarity score."""
        # Weights
        VECTOR_WEIGHT = 0.35
        KEYWORD_WEIGHT = 0.35
        ELIGIBILITY_WEIGHT = 0.20
        ONTOLOGY_WEIGHT = 0.10

        nct_id = trial['nct_id']

        # Get vector score (convert from 0-1 to 0-100)
        vector_score = vector_scores.get(nct_id, 0.5) * 100

        # Calculate ontology boost
        ontology_boost = 0
        trial_conditions = (trial.get('conditions') or '').lower()
        trial_interventions = (trial.get('interventions') or '').lower()

        # Check if trial matches expanded ontology terms
        canonical, terms = self.ontology.normalize_condition(query.condition)
        for term in terms:
            if term in trial_conditions:
                ontology_boost = 20
                break

        if query.intervention_mechanism:
            canonical, terms = self.ontology.normalize_intervention(query.intervention_mechanism)
            for term in terms:
                if term in trial_interventions:
                    ontology_boost += 10
                    break

        # Combine scores
        combined = (
            vector_score * VECTOR_WEIGHT +
            keyword_score * KEYWORD_WEIGHT +
            eligibility_score * ELIGIBILITY_WEIGHT +
            ontology_boost * ONTOLOGY_WEIGHT
        )

        return min(combined, 100)

    def _rerank_with_claude(
        self,
        candidates: List[HybridMatch],
        query: MatchingQuery,
        batch_size: int = 15
    ) -> List[HybridMatch]:
        """Final phase: Re-rank top candidates using Claude."""
        if not candidates:
            return candidates

        reranked = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            # Prepare trials for Claude
            trials_for_claude = []
            for t in batch:
                trials_for_claude.append({
                    "nct_id": t.nct_id,
                    "title": (t.title or "")[:200],
                    "status": t.status,
                    "phase": t.phase,
                    "conditions": (t.conditions or "")[:300],
                    "interventions": (t.interventions or "")[:250],
                    "primary_outcomes": (t.primary_outcomes or "")[:300],
                    "enrollment": t.enrollment,
                    "pre_score": round(t.overall_similarity, 1),
                })

            prompt = self.RERANKING_PROMPT.format(
                condition=query.condition,
                phase=query.phase,
                target_enrollment=query.target_enrollment,
                primary_endpoint=query.primary_endpoint,
                intervention_type=query.intervention_type,
                intervention_name=query.intervention_name or "Not specified",
                intervention_mechanism=query.intervention_mechanism or "Not specified",
                therapeutic_area=query.therapeutic_area,
                eligibility_summary=query.eligibility_criteria[:300],
                trials_json=json.dumps(trials_for_claude, indent=2)
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

                rankings = json.loads(response_text.strip())

                # Update candidates with Claude's analysis
                for ranking in rankings:
                    match = next((t for t in batch if t.nct_id == ranking["nct_id"]), None)
                    if match:
                        # Blend pre-computed score with Claude's score
                        pre_score = match.overall_similarity
                        claude_score = ranking.get("final_score", pre_score)
                        match.overall_similarity = (pre_score * 0.4 + claude_score * 0.6)

                        match.relevance_explanation = ranking.get("relevance_explanation", "")
                        match.key_similarities = ranking.get("key_similarities", [])
                        match.key_differences = ranking.get("key_differences", [])
                        match.lessons = ranking.get("lessons", [])
                        reranked.append(match)

            except Exception as e:
                logger.error(f"Claude re-ranking failed: {e}")
                # Keep original scores
                reranked.extend(batch)

            time.sleep(0.1)

        # Sort by final score
        reranked.sort(key=lambda x: x.overall_similarity, reverse=True)
        return reranked

    def find_similar_trials(
        self,
        protocol_text: str,
        min_similarity: int = 40,
        max_results: int = 100,
        use_vector_search: bool = True,
        use_claude_reranking: bool = True
    ) -> Tuple[MatchingQuery, List[HybridMatch]]:
        """
        Main entry point: Find similar trials using hybrid approach.

        Args:
            protocol_text: User's protocol text
            min_similarity: Minimum similarity score (0-100)
            max_results: Maximum results to return
            use_vector_search: Whether to use vector semantic search
            use_claude_reranking: Whether to use Claude for final re-ranking

        Returns:
            Tuple of (MatchingQuery, List[HybridMatch])
        """
        logger.info("Starting hybrid trial matching...")

        # Step 1: Build structured query
        logger.info("Step 1: Extracting query from protocol...")
        query = self._build_query(protocol_text)
        logger.info(f"Query: {query.condition} | {query.phase} | {query.intervention_mechanism}")

        # Step 2: Vector semantic search
        vector_scores = {}
        if use_vector_search and self.vector_store.is_initialized():
            logger.info("Step 2: Vector semantic search...")
            try:
                vector_scores = self._vector_search(query, n_results=500)
                logger.info(f"Vector search found {len(vector_scores)} candidates")
            except Exception as e:
                logger.warning(f"Vector search failed (may be building), falling back to keyword: {e}")
                vector_scores = {}
        else:
            logger.info("Step 2: Skipping vector search (not initialized)")

        # Step 3: Keyword/SQL search with ontology expansion
        logger.info("Step 3: Keyword search with ontology expansion...")
        keyword_candidates = self._keyword_search(query, max_results=500)
        logger.info(f"Keyword search found {len(keyword_candidates)} candidates")

        # Combine candidates
        all_candidates = {}
        for trial in keyword_candidates:
            all_candidates[trial['nct_id']] = trial

        # Add any vector-only candidates (if we have vector scores but not in keyword results)
        # This ensures we don't miss semantically similar trials that don't match keywords

        # Step 4: Score all candidates
        logger.info("Step 4: Scoring candidates...")
        scored_matches = []

        for nct_id, trial in all_candidates.items():
            keyword_score = self._calculate_keyword_score(trial, query)
            eligibility_score = self._calculate_eligibility_score(trial, query)

            combined_score = self._combine_scores(
                trial, vector_scores, keyword_score, eligibility_score, query
            )

            if combined_score >= min_similarity * 0.7:  # Pre-filter before Claude
                match = HybridMatch(
                    nct_id=nct_id,
                    title=trial.get('title', ''),
                    status=trial.get('status', ''),
                    phase=trial.get('phase', ''),
                    sponsor=trial.get('sponsor', ''),
                    conditions=trial.get('conditions', ''),
                    interventions=trial.get('interventions', ''),
                    enrollment=trial.get('enrollment') or 0,
                    num_sites=trial.get('num_sites') or 0,
                    primary_outcomes=trial.get('primary_outcomes', ''),
                    eligibility_criteria=trial.get('eligibility_criteria', ''),
                    duration_months=self._calculate_duration(
                        trial.get('start_date'),
                        trial.get('completion_date')
                    ),
                    completion_date=trial.get('completion_date'),
                    why_stopped=trial.get('why_stopped'),
                    overall_similarity=combined_score,
                    vector_score=vector_scores.get(nct_id, 0.5),
                    keyword_score=keyword_score,
                    eligibility_score=eligibility_score,
                    ontology_boost=0,  # Already factored into combined score
                )
                scored_matches.append(match)

        # Sort by combined score
        scored_matches.sort(key=lambda x: x.overall_similarity, reverse=True)
        logger.info(f"Scored {len(scored_matches)} candidates above threshold")

        # Step 5: Claude re-ranking (only top candidates)
        if use_claude_reranking and scored_matches:
            logger.info("Step 5: Claude re-ranking top candidates...")
            top_for_reranking = scored_matches[:min(150, max_results * 2)]
            reranked = self._rerank_with_claude(top_for_reranking, query)
            scored_matches = reranked

        # Final filter and limit
        final_matches = [m for m in scored_matches if m.overall_similarity >= min_similarity]
        final_matches = final_matches[:max_results]

        logger.info(f"Final: {len(final_matches)} matches with similarity >= {min_similarity}")
        return query, final_matches

    def get_match_summary(self, matches: List[HybridMatch]) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not matches:
            return {}

        completed = [m for m in matches if m.status == "COMPLETED"]
        terminated = [m for m in matches if m.status in ["TERMINATED", "WITHDRAWN"]]

        return {
            "total_matches": len(matches),
            "completed_count": len(completed),
            "terminated_count": len(terminated),
            "avg_similarity": sum(m.overall_similarity for m in matches) / len(matches),
            "high_similarity_count": len([m for m in matches if m.overall_similarity >= 70]),
            "completion_rate": len(completed) / len(matches) * 100 if matches else 0,
            "avg_enrollment": sum(m.enrollment for m in matches) / len(matches) if matches else 0,
        }
