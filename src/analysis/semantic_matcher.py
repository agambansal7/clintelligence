"""
Semantic Trial Matcher using Claude API - IMPROVED VERSION

Better matching through:
1. Multi-field search (conditions, interventions, endpoints, mechanisms)
2. Phrase matching (not just individual words)
3. Smarter pre-scoring with weighted factors
4. More candidates sent to Claude for ranking
5. Better synonym handling
"""

import os
import json
import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class SemanticMatch:
    """A trial matched semantically to the user's protocol."""
    nct_id: str
    title: str
    status: str
    phase: str
    sponsor: str
    enrollment: int
    num_sites: int
    duration_months: Optional[float]
    completion_date: Optional[str]
    why_stopped: Optional[str]

    # Semantic matching scores (0-100)
    overall_similarity: float
    condition_similarity: float
    endpoint_similarity: float
    eligibility_similarity: float
    design_similarity: float
    intervention_similarity: float

    # Claude's explanation
    relevance_explanation: str
    key_similarities: List[str] = field(default_factory=list)
    key_differences: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)


@dataclass
class MatchingContext:
    """Context extracted from user's protocol for matching."""
    condition: str
    condition_synonyms: List[str]
    phase: str
    target_enrollment: int
    primary_endpoint: str
    endpoint_type: str
    key_inclusion: List[str]
    key_exclusion: List[str]
    intervention_type: str
    intervention_name: Optional[str]
    intervention_mechanism: Optional[str]
    comparator: Optional[str]
    study_design: str
    therapeutic_area: str
    # Additional fields for better matching
    biomarkers: List[str] = field(default_factory=list)
    target_population: str = ""


class SemanticTrialMatcher:
    """
    Improved semantic matching with multi-field search and better scoring.
    """

    CONTEXT_EXTRACTION_PROMPT = """You are an expert clinical trial analyst. Extract comprehensive matching criteria from this protocol.

<protocol>
{protocol_text}
</protocol>

Extract the following as JSON. Be THOROUGH - the quality of trial matching depends on this:
{{
    "condition": "Primary condition using standard medical terminology (e.g., 'Type 2 Diabetes Mellitus' not 'T2DM')",
    "condition_synonyms": ["ALL synonyms and alternative names - include abbreviations, ICD terms, lay terms"],
    "phase": "PHASE1, PHASE2, PHASE3, PHASE4, or NA",
    "target_enrollment": <integer>,
    "primary_endpoint": "Exact primary endpoint as stated",
    "endpoint_type": "efficacy, safety, pharmacokinetic, biomarker, or quality_of_life",
    "key_inclusion": ["List ALL key inclusion criteria"],
    "key_exclusion": ["List ALL key exclusion criteria"],
    "intervention_type": "drug, biological, device, behavioral, procedure, dietary, or other",
    "intervention_name": "Specific drug/intervention name if mentioned (e.g., 'semaglutide', 'pembrolizumab')",
    "intervention_mechanism": "Mechanism of action (e.g., 'GLP-1 receptor agonist', 'PD-1 inhibitor', 'SGLT2 inhibitor')",
    "comparator": "placebo, active comparator name, standard of care, or none",
    "study_design": "randomized double-blind, randomized open-label, single-arm, crossover, parallel, etc.",
    "therapeutic_area": "Broad area (oncology, cardiology, endocrinology, neurology, immunology, etc.)",
    "biomarkers": ["Key biomarkers mentioned (HbA1c, LDL-C, tumor markers, etc.)"],
    "target_population": "Description of target population (e.g., 'adults with uncontrolled T2DM on metformin')"
}}

Be specific and comprehensive. Return ONLY valid JSON."""

    BATCH_RANKING_PROMPT = """You are an expert clinical trial analyst. Score how similar each candidate trial is to the user's protocol.

<user_protocol>
Condition: {condition}
Synonyms: {condition_synonyms}
Phase: {phase}
Target Enrollment: {target_enrollment}
Primary Endpoint: {primary_endpoint}
Endpoint Type: {endpoint_type}
Intervention Type: {intervention_type}
Intervention Name: {intervention_name}
Mechanism of Action: {intervention_mechanism}
Comparator: {comparator}
Study Design: {study_design}
Therapeutic Area: {therapeutic_area}
Biomarkers: {biomarkers}
Target Population: {target_population}
Key Inclusion: {key_inclusion}
Key Exclusion: {key_exclusion}
</user_protocol>

<candidate_trials>
{trials_json}
</candidate_trials>

Score EACH trial's similarity to the user's protocol. Consider:

1. **Condition Match** (most important): Same disease? Same subtype? Related condition?
2. **Intervention Match**: Same drug class? Same mechanism? Similar approach?
3. **Endpoint Match**: Same primary endpoint? Similar outcome measures?
4. **Design Match**: Same phase? Similar enrollment? Similar design?
5. **Population Match**: Similar eligibility criteria? Same target population?

Scoring Guidelines:
- 85-100: HIGHLY SIMILAR - Same condition, same/similar intervention mechanism, similar endpoints
- 70-84: VERY SIMILAR - Same condition, related intervention, some endpoint overlap
- 55-69: MODERATELY SIMILAR - Related condition OR same intervention class
- 40-54: SOMEWHAT RELATED - Same therapeutic area, different approach
- 25-39: LOOSELY RELATED - Tangential relationship
- 0-24: NOT RELEVANT - Different disease area entirely

Return JSON array (one object per trial):
[
    {{
        "nct_id": "NCT...",
        "overall_similarity": <0-100>,
        "condition_similarity": <0-100>,
        "endpoint_similarity": <0-100>,
        "eligibility_similarity": <0-100>,
        "design_similarity": <0-100>,
        "intervention_similarity": <0-100>,
        "relevance_explanation": "Why this trial is relevant (2-3 sentences)",
        "key_similarities": ["2-3 specific similarities"],
        "key_differences": ["1-2 key differences"],
        "lessons": ["What can be learned from this trial"]
    }}
]

Return ONLY valid JSON array."""

    def __init__(self, db_manager, api_key: Optional[str] = None):
        """Initialize with database and optional API key."""
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

    def extract_matching_context(self, protocol_text: str) -> MatchingContext:
        """Extract structured matching context from protocol text."""
        prompt = self.CONTEXT_EXTRACTION_PROMPT.format(protocol_text=protocol_text)

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text

        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        data = json.loads(response_text.strip())

        return MatchingContext(
            condition=data.get("condition", ""),
            condition_synonyms=data.get("condition_synonyms", []),
            phase=data.get("phase", "NA"),
            target_enrollment=data.get("target_enrollment", 100),
            primary_endpoint=data.get("primary_endpoint", ""),
            endpoint_type=data.get("endpoint_type", "efficacy"),
            key_inclusion=data.get("key_inclusion", []),
            key_exclusion=data.get("key_exclusion", []),
            intervention_type=data.get("intervention_type", "drug"),
            intervention_name=data.get("intervention_name"),
            intervention_mechanism=data.get("intervention_mechanism"),
            comparator=data.get("comparator"),
            study_design=data.get("study_design", "randomized"),
            therapeutic_area=data.get("therapeutic_area", ""),
            biomarkers=data.get("biomarkers", []),
            target_population=data.get("target_population", ""),
        )

    def _build_search_terms(self, context: MatchingContext) -> Dict[str, List[str]]:
        """Build comprehensive search terms from context."""
        terms = {
            "condition": [],
            "intervention": [],
            "endpoint": [],
            "mechanism": [],
        }

        # Condition terms - include full phrase AND individual significant words
        terms["condition"].append(context.condition.lower())
        terms["condition"].extend([s.lower() for s in context.condition_synonyms])

        # Add significant words from condition (3+ chars, not common words)
        stop_words = {'the', 'and', 'for', 'with', 'type', 'stage', 'grade', 'mild', 'moderate', 'severe'}
        for word in context.condition.lower().split():
            if len(word) >= 4 and word not in stop_words:
                terms["condition"].append(word)

        # Intervention terms
        if context.intervention_name:
            terms["intervention"].append(context.intervention_name.lower())
        if context.intervention_mechanism:
            terms["mechanism"].append(context.intervention_mechanism.lower())
            # Also add key words from mechanism
            for word in context.intervention_mechanism.lower().split():
                if len(word) >= 4:
                    terms["mechanism"].append(word)

        # Endpoint terms
        if context.primary_endpoint:
            terms["endpoint"].append(context.primary_endpoint.lower())
            # Key endpoint words
            for word in context.primary_endpoint.lower().split():
                if len(word) >= 4:
                    terms["endpoint"].append(word)

        # Biomarkers
        terms["endpoint"].extend([b.lower() for b in context.biomarkers])

        # Therapeutic area
        if context.therapeutic_area:
            terms["condition"].append(context.therapeutic_area.lower())

        # Deduplicate
        for key in terms:
            terms[key] = list(set(terms[key]))

        return terms

    def get_candidate_trials(
        self,
        context: MatchingContext,
        max_candidates: int = 500
    ) -> List[Dict[str, Any]]:
        """
        IMPROVED Phase 1: Multi-field candidate retrieval.

        Searches across conditions, interventions, outcomes, and mechanisms.
        """
        from sqlalchemy import text

        search_terms = self._build_search_terms(context)
        candidates = []
        seen_nct_ids = set()

        def add_candidate(row):
            """Add a trial to candidates if not already present."""
            nct_id = row[0]
            if nct_id not in seen_nct_ids:
                seen_nct_ids.add(nct_id)
                candidates.append({
                    "nct_id": nct_id,
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
                })

        with self.db.session() as session:
            # 1. EXACT CONDITION MATCH (highest priority) - full phrase
            for term in search_terms["condition"][:3]:  # Top 3 condition terms
                if len(term) >= 5:  # Only meaningful terms
                    query = text("""
                        SELECT nct_id, title, status, phase, sponsor, enrollment,
                               num_sites, start_date, completion_date, why_stopped,
                               conditions, interventions, eligibility_criteria,
                               primary_outcomes, therapeutic_area
                        FROM trials
                        WHERE LOWER(conditions) LIKE :term
                        AND status IN ('COMPLETED', 'TERMINATED', 'WITHDRAWN', 'RECRUITING', 'ACTIVE_NOT_RECRUITING')
                        ORDER BY
                            CASE WHEN status = 'COMPLETED' THEN 1
                                 WHEN status = 'TERMINATED' THEN 2
                                 ELSE 3 END,
                            enrollment DESC NULLS LAST
                        LIMIT :limit
                    """)
                    result = session.execute(query, {"term": f"%{term}%", "limit": 200})
                    for row in result:
                        add_candidate(row)

            # 2. INTERVENTION/MECHANISM MATCH
            for term in search_terms["mechanism"][:3]:
                if len(term) >= 4:
                    query = text("""
                        SELECT nct_id, title, status, phase, sponsor, enrollment,
                               num_sites, start_date, completion_date, why_stopped,
                               conditions, interventions, eligibility_criteria,
                               primary_outcomes, therapeutic_area
                        FROM trials
                        WHERE LOWER(interventions) LIKE :term
                        AND status IN ('COMPLETED', 'TERMINATED', 'WITHDRAWN', 'RECRUITING')
                        LIMIT :limit
                    """)
                    result = session.execute(query, {"term": f"%{term}%", "limit": 100})
                    for row in result:
                        add_candidate(row)

            # 3. INTERVENTION NAME MATCH (if specific drug mentioned)
            for term in search_terms["intervention"][:2]:
                if len(term) >= 4:
                    query = text("""
                        SELECT nct_id, title, status, phase, sponsor, enrollment,
                               num_sites, start_date, completion_date, why_stopped,
                               conditions, interventions, eligibility_criteria,
                               primary_outcomes, therapeutic_area
                        FROM trials
                        WHERE LOWER(interventions) LIKE :term
                        AND status IN ('COMPLETED', 'TERMINATED', 'RECRUITING')
                        LIMIT :limit
                    """)
                    result = session.execute(query, {"term": f"%{term}%", "limit": 100})
                    for row in result:
                        add_candidate(row)

            # 4. ENDPOINT/BIOMARKER MATCH
            for term in search_terms["endpoint"][:3]:
                if len(term) >= 4:
                    query = text("""
                        SELECT nct_id, title, status, phase, sponsor, enrollment,
                               num_sites, start_date, completion_date, why_stopped,
                               conditions, interventions, eligibility_criteria,
                               primary_outcomes, therapeutic_area
                        FROM trials
                        WHERE LOWER(primary_outcomes) LIKE :term
                        AND status IN ('COMPLETED', 'TERMINATED')
                        LIMIT :limit
                    """)
                    result = session.execute(query, {"term": f"%{term}%", "limit": 100})
                    for row in result:
                        add_candidate(row)

            # 5. PHASE + THERAPEUTIC AREA (if still need more candidates)
            if len(candidates) < 200 and context.phase != "NA":
                query = text("""
                    SELECT nct_id, title, status, phase, sponsor, enrollment,
                           num_sites, start_date, completion_date, why_stopped,
                           conditions, interventions, eligibility_criteria,
                           primary_outcomes, therapeutic_area
                    FROM trials
                    WHERE phase = :phase
                    AND LOWER(therapeutic_area) LIKE :area
                    AND status IN ('COMPLETED', 'TERMINATED')
                    ORDER BY enrollment DESC NULLS LAST
                    LIMIT 150
                """)
                result = session.execute(query, {
                    "phase": context.phase,
                    "area": f"%{context.therapeutic_area.lower()}%"
                })
                for row in result:
                    add_candidate(row)

        logger.info(f"Retrieved {len(candidates)} candidate trials from multi-field search")
        return candidates[:max_candidates]

    def _prescore_candidates(
        self,
        candidates: List[Dict[str, Any]],
        context: MatchingContext
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        IMPROVED pre-scoring with weighted multi-factor matching.
        """
        scored = []

        # Build search patterns
        condition_lower = context.condition.lower()
        condition_words = set(w for w in condition_lower.split() if len(w) >= 4)
        synonym_set = set(s.lower() for s in context.condition_synonyms)

        mechanism_lower = (context.intervention_mechanism or "").lower()
        mechanism_words = set(w for w in mechanism_lower.split() if len(w) >= 4)

        intervention_lower = (context.intervention_name or "").lower()
        endpoint_lower = (context.primary_endpoint or "").lower()
        endpoint_words = set(w for w in endpoint_lower.split() if len(w) >= 4)

        biomarker_set = set(b.lower() for b in context.biomarkers)

        for trial in candidates:
            score = 0.0

            trial_conditions = (trial.get("conditions") or "").lower()
            trial_interventions = (trial.get("interventions") or "").lower()
            trial_outcomes = (trial.get("primary_outcomes") or "").lower()
            trial_title = (trial.get("title") or "").lower()
            trial_eligibility = (trial.get("eligibility_criteria") or "").lower()

            # 1. CONDITION MATCHING (max 40 points)
            # Full condition phrase match
            if condition_lower in trial_conditions:
                score += 40
            else:
                # Synonym match
                for syn in synonym_set:
                    if syn in trial_conditions:
                        score += 35
                        break
                else:
                    # Word-level matching
                    matches = sum(1 for w in condition_words if w in trial_conditions)
                    score += min(25, matches * 8)

            # Title contains condition
            if condition_lower in trial_title:
                score += 10

            # 2. INTERVENTION/MECHANISM MATCHING (max 30 points)
            # Specific intervention name
            if intervention_lower and intervention_lower in trial_interventions:
                score += 30
            # Mechanism match
            elif mechanism_lower:
                if mechanism_lower in trial_interventions:
                    score += 25
                else:
                    matches = sum(1 for w in mechanism_words if w in trial_interventions)
                    score += min(15, matches * 5)

            # 3. ENDPOINT/BIOMARKER MATCHING (max 20 points)
            if endpoint_lower in trial_outcomes:
                score += 20
            else:
                matches = sum(1 for w in endpoint_words if w in trial_outcomes)
                score += min(12, matches * 4)

            # Biomarker match
            for bm in biomarker_set:
                if bm in trial_outcomes or bm in trial_eligibility:
                    score += 5
                    break

            # 4. PHASE MATCHING (max 15 points)
            if trial.get("phase") == context.phase:
                score += 15
            elif trial.get("phase") and context.phase:
                # Adjacent phase gets partial credit
                phase_order = ["PHASE1", "PHASE2", "PHASE3", "PHASE4"]
                try:
                    p1 = phase_order.index(trial.get("phase"))
                    p2 = phase_order.index(context.phase)
                    if abs(p1 - p2) == 1:
                        score += 8
                except ValueError:
                    pass

            # 5. STATUS BONUS (max 10 points)
            if trial.get("status") == "COMPLETED":
                score += 10
            elif trial.get("status") in ["TERMINATED", "WITHDRAWN"]:
                score += 8  # Terminated trials have valuable lessons

            # 6. DATA QUALITY BONUS (max 5 points)
            if trial.get("enrollment") and trial.get("enrollment") > 0:
                score += 2
            if trial.get("primary_outcomes"):
                score += 2
            if trial.get("why_stopped"):
                score += 1  # Terminated with reason is valuable

            scored.append((trial, score))

        return scored

    def _calculate_duration_months(self, start_date: str, completion_date: str) -> Optional[float]:
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
                days = (end - start).days
                return days / 30.44
        except Exception:
            pass
        return None

    def rank_trials_with_claude(
        self,
        context: MatchingContext,
        candidates: List[Dict[str, Any]],
        batch_size: int = 12
    ) -> List[SemanticMatch]:
        """
        Phase 2: Use Claude to semantically rank candidates.
        """
        all_matches = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            # Prepare trials for Claude
            trials_for_claude = []
            for t in batch:
                trials_for_claude.append({
                    "nct_id": t["nct_id"],
                    "title": (t["title"] or "")[:200],
                    "status": t["status"],
                    "phase": t["phase"],
                    "conditions": (t["conditions"] or "")[:400],
                    "interventions": (t["interventions"] or "")[:300],
                    "eligibility": (t["eligibility_criteria"] or "")[:400],
                    "primary_outcomes": (t["primary_outcomes"] or "")[:400],
                    "enrollment": t["enrollment"],
                    "why_stopped": (t["why_stopped"] or "")[:200] if t["why_stopped"] else None,
                })

            prompt = self.BATCH_RANKING_PROMPT.format(
                condition=context.condition,
                condition_synonyms=", ".join(context.condition_synonyms[:5]),
                phase=context.phase,
                target_enrollment=context.target_enrollment,
                primary_endpoint=context.primary_endpoint,
                endpoint_type=context.endpoint_type,
                key_inclusion="; ".join(context.key_inclusion[:4]),
                key_exclusion="; ".join(context.key_exclusion[:3]),
                intervention_type=context.intervention_type,
                intervention_name=context.intervention_name or "Not specified",
                intervention_mechanism=context.intervention_mechanism or "Not specified",
                comparator=context.comparator or "Not specified",
                study_design=context.study_design,
                therapeutic_area=context.therapeutic_area,
                biomarkers=", ".join(context.biomarkers[:5]) if context.biomarkers else "None specified",
                target_population=context.target_population or "Not specified",
                trials_json=json.dumps(trials_for_claude, indent=2)
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

                rankings = json.loads(response_text.strip())

                for ranking in rankings:
                    original = next(
                        (t for t in batch if t["nct_id"] == ranking["nct_id"]),
                        None
                    )
                    if not original:
                        continue

                    duration = self._calculate_duration_months(
                        original.get("start_date"),
                        original.get("completion_date")
                    )

                    match = SemanticMatch(
                        nct_id=ranking["nct_id"],
                        title=original.get("title", ""),
                        status=original.get("status", ""),
                        phase=original.get("phase", ""),
                        sponsor=original.get("sponsor", ""),
                        enrollment=original.get("enrollment") or 0,
                        num_sites=original.get("num_sites") or 0,
                        duration_months=duration,
                        completion_date=original.get("completion_date"),
                        why_stopped=original.get("why_stopped"),
                        overall_similarity=ranking.get("overall_similarity", 0),
                        condition_similarity=ranking.get("condition_similarity", 0),
                        endpoint_similarity=ranking.get("endpoint_similarity", 0),
                        eligibility_similarity=ranking.get("eligibility_similarity", 0),
                        design_similarity=ranking.get("design_similarity", 0),
                        intervention_similarity=ranking.get("intervention_similarity", 0),
                        relevance_explanation=ranking.get("relevance_explanation", ""),
                        key_similarities=ranking.get("key_similarities", []),
                        key_differences=ranking.get("key_differences", []),
                        lessons=ranking.get("lessons", []),
                    )
                    all_matches.append(match)

            except Exception as e:
                logger.error(f"Error ranking batch: {e}")
                # Fallback scoring
                for t in batch:
                    match = SemanticMatch(
                        nct_id=t["nct_id"],
                        title=t.get("title", ""),
                        status=t.get("status", ""),
                        phase=t.get("phase", ""),
                        sponsor=t.get("sponsor", ""),
                        enrollment=t.get("enrollment") or 0,
                        num_sites=t.get("num_sites") or 0,
                        duration_months=self._calculate_duration_months(
                            t.get("start_date"), t.get("completion_date")
                        ),
                        completion_date=t.get("completion_date"),
                        why_stopped=t.get("why_stopped"),
                        overall_similarity=30,
                        condition_similarity=30,
                        endpoint_similarity=30,
                        eligibility_similarity=30,
                        design_similarity=30,
                        intervention_similarity=30,
                        relevance_explanation="Matched by keyword search (Claude ranking failed)",
                        key_similarities=[],
                        key_differences=[],
                        lessons=[],
                    )
                    all_matches.append(match)

            time.sleep(0.1)

        all_matches.sort(key=lambda x: x.overall_similarity, reverse=True)
        return all_matches

    def find_similar_trials(
        self,
        protocol_text: str,
        min_similarity: int = 30,
        max_candidates: int = 500,
        semantic_rank_top_n: int = 100  # INCREASED from 50
    ) -> Tuple[MatchingContext, List[SemanticMatch]]:
        """
        Main entry point: Find semantically similar trials.

        IMPROVED:
        - Better multi-field candidate retrieval
        - Smarter pre-scoring
        - More candidates sent to Claude (100 vs 50)
        """
        logger.info("Starting IMPROVED semantic trial matching...")

        # Phase 1: Extract matching context
        logger.info("Phase 1: Extracting comprehensive matching context...")
        context = self.extract_matching_context(protocol_text)
        logger.info(f"Extracted: {context.condition} | {context.phase} | Mechanism: {context.intervention_mechanism}")

        # Phase 2: Get candidate trials with multi-field search
        logger.info("Phase 2: Multi-field candidate retrieval...")
        candidates = self.get_candidate_trials(context, max_candidates)
        logger.info(f"Retrieved {len(candidates)} candidates")

        if not candidates:
            logger.warning("No candidate trials found")
            return context, []

        # Phase 3: Pre-score with improved algorithm
        logger.info("Phase 3: Smart pre-scoring...")
        scored_candidates = self._prescore_candidates(candidates, context)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Log top pre-scores for debugging
        for trial, score in scored_candidates[:5]:
            logger.info(f"  Pre-score {score:.0f}: {trial['nct_id']} - {trial.get('conditions', '')[:50]}")

        # Take top N for Claude ranking
        top_candidates = [c[0] for c in scored_candidates[:semantic_rank_top_n]]
        logger.info(f"Selected top {len(top_candidates)} for Claude ranking")

        # Phase 4: Semantic ranking with Claude
        logger.info("Phase 4: Claude semantic ranking...")
        matches = self.rank_trials_with_claude(context, top_candidates)
        logger.info(f"Ranked {len(matches)} trials")

        # Filter by minimum similarity
        filtered_matches = [m for m in matches if m.overall_similarity >= min_similarity]
        logger.info(f"Final: {len(filtered_matches)} trials with similarity >= {min_similarity}")

        return context, filtered_matches

    def get_match_summary(self, matches: List[SemanticMatch]) -> Dict[str, Any]:
        """Generate summary statistics for matches."""
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
            "termination_rate": len(terminated) / len(matches) * 100 if matches else 0,
            "avg_enrollment": sum(m.enrollment for m in matches) / len(matches) if matches else 0,
            "avg_duration": sum(m.duration_months or 0 for m in completed if m.duration_months) / len(completed) if completed else 0,
        }
