"""
Vector Store Manager for Semantic Trial Search

Uses ChromaDB for vector storage and OpenAI embeddings.
Enables semantic search across 566K+ clinical trials.
"""

import os
import json
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

# Constants - Using OpenAI embeddings for better medical domain understanding
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model, 1536 dimensions
EMBEDDING_DIMENSIONS = 1536
COLLECTION_NAME = "clinical_trials_openai"  # New collection for OpenAI embeddings
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma_db")


@dataclass
class VectorSearchResult:
    """A single vector search result."""
    nct_id: str
    score: float  # Similarity score (0-1)
    title: str
    conditions: str
    interventions: str
    phase: str
    status: str
    enrollment: Optional[int]


class VectorStoreManager:
    """
    Manages vector embeddings and semantic search for clinical trials.
    """

    def __init__(self, db_path: str = CHROMA_PATH):
        """Initialize vector store."""
        self.db_path = db_path
        self._client = None
        self._collection = None
        self._embedding_model = None

    @property
    def openai_client(self):
        """Lazy-load OpenAI client."""
        if self._embedding_model is None:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self._embedding_model = OpenAI(api_key=api_key)
                logger.info(f"OpenAI client initialized for embeddings: {EMBEDDING_MODEL}")
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        return self._embedding_model

    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                os.makedirs(self.db_path, exist_ok=True)

                self._client = chromadb.PersistentClient(
                    path=self.db_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info(f"ChromaDB initialized at {self.db_path}")
            except ImportError:
                raise ImportError("Please install chromadb: pip install chromadb")
        return self._client

    @property
    def collection(self):
        """Get or create the trials collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        return self._collection

    def _create_trial_text(self, trial: Dict[str, Any]) -> str:
        """Create searchable text representation of a trial."""
        parts = []

        if trial.get("title"):
            parts.append(f"Title: {trial['title']}")

        if trial.get("conditions"):
            parts.append(f"Conditions: {trial['conditions']}")

        if trial.get("interventions"):
            parts.append(f"Interventions: {trial['interventions']}")

        if trial.get("primary_outcomes"):
            parts.append(f"Primary Outcomes: {trial['primary_outcomes'][:500]}")

        if trial.get("eligibility_criteria"):
            # Truncate eligibility to avoid too long text
            parts.append(f"Eligibility: {trial['eligibility_criteria'][:500]}")

        if trial.get("phase"):
            parts.append(f"Phase: {trial['phase']}")

        return " | ".join(parts)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        # Truncate to avoid token limits (8191 tokens max for text-embedding-3-small)
        text = text[:30000]  # Rough character limit

        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI with batching."""
        all_embeddings = []

        # OpenAI recommends batches of up to 2048 inputs
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Truncate each text
            batch = [t[:30000] for t in batch]

            try:
                response = self.openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # Rate limiting - be gentle with API
                if i + batch_size < len(texts):
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"OpenAI embedding batch failed: {e}")
                # Retry with smaller batches
                for text in batch:
                    try:
                        emb = self.embed_text(text)
                        all_embeddings.append(emb)
                        time.sleep(0.05)
                    except Exception as e2:
                        logger.error(f"Single embedding failed: {e2}")
                        # Use zero vector as fallback
                        all_embeddings.append([0.0] * EMBEDDING_DIMENSIONS)

        return all_embeddings

    def add_trials(self, trials: List[Dict[str, Any]], batch_size: int = 500):
        """
        Add trials to the vector store.

        Args:
            trials: List of trial dictionaries
            batch_size: Number of trials to process at once
        """
        total = len(trials)
        logger.info(f"Adding {total} trials to vector store...")

        for i in range(0, total, batch_size):
            batch = trials[i:i + batch_size]

            ids = []
            documents = []
            metadatas = []

            for trial in batch:
                nct_id = trial.get("nct_id")
                if not nct_id:
                    continue

                ids.append(nct_id)
                documents.append(self._create_trial_text(trial))
                metadatas.append({
                    "nct_id": nct_id,
                    "title": (trial.get("title") or "")[:500],
                    "conditions": (trial.get("conditions") or "")[:500],
                    "interventions": (trial.get("interventions") or "")[:500],
                    "phase": trial.get("phase") or "",
                    "status": trial.get("status") or "",
                    "enrollment": trial.get("enrollment") or 0,
                    "sponsor": (trial.get("sponsor") or "")[:200],
                })

            if ids:
                # Generate embeddings
                embeddings = self.embed_texts(documents)

                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )

            logger.info(f"Processed {min(i + batch_size, total)}/{total} trials")

        logger.info(f"Vector store now contains {self.collection.count()} trials")

    def search(
        self,
        query: str,
        n_results: int = 100,
        filter_dict: Optional[Dict] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar trials using vector similarity.

        Args:
            query: Search query (protocol description, condition, etc.)
            n_results: Maximum number of results
            filter_dict: Optional metadata filters

        Returns:
            List of VectorSearchResult objects
        """
        # Generate query embedding
        query_embedding = self.embed_text(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )

        # Convert to VectorSearchResult objects
        search_results = []

        if results and results['ids'] and results['ids'][0]:
            for i, nct_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 1.0

                # Convert distance to similarity (cosine distance to similarity)
                similarity = 1 - distance

                search_results.append(VectorSearchResult(
                    nct_id=nct_id,
                    score=similarity,
                    title=metadata.get("title", ""),
                    conditions=metadata.get("conditions", ""),
                    interventions=metadata.get("interventions", ""),
                    phase=metadata.get("phase", ""),
                    status=metadata.get("status", ""),
                    enrollment=metadata.get("enrollment"),
                ))

        return search_results

    def search_multi_field(
        self,
        condition_query: str = "",
        intervention_query: str = "",
        endpoint_query: str = "",
        n_results: int = 100,
        weights: Dict[str, float] = None
    ) -> List[VectorSearchResult]:
        """
        Multi-field semantic search with weighted combination.

        Args:
            condition_query: Query for condition matching
            intervention_query: Query for intervention matching
            endpoint_query: Query for endpoint matching
            n_results: Maximum results
            weights: Field weights (default: condition=0.4, intervention=0.3, endpoint=0.3)

        Returns:
            Combined and re-ranked results
        """
        if weights is None:
            weights = {"condition": 0.4, "intervention": 0.3, "endpoint": 0.3}

        all_scores = {}  # nct_id -> weighted score

        # Search each field
        if condition_query:
            results = self.search(f"Condition: {condition_query}", n_results=n_results * 2)
            for r in results:
                if r.nct_id not in all_scores:
                    all_scores[r.nct_id] = {"scores": {}, "result": r}
                all_scores[r.nct_id]["scores"]["condition"] = r.score

        if intervention_query:
            results = self.search(f"Intervention: {intervention_query}", n_results=n_results * 2)
            for r in results:
                if r.nct_id not in all_scores:
                    all_scores[r.nct_id] = {"scores": {}, "result": r}
                all_scores[r.nct_id]["scores"]["intervention"] = r.score

        if endpoint_query:
            results = self.search(f"Endpoint: {endpoint_query}", n_results=n_results * 2)
            for r in results:
                if r.nct_id not in all_scores:
                    all_scores[r.nct_id] = {"scores": {}, "result": r}
                all_scores[r.nct_id]["scores"]["endpoint"] = r.score

        # Calculate weighted scores
        final_results = []
        for nct_id, data in all_scores.items():
            weighted_score = 0
            total_weight = 0

            for field, weight in weights.items():
                if field in data["scores"]:
                    weighted_score += data["scores"][field] * weight
                    total_weight += weight

            if total_weight > 0:
                final_score = weighted_score / total_weight
                result = data["result"]
                result.score = final_score
                final_results.append(result)

        # Sort by score and return top N
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:n_results]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_trials": self.collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "db_path": self.db_path,
        }

    def is_initialized(self) -> bool:
        """Check if vector store has been populated."""
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def clear(self):
        """Clear all data from the vector store."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self._collection = None
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")


def build_vector_store(db_manager, batch_size: int = 500, limit: int = None, rebuild: bool = False):
    """
    Build the vector store from the trials database incrementally.

    Args:
        db_manager: Database manager instance
        batch_size: Batch size for processing
        limit: Optional limit on number of trials (for testing)
        rebuild: If True, clear and rebuild from scratch
    """
    from sqlalchemy import text

    logger.info("Building vector store from trials database...")

    vector_store = VectorStoreManager()

    # Check current state
    existing_count = 0
    existing_ids = set()

    if vector_store.is_initialized():
        existing_count = vector_store.collection.count()
        logger.info(f"Vector store already has {existing_count} trials")

        if existing_count > 500000 and not rebuild:
            logger.info("Vector store appears complete (>500K trials)")
            return vector_store

        if rebuild:
            logger.info("Rebuild requested - clearing vector store...")
            vector_store.clear()
            existing_count = 0
        else:
            logger.info(f"Continuing from existing {existing_count} trials...")

    # Fetch trials in batches - start from where we left off
    # Calculate starting offset based on existing count (trials are ordered by nct_id)
    offset = existing_count if existing_count > 0 and not rebuild else 0
    total_added = 0
    total_skipped = 0

    logger.info(f"Starting from offset {offset}")

    while True:
        query = text(f"""
            SELECT
                nct_id, title, conditions, interventions, primary_outcomes,
                eligibility_criteria, phase, status, enrollment, sponsor
            FROM trials
            WHERE title IS NOT NULL
            ORDER BY nct_id
            LIMIT {batch_size} OFFSET {offset}
        """)

        trials = db_manager.execute_raw(query.text, {})

        if not trials:
            break

        # Convert to dicts (no need to filter since we start from correct offset)
        trial_dicts = []
        for row in trials:
            trial_dicts.append({
                "nct_id": row[0],
                "title": row[1],
                "conditions": row[2],
                "interventions": row[3],
                "primary_outcomes": row[4],
                "eligibility_criteria": row[5],
                "phase": row[6],
                "status": row[7],
                "enrollment": row[8],
                "sponsor": row[9],
            })

        vector_store.add_trials(trial_dicts, batch_size=batch_size)
        total_added += len(trial_dicts)

        offset += batch_size
        current_total = vector_store.collection.count()

        if total_added > 0 and total_added % 5000 == 0:
            logger.info(f"Progress: Added {total_added}, skipped {total_skipped}, total in store: {current_total}")

        if limit and total_added >= limit:
            break

    final_count = vector_store.collection.count()
    logger.info(f"Vector store build complete. Added {total_added}, skipped {total_skipped}. Total: {final_count}")
    return vector_store


# Singleton instance
_vector_store = None

def get_vector_store() -> VectorStoreManager:
    """Get singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreManager()
    return _vector_store
