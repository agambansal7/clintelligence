"""
Rebuild embeddings using OpenAI text-embedding-3-small model.
This will create a new collection with better medical domain understanding.
"""

import os
import sys
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def rebuild_embeddings(batch_size: int = 100, limit: int = None):
    """Rebuild all trial embeddings using OpenAI."""
    from sqlalchemy import text
    from src.database import DatabaseManager
    from src.analysis.vector_store import VectorStoreManager, COLLECTION_NAME

    logger.info("=" * 60)
    logger.info("Starting OpenAI Embeddings Rebuild")
    logger.info(f"Collection: {COLLECTION_NAME}")
    logger.info("=" * 60)

    # Initialize
    db = DatabaseManager.get_instance()
    vector_store = VectorStoreManager()

    # Check if collection exists and get current count
    try:
        existing_count = vector_store.collection.count()
        logger.info(f"Existing embeddings in collection: {existing_count}")
    except Exception:
        existing_count = 0
        logger.info("Starting fresh collection")

    # Get total trial count
    with db.engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM trials WHERE title IS NOT NULL"))
        total_trials = result.scalar()

    logger.info(f"Total trials to embed: {total_trials}")

    if limit:
        total_trials = min(total_trials, limit)
        logger.info(f"Limited to: {total_trials}")

    # Process in batches
    offset = existing_count  # Resume from where we left off
    processed = 0
    start_time = time.time()

    while offset < total_trials:
        batch_start = time.time()

        # Fetch batch of trials
        query = text("""
            SELECT
                nct_id, title, conditions, interventions, primary_outcomes,
                eligibility_criteria, phase, status, enrollment, sponsor
            FROM trials
            WHERE title IS NOT NULL
            ORDER BY nct_id
            LIMIT :limit OFFSET :offset
        """)

        with db.engine.connect() as conn:
            result = conn.execute(query, {"limit": batch_size, "offset": offset})
            rows = result.fetchall()

        if not rows:
            break

        # Prepare trial data
        trial_dicts = []
        for row in rows:
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

        # Add to vector store (this calls OpenAI API)
        try:
            vector_store.add_trials(trial_dicts, batch_size=batch_size)
            processed += len(trial_dicts)
            offset += batch_size

            # Progress reporting
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta_seconds = (total_trials - offset) / rate if rate > 0 else 0
            eta_hours = eta_seconds / 3600

            current_count = vector_store.collection.count()
            pct = (current_count / total_trials) * 100

            batch_time = time.time() - batch_start

            logger.info(
                f"Progress: {current_count:,}/{total_trials:,} ({pct:.1f}%) | "
                f"Rate: {rate:.1f}/sec | ETA: {eta_hours:.1f}h | "
                f"Batch: {batch_time:.1f}s"
            )

        except Exception as e:
            logger.error(f"Error processing batch at offset {offset}: {e}")
            # Wait and retry
            time.sleep(5)
            continue

        # Small delay between batches to avoid rate limits
        time.sleep(0.2)

    # Final stats
    final_count = vector_store.collection.count()
    total_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("Rebuild Complete!")
    logger.info(f"Total embeddings: {final_count:,}")
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild embeddings with OpenAI")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trials")

    args = parser.parse_args()

    rebuild_embeddings(batch_size=args.batch_size, limit=args.limit)
