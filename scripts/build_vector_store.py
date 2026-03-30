#!/usr/bin/env python3
"""
Build Vector Store for Semantic Trial Search

This script generates embeddings for all trials and stores them in ChromaDB.
Run this once to enable semantic search functionality.

Usage:
    python3 scripts/build_vector_store.py [--limit N] [--batch-size N]
"""

import os
import sys
import argparse
import logging
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build vector store for trial search")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trials (for testing)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for processing")
    parser.add_argument("--rebuild", action="store_true", help="Clear and rebuild entire store")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Building Vector Store for Clinical Trials")
    logger.info("=" * 60)

    # Load environment
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

    # Import components
    from src.database import DatabaseManager
    from src.analysis.vector_store import VectorStoreManager, build_vector_store

    # Check database
    logger.info("Connecting to database...")
    db = DatabaseManager.get_instance()
    stats = db.get_stats()
    total_trials = stats.get('total_trials', 0)
    logger.info(f"Database contains {total_trials:,} trials")

    # Initialize vector store
    vector_store = VectorStoreManager()

    # Check current state
    if vector_store.is_initialized():
        current_count = vector_store.collection.count()
        logger.info(f"Vector store already has {current_count:,} trials")

        if current_count >= total_trials * 0.95 and not args.rebuild:  # 95% complete
            logger.info("Vector store appears complete. Use --rebuild to regenerate.")
            return
    else:
        logger.info("Vector store is empty, will build from scratch")

    # Build vector store incrementally (or rebuild if requested)
    start_time = time.time()

    limit = args.limit if args.limit else None
    logger.info(f"Building vector store (limit={limit}, batch_size={args.batch_size}, rebuild={args.rebuild})...")

    final_store = build_vector_store(db, batch_size=args.batch_size, limit=limit, rebuild=args.rebuild)

    elapsed = time.time() - start_time
    final_count = final_store.collection.count()

    logger.info("=" * 60)
    logger.info(f"Vector store build complete!")
    logger.info(f"Total trials indexed: {final_count:,}")
    logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"Rate: {final_count/elapsed:.1f} trials/second")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
