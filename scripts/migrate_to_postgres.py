#!/usr/bin/env python3
"""
Migrate SQLite database to PostgreSQL for Railway deployment.

Usage:
    python scripts/migrate_to_postgres.py --postgres-url "postgresql://user:pass@host:port/db"

Or set DATABASE_URL environment variable and run:
    DATABASE_URL="postgresql://..." python scripts/migrate_to_postgres.py
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def migrate_trials(sqlite_path: str, postgres_url: str, batch_size: int = 1000):
    """Migrate trials from SQLite to PostgreSQL."""
    import psycopg2
    from psycopg2.extras import execute_values

    print(f"Source: {sqlite_path}")
    print(f"Target: {postgres_url.split('@')[1] if '@' in postgres_url else postgres_url}")
    print(f"Batch size: {batch_size}")
    print()

    # Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()

    # Connect to PostgreSQL
    pg_conn = psycopg2.connect(postgres_url)
    pg_cursor = pg_conn.cursor()

    # Create tables in PostgreSQL
    print("Creating PostgreSQL tables...")
    pg_cursor.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            nct_id VARCHAR(20) PRIMARY KEY,
            title TEXT,
            status VARCHAR(50),
            phase VARCHAR(50),
            study_type VARCHAR(50),
            conditions TEXT,
            interventions TEXT,
            therapeutic_area VARCHAR(100),
            sponsor VARCHAR(255),
            sponsor_type VARCHAR(50),
            enrollment INTEGER,
            enrollment_type VARCHAR(20),
            start_date VARCHAR(20),
            completion_date VARCHAR(20),
            primary_completion_date VARCHAR(20),
            eligibility_criteria TEXT,
            min_age VARCHAR(20),
            max_age VARCHAR(20),
            sex VARCHAR(20),
            primary_outcomes TEXT,
            secondary_outcomes TEXT,
            locations TEXT,
            num_sites INTEGER,
            why_stopped TEXT,
            has_results BOOLEAN,
            raw_json TEXT,
            ingested_at TIMESTAMP,
            updated_at TIMESTAMP
        )
    """)

    # Create indexes
    print("Creating indexes...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_trials_phase ON trials(phase)",
        "CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status)",
        "CREATE INDEX IF NOT EXISTS idx_trials_therapeutic_area ON trials(therapeutic_area)",
        "CREATE INDEX IF NOT EXISTS idx_trials_sponsor ON trials(sponsor)",
        "CREATE INDEX IF NOT EXISTS idx_trials_status_phase ON trials(status, phase)",
    ]
    for idx in indexes:
        pg_cursor.execute(idx)

    pg_conn.commit()

    # Get total count
    sqlite_cursor.execute("SELECT COUNT(*) FROM trials")
    total = sqlite_cursor.fetchone()[0]
    print(f"Total trials to migrate: {total:,}")
    print()

    # Migrate in batches
    columns = [
        'nct_id', 'title', 'status', 'phase', 'study_type', 'conditions',
        'interventions', 'therapeutic_area', 'sponsor', 'sponsor_type',
        'enrollment', 'enrollment_type', 'start_date', 'completion_date',
        'primary_completion_date', 'eligibility_criteria', 'min_age', 'max_age',
        'sex', 'primary_outcomes', 'secondary_outcomes', 'locations', 'num_sites',
        'why_stopped', 'has_results', 'raw_json', 'ingested_at', 'updated_at'
    ]

    insert_sql = f"""
        INSERT INTO trials ({', '.join(columns)})
        VALUES %s
        ON CONFLICT (nct_id) DO UPDATE SET
            title = EXCLUDED.title,
            status = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at
    """

    offset = 0
    migrated = 0
    start_time = datetime.now()

    while True:
        sqlite_cursor.execute(f"SELECT * FROM trials LIMIT {batch_size} OFFSET {offset}")
        rows = sqlite_cursor.fetchall()

        if not rows:
            break

        # Convert to list of tuples
        values = []
        for row in rows:
            values.append(tuple(row[col] for col in columns))

        # Insert batch
        execute_values(pg_cursor, insert_sql, values, page_size=batch_size)
        pg_conn.commit()

        migrated += len(rows)
        offset += batch_size

        # Progress
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = migrated / elapsed if elapsed > 0 else 0
        eta = (total - migrated) / rate if rate > 0 else 0

        print(f"\rMigrated: {migrated:,}/{total:,} ({100*migrated/total:.1f}%) - {rate:.0f} rows/sec - ETA: {eta/60:.1f} min", end="", flush=True)

    print()
    print()

    # Verify
    pg_cursor.execute("SELECT COUNT(*) FROM trials")
    pg_count = pg_cursor.fetchone()[0]
    print(f"Migration complete! PostgreSQL now has {pg_count:,} trials")

    # Close connections
    sqlite_cursor.close()
    sqlite_conn.close()
    pg_cursor.close()
    pg_conn.close()

    return pg_count


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite to PostgreSQL")
    parser.add_argument(
        "--sqlite-path",
        default="./data/trials.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--postgres-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for migration"
    )

    args = parser.parse_args()

    if not args.postgres_url:
        print("Error: PostgreSQL URL required. Set DATABASE_URL or use --postgres-url")
        sys.exit(1)

    if not os.path.exists(args.sqlite_path):
        print(f"Error: SQLite database not found at {args.sqlite_path}")
        sys.exit(1)

    migrate_trials(args.sqlite_path, args.postgres_url, args.batch_size)


if __name__ == "__main__":
    main()
