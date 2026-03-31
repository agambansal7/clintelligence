#!/usr/bin/env python3
"""
Migrate local SQLite database to Railway PostgreSQL.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from src.database.models import Base
from tqdm import tqdm

# Local SQLite
LOCAL_DB = "sqlite:///./data/trials.db"

# Get PostgreSQL URL from environment
POSTGRES_URL = os.environ.get("DATABASE_URL")

if not POSTGRES_URL:
    print("ERROR: DATABASE_URL environment variable not set")
    print("Copy the PostgreSQL URL from 'railway variables' and run:")
    print("  export DATABASE_URL='postgresql://...'")
    sys.exit(1)

# Fix for Railway's postgres:// vs postgresql://
if POSTGRES_URL.startswith("postgres://"):
    POSTGRES_URL = POSTGRES_URL.replace("postgres://", "postgresql://", 1)

print(f"Source: {LOCAL_DB}")
print(f"Target: {POSTGRES_URL[:60]}...")

# Connect to both databases
sqlite_engine = create_engine(LOCAL_DB)
postgres_engine = create_engine(POSTGRES_URL)

# Create tables in PostgreSQL
print("\nCreating tables in PostgreSQL...")
Base.metadata.create_all(bind=postgres_engine)
print("Tables created!")

# Get list of tables to migrate
inspector = inspect(sqlite_engine)
tables = inspector.get_table_names()
print(f"\nTables to migrate: {tables}")

# Migrate each table
for table_name in tables:
    print(f"\n{'='*60}")
    print(f"Migrating table: {table_name}")
    print('='*60)

    # Count rows
    with sqlite_engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    print(f"Rows to migrate: {count:,}")

    if count == 0:
        print("Skipping empty table")
        continue

    # Get columns
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    columns_str = ', '.join(columns)
    placeholders = ', '.join([f':{col}' for col in columns])

    # Build upsert query for PostgreSQL
    # For trials table, use nct_id as conflict key
    # For other tables, just insert
    if table_name == 'trials':
        insert_sql = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT (nct_id) DO NOTHING
        """
    else:
        insert_sql = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
        """

    # Batch migrate
    batch_size = 500
    offset = 0
    migrated = 0

    with tqdm(total=count, desc=f"Migrating {table_name}") as pbar:
        while offset < count:
            # Fetch batch from SQLite
            with sqlite_engine.connect() as sqlite_conn:
                rows = sqlite_conn.execute(
                    text(f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}")
                ).fetchall()

            if not rows:
                break

            # Insert into PostgreSQL
            with postgres_engine.connect() as pg_conn:
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    try:
                        pg_conn.execute(text(insert_sql), row_dict)
                    except Exception as e:
                        # Skip problematic rows
                        pass
                pg_conn.commit()

            migrated += len(rows)
            offset += batch_size
            pbar.update(len(rows))

    # Verify
    with postgres_engine.connect() as conn:
        pg_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    print(f"Migrated: {pg_count:,} rows")

print("\n" + "="*60)
print("MIGRATION COMPLETE!")
print("="*60)

# Final stats
with postgres_engine.connect() as conn:
    for table_name in tables:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        print(f"  {table_name}: {count:,} rows")

print("\nYour Railway PostgreSQL database is now populated!")
