"""
Create and populate trial_locations table for geo-filtering.
Extracts location data from JSON in trials.locations column.
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment
with open('.env') as f:
    for line in f:
        if '=' in line and not line.startswith('#'):
            key, value = line.strip().split('=', 1)
            os.environ[key] = value

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_trial_locations():
    """Create and populate the trial_locations table."""
    from sqlalchemy import text
    from src.database import DatabaseManager
    
    db = DatabaseManager.get_instance()
    
    logger.info("Creating trial_locations table...")
    
    with db.engine.connect() as conn:
        # Create table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS trial_locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nct_id VARCHAR(15) NOT NULL,
                facility_name VARCHAR(500),
                city VARCHAR(100),
                state VARCHAR(100),
                country VARCHAR(100),
                zip_code VARCHAR(20),
                contact_name VARCHAR(200),
                contact_phone VARCHAR(50),
                contact_email VARCHAR(200),
                FOREIGN KEY (nct_id) REFERENCES trials(nct_id)
            )
        """))
        
        # Create indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trial_locations_nct_id ON trial_locations(nct_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trial_locations_country ON trial_locations(country)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trial_locations_state ON trial_locations(state)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trial_locations_city ON trial_locations(city)"))
        
        conn.commit()
    
    logger.info("Table created. Checking existing data...")
    
    # Check if already populated
    with db.engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM trial_locations"))
        existing_count = result.scalar()
        
        if existing_count > 100000:
            logger.info(f"Table already has {existing_count:,} records. Skipping population.")
            return
    
    logger.info("Populating trial_locations from JSON data...")
    
    # Process in batches
    batch_size = 1000
    offset = 0
    total_locations = 0
    
    while True:
        with db.engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT nct_id, locations FROM trials 
                WHERE locations IS NOT NULL AND locations != '' AND locations != '[]'
                LIMIT {batch_size} OFFSET {offset}
            """))
            rows = result.fetchall()
        
        if not rows:
            break
        
        locations_to_insert = []
        
        for row in rows:
            nct_id = row[0]
            locations_json = row[1]
            
            try:
                locations = json.loads(locations_json) if isinstance(locations_json, str) else locations_json
                
                if not isinstance(locations, list):
                    continue
                
                for loc in locations:
                    if not isinstance(loc, dict):
                        continue
                    
                    locations_to_insert.append({
                        "nct_id": nct_id,
                        "facility_name": (loc.get("facility") or "")[:500],
                        "city": (loc.get("city") or "")[:100],
                        "state": (loc.get("state") or "")[:100],
                        "country": (loc.get("country") or "")[:100],
                        "zip_code": (loc.get("zip") or "")[:20],
                        "contact_name": (loc.get("contact_name") or "")[:200],
                        "contact_phone": (loc.get("contact_phone") or "")[:50],
                        "contact_email": (loc.get("contact_email") or "")[:200],
                    })
                    
            except (json.JSONDecodeError, TypeError) as e:
                continue
        
        # Bulk insert
        if locations_to_insert:
            with db.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO trial_locations 
                    (nct_id, facility_name, city, state, country, zip_code, contact_name, contact_phone, contact_email)
                    VALUES (:nct_id, :facility_name, :city, :state, :country, :zip_code, :contact_name, :contact_phone, :contact_email)
                """), locations_to_insert)
                conn.commit()
            
            total_locations += len(locations_to_insert)
        
        offset += batch_size
        
        if offset % 10000 == 0:
            logger.info(f"Processed {offset:,} trials, {total_locations:,} locations inserted")
    
    # Final count
    with db.engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM trial_locations"))
        final_count = result.scalar()
    
    logger.info(f"Done! Total locations: {final_count:,}")
    
    # Show sample statistics
    with db.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT country, COUNT(*) as cnt 
            FROM trial_locations 
            GROUP BY country 
            ORDER BY cnt DESC 
            LIMIT 10
        """))
        logger.info("Top countries:")
        for row in result.fetchall():
            logger.info(f"  {row[0]}: {row[1]:,}")


if __name__ == "__main__":
    create_trial_locations()
