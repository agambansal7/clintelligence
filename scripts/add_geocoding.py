"""
Add latitude/longitude columns to trial_locations and geocode US ZIP codes.
This enables distance-based trial searching for patients.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# US ZIP code coordinates (subset for common locations)
# In production, use a complete ZIP database or geocoding service
US_ZIP_COORDINATES = {
    # Major cities - approximate center coordinates
    "10001": (40.7484, -73.9967),  # New York
    "90001": (33.9425, -118.2551),  # Los Angeles
    "60601": (41.8819, -87.6278),  # Chicago
    "77001": (29.7604, -95.3698),  # Houston
    "85001": (33.4484, -112.0740),  # Phoenix
    "19101": (39.9526, -75.1652),  # Philadelphia
    "78201": (29.4241, -98.4936),  # San Antonio
    "92101": (32.7157, -117.1611),  # San Diego
    "75201": (32.7767, -96.7970),  # Dallas
    "95101": (37.3382, -121.8863),  # San Jose
    "94102": (37.7749, -122.4194),  # San Francisco
    "98101": (47.6062, -122.3321),  # Seattle
    "80201": (39.7392, -104.9903),  # Denver
    "02101": (42.3601, -71.0589),  # Boston
    "20001": (38.9072, -77.0369),  # Washington DC
    "30301": (33.7490, -84.3880),  # Atlanta
    "33101": (25.7617, -80.1918),  # Miami
    "48201": (42.3314, -83.0458),  # Detroit
    "55401": (44.9778, -93.2650),  # Minneapolis
    "28201": (35.2271, -80.8431),  # Charlotte
}

# Regional approximations based on ZIP prefix
REGIONAL_COORDS = {
    # Northeast
    "010": (42.1, -72.6), "011": (42.2, -72.5), "012": (42.4, -73.2),
    "020": (42.3, -71.1), "021": (42.4, -71.0),
    # New York
    "100": (40.8, -74.0), "101": (40.7, -73.9), "102": (40.7, -74.0),
    "110": (40.8, -73.5), "111": (40.7, -73.7), "112": (40.6, -73.9),
    # Mid-Atlantic
    "190": (40.0, -75.2), "191": (40.0, -75.1),
    "200": (38.9, -77.0), "201": (38.9, -77.0),
    "210": (39.3, -76.6), "211": (39.2, -76.7),
    # Southeast
    "300": (33.8, -84.4), "303": (33.7, -84.4),
    "320": (30.3, -81.7), "327": (28.5, -81.4),
    "330": (25.8, -80.2), "331": (25.8, -80.2),
    # Texas
    "750": (32.8, -97.0), "751": (32.8, -96.8),
    "770": (29.8, -95.4), "771": (29.8, -95.4),
    "780": (29.4, -98.5), "786": (30.3, -97.7),
    # California
    "900": (34.0, -118.3), "901": (34.0, -118.3),
    "920": (32.7, -117.2), "921": (32.8, -117.2),
    "940": (37.8, -122.4), "941": (37.8, -122.4),
    "950": (37.3, -121.9), "951": (37.4, -122.1),
    # Mountain/West
    "800": (39.7, -105.0), "801": (39.7, -105.0),
    "840": (40.8, -111.9), "850": (33.4, -112.1),
    # Pacific Northwest
    "970": (45.5, -122.7), "980": (47.6, -122.3),
}


def get_coords_from_zip(zip_code):
    """Get coordinates from a ZIP code using lookup or regional approximation."""
    if not zip_code:
        return None, None

    zip_clean = zip_code.strip()[:5]

    # Direct lookup
    if zip_clean in US_ZIP_COORDINATES:
        return US_ZIP_COORDINATES[zip_clean]

    # Regional approximation
    zip_prefix = zip_clean[:3]
    if zip_prefix in REGIONAL_COORDS:
        return REGIONAL_COORDS[zip_prefix]

    return None, None


# Major US cities with coordinates
US_CITY_COORDS = {
    # Format: "city_state": (lat, lon)
    # Northeast
    "new york_new york": (40.7128, -74.0060),
    "manhattan_new york": (40.7831, -73.9712),
    "brooklyn_new york": (40.6782, -73.9442),
    "bronx_new york": (40.8448, -73.8648),
    "boston_massachusetts": (42.3601, -71.0589),
    "cambridge_massachusetts": (42.3736, -71.1097),
    "philadelphia_pennsylvania": (39.9526, -75.1652),
    "pittsburgh_pennsylvania": (40.4406, -79.9959),
    "baltimore_maryland": (39.2904, -76.6122),
    "bethesda_maryland": (38.9847, -77.0947),
    "washington_district of columbia": (38.9072, -77.0369),
    "washington d.c._district of columbia": (38.9072, -77.0369),
    "newark_new jersey": (40.7357, -74.1724),
    "hackensack_new jersey": (40.8859, -74.0435),
    "new haven_connecticut": (41.3083, -72.9279),
    "hartford_connecticut": (41.7658, -72.6734),
    "providence_rhode island": (41.8240, -71.4128),
    "portland_maine": (43.6591, -70.2568),
    "burlington_vermont": (44.4759, -73.2121),
    # Southeast
    "atlanta_georgia": (33.7490, -84.3880),
    "miami_florida": (25.7617, -80.1918),
    "tampa_florida": (27.9506, -82.4572),
    "orlando_florida": (28.5383, -81.3792),
    "jacksonville_florida": (30.3322, -81.6557),
    "gainesville_florida": (29.6516, -82.3248),
    "charlotte_north carolina": (35.2271, -80.8431),
    "raleigh_north carolina": (35.7796, -78.6382),
    "durham_north carolina": (35.9940, -78.8986),
    "chapel hill_north carolina": (35.9132, -79.0558),
    "winston-salem_north carolina": (36.0999, -80.2442),
    "nashville_tennessee": (36.1627, -86.7816),
    "memphis_tennessee": (35.1495, -90.0490),
    "birmingham_alabama": (33.5207, -86.8025),
    "louisville_kentucky": (38.2527, -85.7585),
    "charleston_south carolina": (32.7765, -79.9311),
    "columbia_south carolina": (34.0007, -81.0348),
    "richmond_virginia": (37.5407, -77.4360),
    "charlottesville_virginia": (38.0293, -78.4767),
    "norfolk_virginia": (36.8508, -76.2859),
    "new orleans_louisiana": (29.9511, -90.0715),
    "jackson_mississippi": (32.2988, -90.1848),
    "little rock_arkansas": (34.7465, -92.2896),
    # Midwest
    "chicago_illinois": (41.8781, -87.6298),
    "detroit_michigan": (42.3314, -83.0458),
    "ann arbor_michigan": (42.2808, -83.7430),
    "grand rapids_michigan": (42.9634, -85.6681),
    "cleveland_ohio": (41.4993, -81.6944),
    "columbus_ohio": (39.9612, -82.9988),
    "cincinnati_ohio": (39.1031, -84.5120),
    "indianapolis_indiana": (39.7684, -86.1581),
    "milwaukee_wisconsin": (43.0389, -87.9065),
    "madison_wisconsin": (43.0731, -89.4012),
    "minneapolis_minnesota": (44.9778, -93.2650),
    "st. paul_minnesota": (44.9537, -93.0900),
    "rochester_minnesota": (44.0121, -92.4802),
    "st. louis_missouri": (38.6270, -90.1994),
    "kansas city_missouri": (39.0997, -94.5786),
    "kansas city_kansas": (39.1141, -94.6275),
    "omaha_nebraska": (41.2565, -95.9345),
    "des moines_iowa": (41.5868, -93.6250),
    "iowa city_iowa": (41.6611, -91.5302),
    # Southwest
    "dallas_texas": (32.7767, -96.7970),
    "houston_texas": (29.7604, -95.3698),
    "san antonio_texas": (29.4241, -98.4936),
    "austin_texas": (30.2672, -97.7431),
    "fort worth_texas": (32.7555, -97.3308),
    "el paso_texas": (31.7619, -106.4850),
    "phoenix_arizona": (33.4484, -112.0740),
    "tucson_arizona": (32.2226, -110.9747),
    "scottsdale_arizona": (33.4942, -111.9261),
    "albuquerque_new mexico": (35.0844, -106.6504),
    "las vegas_nevada": (36.1699, -115.1398),
    "denver_colorado": (39.7392, -104.9903),
    "aurora_colorado": (39.7294, -104.8319),
    "boulder_colorado": (40.0150, -105.2705),
    "colorado springs_colorado": (38.8339, -104.8214),
    "salt lake city_utah": (40.7608, -111.8910),
    "oklahoma city_oklahoma": (35.4676, -97.5164),
    "tulsa_oklahoma": (36.1540, -95.9928),
    # West Coast
    "los angeles_california": (34.0522, -118.2437),
    "san francisco_california": (37.7749, -122.4194),
    "san diego_california": (32.7157, -117.1611),
    "san jose_california": (37.3382, -121.8863),
    "sacramento_california": (38.5816, -121.4944),
    "fresno_california": (36.7378, -119.7871),
    "oakland_california": (37.8044, -122.2712),
    "palo alto_california": (37.4419, -122.1430),
    "stanford_california": (37.4275, -122.1697),
    "santa monica_california": (34.0195, -118.4912),
    "la jolla_california": (32.8328, -117.2713),
    "irvine_california": (33.6846, -117.8265),
    "duarte_california": (34.1395, -117.9773),
    "seattle_washington": (47.6062, -122.3321),
    "tacoma_washington": (47.2529, -122.4443),
    "portland_oregon": (45.5152, -122.6784),
    "honolulu_hawaii": (21.3069, -157.8583),
    "anchorage_alaska": (61.2181, -149.9003),
    # Other major research centers
    "rochester_new york": (43.1566, -77.6088),
    "buffalo_new york": (42.8864, -78.8784),
    "albany_new york": (42.6526, -73.7562),
    "worcester_massachusetts": (42.2626, -71.8023),
    "springfield_massachusetts": (42.1015, -72.5898),
    "new brunswick_new jersey": (40.4862, -74.4518),
    "newtown_pennsylvania": (40.2293, -74.9362),
    "hershey_pennsylvania": (40.2856, -76.6508),
    "winston salem_north carolina": (36.0999, -80.2442),
    "greenville_south carolina": (34.8526, -82.3940),
    "lexington_kentucky": (38.0406, -84.5037),
    "knoxville_tennessee": (35.9606, -83.9207),
    "chattanooga_tennessee": (35.0456, -85.3097),
    "mobile_alabama": (30.6954, -88.0399),
    "baton rouge_louisiana": (30.4515, -91.1871),
    "gainesville_georgia": (34.2979, -83.8241),
    "augusta_georgia": (33.4735, -82.0105),
    "savannah_georgia": (32.0809, -81.0912),
    "toledo_ohio": (41.6528, -83.5379),
    "akron_ohio": (41.0814, -81.5190),
    "dayton_ohio": (39.7589, -84.1916),
    "peoria_illinois": (40.6936, -89.5890),
    "springfield_illinois": (39.7817, -89.6501),
    "evanston_illinois": (42.0451, -87.6877),
    "maywood_illinois": (41.8792, -87.8431),
    "wichita_kansas": (37.6872, -97.3301),
    "st. louis_illinois": (38.6270, -90.1994),
    "fargo_north dakota": (46.8772, -96.7898),
    "sioux falls_south dakota": (43.5446, -96.7311),
    "billings_montana": (45.7833, -108.5007),
    "boise_idaho": (43.6150, -116.2023),
    "spokane_washington": (47.6588, -117.4260),
    "eugene_oregon": (44.0521, -123.0868),
    "reno_nevada": (39.5296, -119.8138),
}


def get_coords_from_city_state(city, state):
    """Get coordinates from city and state name."""
    if not city or not state:
        return None, None

    # Normalize
    city_clean = city.strip().lower()
    state_clean = state.strip().lower()

    # Try exact match
    key = f"{city_clean}_{state_clean}"
    if key in US_CITY_COORDS:
        return US_CITY_COORDS[key]

    # Try without punctuation
    city_clean = city_clean.replace(".", "").replace("-", " ")
    key = f"{city_clean}_{state_clean}"
    if key in US_CITY_COORDS:
        return US_CITY_COORDS[key]

    # Try matching just the city (for major cities)
    for full_key, coords in US_CITY_COORDS.items():
        if full_key.startswith(f"{city_clean}_"):
            return coords

    return None, None


def add_geocoding():
    """Add lat/lon columns and populate from city/state lookup."""
    from sqlalchemy import text
    from src.database import DatabaseManager

    db = DatabaseManager.get_instance()

    # Step 1: Add columns if they don't exist
    logger.info("Adding latitude/longitude columns...")

    with db.engine.connect() as conn:
        # Check if columns exist
        result = conn.execute(text("PRAGMA table_info(trial_locations)"))
        columns = [row[1] for row in result.fetchall()]

        if 'latitude' not in columns:
            conn.execute(text("ALTER TABLE trial_locations ADD COLUMN latitude REAL"))
            logger.info("Added latitude column")

        if 'longitude' not in columns:
            conn.execute(text("ALTER TABLE trial_locations ADD COLUMN longitude REAL"))
            logger.info("Added longitude column")

        conn.commit()

    # Step 2: Geocode US locations using city/state lookup
    logger.info("Geocoding US locations using city/state lookup...")

    # Get count of US locations without coordinates
    with db.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM trial_locations
            WHERE country = 'United States'
            AND (latitude IS NULL OR longitude IS NULL)
        """))
        to_geocode = result.scalar()

    logger.info(f"Found {to_geocode:,} US locations to geocode")

    if to_geocode == 0:
        logger.info("No locations need geocoding")
        return

    # Process in batches
    batch_size = 5000
    total_geocoded = 0

    while True:
        with db.engine.connect() as conn:
            # Get batch of locations needing geocoding
            result = conn.execute(text("""
                SELECT id, city, state FROM trial_locations
                WHERE country = 'United States'
                AND (latitude IS NULL OR longitude IS NULL)
                LIMIT :batch_size
            """), {"batch_size": batch_size})
            rows = result.fetchall()

        if not rows:
            break

        updates = []
        for row in rows:
            loc_id, city, state = row
            lat, lon = get_coords_from_city_state(city, state)
            if lat and lon:
                updates.append({"id": loc_id, "lat": lat, "lon": lon})

        # Bulk update
        if updates:
            with db.engine.connect() as conn:
                for update in updates:
                    conn.execute(text("""
                        UPDATE trial_locations
                        SET latitude = :lat, longitude = :lon
                        WHERE id = :id
                    """), update)
                conn.commit()

            total_geocoded += len(updates)
            logger.info(f"Geocoded {total_geocoded:,} locations...")
        else:
            # No more matches found - exit
            logger.info(f"No more cities matched. Stopping with {total_geocoded:,} total geocoded.")
            break

        # Safety break for very large datasets
        if total_geocoded >= 500000:
            logger.info("Reached 500k limit, stopping...")
            break

    # Final stats
    with db.engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) FROM trial_locations
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """))
        with_coords = result.scalar()

        result = conn.execute(text("SELECT COUNT(*) FROM trial_locations"))
        total = result.scalar()

    logger.info(f"Done! {with_coords:,} of {total:,} locations have coordinates ({100*with_coords/total:.1f}%)")


if __name__ == "__main__":
    add_geocoding()
