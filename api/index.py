"""
Vercel Serverless Function Entry Point
Uses Mangum to adapt FastAPI ASGI app for serverless
"""
import os
import sys

# Add project root to path for imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Set environment variables
os.environ.setdefault('PYTHONPATH', ROOT_DIR)

from mangum import Mangum
from web_app.main import app

# Create the handler for Vercel/AWS Lambda
handler = Mangum(app, lifespan="off")
