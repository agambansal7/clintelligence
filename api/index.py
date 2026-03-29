"""
Vercel Serverless Function Entry Point
"""
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_app.main import app

# Export for Vercel
handler = app
