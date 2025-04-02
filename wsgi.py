"""
ProjectPilot - AI-powered project management system
Main application entry point for AWS Elastic Beanstalk
"""

import os
import sys
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add the project directory to the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from app import create_app

application = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    application.run(host='0.0.0.0', port=port)