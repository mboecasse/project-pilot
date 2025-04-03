"""
ProjectPilot - AI-powered project management system
Configuration for development, testing, and production environments.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration."""
    # Secret key for signing sessions
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-insecure')
    
    # SQLAlchemy settings
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///projectpilot.db')
    
    # AWS settings
    AWS_REGION = os.environ.get('AWS_REGION', 'eu-west-2')
    SECRET_NAME = os.environ.get('SECRET_NAME', 'manager-bot-secrets')
    
    # AI API settings
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
    AWS_BEDROCK_ENABLED = os.environ.get('AWS_BEDROCK_ENABLED', 'false').lower() == 'true'
    GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
    S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', '')
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Upload folder
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Try to load secrets from AWS Secrets Manager in production
    @classmethod
    def init_app(cls, app):
        """Initialize app with this config"""
        if not app.debug and not app.testing:
            try:
                from app.utils.secrets import get_secret
                secrets = get_secret(cls.SECRET_NAME, cls.AWS_REGION)
                
                # Update config with secrets if available
                if secrets:
                    cls.SECRET_KEY = secrets.get('SLACK_SIGNING_SECRET', cls.SECRET_KEY)
                    cls.SQLALCHEMY_DATABASE_URI = secrets.get('DATABASE_URL', cls.SQLALCHEMY_DATABASE_URI)
                    cls.OPENAI_API_KEY = secrets.get('OPENAI_API_KEY', cls.OPENAI_API_KEY)
                    cls.ANTHROPIC_API_KEY = secrets.get('ANTHROPIC_API_KEY', cls.ANTHROPIC_API_KEY)
                    cls.GITHUB_TOKEN = secrets.get('GITHUB_TOKEN', os.environ.get('GITHUB_TOKEN', ''))
                    cls.S3_BUCKET_NAME = secrets.get('S3_BUCKET_NAME', os.environ.get('S3_BUCKET_NAME', ''))
                    
                    app.logger.info(f"Configuration loaded from Secrets Manager: {cls.SECRET_NAME}")
            except Exception as e:
                app.logger.error(f"Error loading secrets: {str(e)}")


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    FLASK_ENV = 'development'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL', 'sqlite:///dev-projectpilot.db')
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Development-specific setup
        app.logger.info("Running in development mode")


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL', 'sqlite:///test-projectpilot.db')
    WTF_CSRF_ENABLED = False
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Testing-specific setup
        app.logger.info("Running in testing mode")


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    FLASK_ENV = 'production'
    
    # Override with RDS in production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', '')
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Production-specific setup
        import logging
        from logging.handlers import RotatingFileHandler
        
        # File handler for production logs
        file_handler = RotatingFileHandler('logs/projectpilot.log', maxBytes=10*1024*1024, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info("ProjectPilot startup")


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}