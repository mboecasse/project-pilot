"""
ProjectPilot - AI-powered project management system
Application factory pattern implementation.
"""

import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()

def create_app(config_name=None):
    """Application factory function"""
    app = Flask(__name__)
    
    # Configure the app
    from app.config import config
    config_name = config_name or os.getenv('FLASK_CONFIG', 'default')
    app.config.from_object(config[config_name])
    
    # Initialize logging
    configure_logging(app)
    
    # Initialize extensions
    initialize_extensions(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Create database tables if in development
    initialize_database(app)
    
    # Log successful application creation
    app.logger.info(f"ProjectPilot application initialized with config: {config_name}")
    
    return app

def configure_logging(app):
    """Configure application logging"""
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
    app.logger.setLevel(log_level)
    
    # Add handlers as needed
    if not app.debug and not app.testing:
        # For production, add additional handlers here if needed
        pass

def initialize_extensions(app):
    """Initialize Flask extensions"""
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Configure Login Manager
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        # Import here to avoid circular imports
        from app.models.user import User
        return User.query.get(int(user_id))

def register_blueprints(app):
    """Register Flask blueprints"""
    # Import blueprints
    from app.routes.auth import auth_bp
    from app.routes.projects import projects_bp
    from app.routes.tasks import tasks_bp
    from app.routes.api import api_bp
    from app.routes.dashboard import dashboard_bp
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(dashboard_bp)

def register_error_handlers(app):
    """Register error handlers"""
    @app.errorhandler(404)
    def page_not_found(e):
        return {"error": "Not found"}, 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        app.logger.error(f"Server error: {str(e)}")
        return {"error": "Internal server error"}, 500

def initialize_database(app):
    """Initialize database tables"""
    with app.app_context():
        if app.config.get('FLASK_ENV') == 'development':
            db.create_all()