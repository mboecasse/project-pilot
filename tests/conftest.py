"""
Configuration and fixtures for pytest.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
import tempfile
from flask import Flask

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from app
from app import create_app, db
from app.models.user import User
from app.models.project import Project
from app.models.task import Task
from app.models.worker_bot import WorkerBot

@pytest.fixture
def app():
    """Create and configure a Flask app for testing."""
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp()
    
    app = create_app({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test-key'
    })
    
    # Create the database and tables
    with app.app_context():
        db.create_all()
    
    yield app
    
    # Close and remove the temporary database
    os.close(db_fd)
    os.unlink(db_path)

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """A CLI test runner for the app."""
    return app.test_cli_runner()

@pytest.fixture
def _db(app):
    """Provide the database engine to test functions."""
    with app.app_context():
        yield db

@pytest.fixture
def mock_current_user():
    """Mock the current_user from Flask-Login."""
    with patch('flask_login.utils._get_user') as mock_get_user:
        user = MagicMock()
        user.id = 1
        user.username = 'testuser'
        user.email = 'test@example.com'
        user.is_authenticated = True
        mock_get_user.return_value = user
        yield user

@pytest.fixture
def sample_data(_db):
    """Create sample data for testing."""
    with _db.session.begin_nested():
        # Create sample user
        user = User(username='testuser', email='test@example.com')
        user.set_password('password')
        _db.session.add(user)
        _db.session.flush()
        
        # Create sample project
        project = Project(
            name='Test Project',
            description='This is a test project',
            owner_id=user.id
        )
        _db.session.add(project)
        _db.session.flush()
        
        # Create sample worker bots
        developer_bot = WorkerBot(
            name='Developer Bot',
            type='developer',
            status='idle',
            project_id=project.id,
            capabilities=['coding', 'testing'],
            ai_provider='openai',
            ai_model='gpt-4'
        )
        
        tester_bot = WorkerBot(
            name='Tester Bot',
            type='tester',
            status='idle',
            project_id=project.id,
            capabilities=['testing'],
            ai_provider='anthropic',
            ai_model='claude-3-opus-20240229'
        )
        
        _db.session.add(developer_bot)
        _db.session.add(tester_bot)
        _db.session.flush()
        
        # Create sample tasks
        task1 = Task(
            name='Implement Feature',
            description='Implement new feature X',
            type='feature',
            status='pending',
            priority=1,
            project_id=project.id,
            assigned_to_bot_id=developer_bot.id
        )
        
        task2 = Task(
            name='Fix Bug',
            description='Fix bug in module Y',
            type='bug',
            status='pending',
            priority=2,
            project_id=project.id,
            assigned_to_bot_id=None
        )
        
        _db.session.add(task1)
        _db.session.add(task2)
    
    # Return the created data
    return {
        'user': user,
        'project': project,
        'bots': [developer_bot, tester_bot],
        'tasks': [task1, task2]
    }

@pytest.fixture
def mock_ai_provider():
    """Mock the AIProvider class."""
    with patch('app.services._provider.AIProvider') as mock_provider_class:
        mock_provider = MagicMock()
        mock_provider.generate_text.return_value = {
            'text': 'This is a test response from the AI provider.',
            'provider': 'openai',
            'model': 'gpt-4'
        }
        mock_provider_class.return_value = mock_provider
        yield mock_provider