"""
ProjectPilot - AI-powered project management system
API endpoints.
"""

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from app import db
from app.models.project import Project
from app.models.task import Task
from app.models.worker_bot import WorkerBot
from werkzeug.exceptions import NotFound, Forbidden, BadRequest
from functools import wraps
import json

api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for AWS ELB"""
    return jsonify({"status": "healthy", "version": "1.0.0"}), 200

# Authentication decorator for API routes
def api_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

# --- Project API endpoints ---

@api_bp.route('/projects', methods=['GET'])
@api_login_required
def get_projects():
    """API endpoint to get all user projects."""
    projects = Project.query.filter_by(owner_id=current_user.id).all()
    return jsonify({
        "success": True,
        "projects": [p.to_dict() for p in projects]
    })

@api_bp.route('/projects/<int:project_id>', methods=['GET'])
@api_login_required
def get_project(project_id):
    """API endpoint to get a specific project."""
    project = Project.query.get_or_404(project_id)
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    return jsonify({
        "success": True,
        "project": project.to_dict()
    })

@api_bp.route('/projects', methods=['POST'])
@api_login_required
def create_project():
    """API endpoint to create a new project."""
    data = request.get_json()
    
    if not data or 'name' not in data:
        return jsonify({"error": "Name is required"}), 400
    
    project = Project(
        name=data['name'],
        description=data.get('description', ''),
        type=data.get('type', 'default'),
        owner_id=current_user.id,
        status='planning'
    )
    
    db.session.add(project)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": f"Project '{data['name']}' created successfully",
        "project": project.to_dict()
    }), 201

@api_bp.route('/projects/<int:project_id>', methods=['PUT'])
@api_login_required
def update_project(project_id):
    """API endpoint to update a project."""
    project = Project.query.get_or_404(project_id)
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No update data provided"}), 400
    
    # Update project attributes
    if 'name' in data:
        project.name = data['name']
    if 'description' in data:
        project.description = data['description']
    if 'type' in data:
        project.type = data['type']
    if 'status' in data:
        project.status = data['status']
    
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": "Project updated successfully",
        "project": project.to_dict()
    })

@api_bp.route('/projects/<int:project_id>', methods=['DELETE'])
@api_login_required
def delete_project(project_id):
    """API endpoint to delete a project."""
    project = Project.query.get_or_404(project_id)
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    db.session.delete(project)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": f"Project '{project.name}' deleted successfully"
    })

# --- Task API endpoints ---

@api_bp.route('/projects/<int:project_id>/tasks', methods=['GET'])
@api_login_required
def get_tasks(project_id):
    """API endpoint to get all tasks for a project."""
    project = Project.query.get_or_404(project_id)
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    tasks = project.tasks.all()
    return jsonify({
        "success": True,
        "tasks": [t.to_dict() for t in tasks]
    })

@api_bp.route('/tasks/<int:task_id>', methods=['GET'])
@api_login_required
def get_task(task_id):
    """API endpoint to get a specific task."""
    task = Task.query.get_or_404(task_id)
    project = task.project
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    return jsonify({
        "success": True,
        "task": task.to_dict()
    })

@api_bp.route('/projects/<int:project_id>/tasks', methods=['POST'])
@api_login_required
def create_task(project_id):
    """API endpoint to create a new task."""
    project = Project.query.get_or_404(project_id)
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    data = request.get_json()
    
    if not data or 'name' not in data:
        return jsonify({"error": "Name is required"}), 400
    
    task = Task(
        name=data['name'],
        description=data.get('description', ''),
        type=data.get('type', 'default'),
        status='pending',
        priority=data.get('priority', 2),
        weight=data.get('weight', 1),
        project_id=project_id
    )
    
    db.session.add(task)
    db.session.commit()
    project.update_progress()
    
    return jsonify({
        "success": True,
        "message": f"Task '{data['name']}' created successfully",
        "task": task.to_dict()
    }), 201

@api_bp.route('/tasks/<int:task_id>', methods=['PUT'])
@api_login_required
def update_task(task_id):
    """API endpoint to update a task."""
    task = Task.query.get_or_404(task_id)
    project = task.project
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No update data provided"}), 400
    
    # Update task attributes
    if 'name' in data:
        task.name = data['name']
    if 'description' in data:
        task.description = data['description']
    if 'type' in data:
        task.type = data['type']
    if 'priority' in data:
        task.priority = data['priority']
    if 'weight' in data:
        task.weight = data['weight']
    if 'status' in data and data['status'] != task.status:
        task.update_status(data['status'])
    
    db.session.commit()
    project.update_progress()
    
    return jsonify({
        "success": True,
        "message": "Task updated successfully",
        "task": task.to_dict()
    })

@api_bp.route('/tasks/<int:task_id>', methods=['DELETE'])
@api_login_required
def delete_task(task_id):
    """API endpoint to delete a task."""
    task = Task.query.get_or_404(task_id)
    project = task.project
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    db.session.delete(task)
    db.session.commit()
    project.update_progress()
    
    return jsonify({
        "success": True,
        "message": f"Task '{task.name}' deleted successfully"
    })

# --- Worker Bot API endpoints ---

@api_bp.route('/projects/<int:project_id>/bots', methods=['GET'])
@api_login_required
def get_worker_bots(project_id):
    """API endpoint to get all worker bots for a project."""
    project = Project.query.get_or_404(project_id)
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    bots = project.worker_bots.all()
    return jsonify({
        "success": True,
        "worker_bots": [b.to_dict() for b in bots]
    })

@api_bp.route('/bots/<int:bot_id>', methods=['GET'])
@api_login_required
def get_worker_bot(bot_id):
    """API endpoint to get a specific worker bot."""
    bot = WorkerBot.query.get_or_404(bot_id)
    project = Project.query.get(bot.project_id)
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    return jsonify({
        "success": True,
        "worker_bot": bot.to_dict()
    })

@api_bp.route('/projects/<int:project_id>/bots', methods=['POST'])
@api_login_required
def create_worker_bot(project_id):
    """API endpoint to create a new worker bot."""
    project = Project.query.get_or_404(project_id)
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    data = request.get_json()
    
    if not data or 'name' not in data or 'type' not in data:
        return jsonify({"error": "Name and type are required"}), 400
    
    # Define bot capabilities based on type
    capabilities = {
        'architect': ['system_design', 'requirements_analysis', 'task_decomposition'],
        'developer': ['coding', 'refactoring', 'debugging'],
        'tester': ['test_creation', 'bug_finding', 'quality_assurance'],
        'devops': ['deployment', 'infrastructure', 'monitoring']
    }.get(data['type'], [])
    
    bot = WorkerBot(
        name=data['name'],
        type=data['type'],
        ai_provider=data.get('ai_provider', 'openai'),
        ai_model=data.get('ai_model', 'gpt-4'),
        capabilities=capabilities,
        status='idle',
        project_id=project_id,
        created_by_id=current_user.id
    )
    
    db.session.add(bot)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": f"Worker bot '{data['name']}' created successfully",
        "worker_bot": bot.to_dict()
    }), 201

@api_bp.route('/bots/<int:bot_id>', methods=['PUT'])
@api_login_required
def update_worker_bot(bot_id):
    """API endpoint to update a worker bot."""
    bot = WorkerBot.query.get_or_404(bot_id)
    project = Project.query.get(bot.project_id)
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No update data provided"}), 400
    
    # Update bot attributes
    if 'name' in data:
        bot.name = data['name']
    if 'status' in data:
        bot.update_status(data['status'])
    if 'ai_provider' in data:
        bot.ai_provider = data['ai_provider']
    if 'ai_model' in data:
        bot.ai_model = data['ai_model']
    
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": "Worker bot updated successfully",
        "worker_bot": bot.to_dict()
    })

@api_bp.route('/bots/<int:bot_id>', methods=['DELETE'])
@api_login_required
def delete_worker_bot(bot_id):
    """API endpoint to delete a worker bot."""
    bot = WorkerBot.query.get_or_404(bot_id)
    project = Project.query.get(bot.project_id)
    
    if project.owner_id != current_user.id and not current_user.is_admin:
        return jsonify({"error": "Permission denied"}), 403
    
    # Check if bot has assigned tasks
    if bot.assigned_tasks.count() > 0:
        return jsonify({
            "error": "Worker bot has assigned tasks. Reassign or complete tasks before deleting."
        }), 400
    
    db.session.delete(bot)
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": f"Worker bot '{bot.name}' deleted successfully"
    })