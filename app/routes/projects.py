"""
ProjectPilot - AI-powered project management system
Project management routes.
"""

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from app import db
from app.models.project import Project
from app.models.task import Task
from app.models.worker_bot import WorkerBot
from datetime import datetime

projects_bp = Blueprint('projects', __name__, url_prefix='/projects')

@projects_bp.route('/')
@login_required
def index():
    """List all projects."""
    projects = Project.query.filter_by(owner_id=current_user.id).all()
    return render_template('projects/list.html', projects=projects)

@projects_bp.route('/new', methods=['GET', 'POST'])
@login_required
def new_project():
    """Create a new project."""
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        project_type = request.form.get('type')
        start_date_str = request.form.get('start_date')
        
        if not name:
            flash('Project name is required.', 'danger')
            return render_template('projects/new.html')
        
        project = Project(
            name=name,
            description=description,
            type=project_type,
            owner_id=current_user.id,
            status='planning'
        )
        
        if start_date_str:
            try:
                project.start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            except ValueError:
                flash('Invalid start date format.', 'warning')
        
        db.session.add(project)
        db.session.commit()
        
        flash(f'Project "{name}" created successfully.', 'success')
        return redirect(url_for('projects.detail', project_id=project.id))
    
    return render_template('projects/new.html')

@projects_bp.route('/<int:project_id>')
@login_required
def detail(project_id):
    """Show project details."""
    project = Project.query.get_or_404(project_id)
    
    # Check ownership
    if project.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to view this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    tasks = project.tasks.order_by(Task.priority.desc(), Task.created_at.desc()).all()
    worker_bots = project.worker_bots.all()
    
    return render_template('projects/detail.html', 
                         project=project, 
                         tasks=tasks,
                         worker_bots=worker_bots)

@projects_bp.route('/<int:project_id>/edit', methods=['GET', 'POST'])
@login_required
def edit(project_id):
    """Edit project details."""
    project = Project.query.get_or_404(project_id)
    
    # Check ownership
    if project.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to edit this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        project_type = request.form.get('type')
        status = request.form.get('status')
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')
        
        if not name:
            flash('Project name is required.', 'danger')
            return render_template('projects/edit.html', project=project)
        
        project.name = name
        project.description = description
        project.type = project_type
        project.status = status
        
        if start_date_str:
            try:
                project.start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            except ValueError:
                flash('Invalid start date format.', 'warning')
        
        if end_date_str:
            try:
                project.end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            except ValueError:
                flash('Invalid end date format.', 'warning')
        
        db.session.commit()
        
        flash('Project updated successfully.', 'success')
        return redirect(url_for('projects.detail', project_id=project.id))
    
    return render_template('projects/edit.html', project=project)

@projects_bp.route('/<int:project_id>/delete', methods=['POST'])
@login_required
def delete(project_id):
    """Delete a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check ownership
    if project.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to delete this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    project_name = project.name
    db.session.delete(project)
    db.session.commit()
    
    flash(f'Project "{project_name}" deleted successfully.', 'success')
    return redirect(url_for('projects.index'))

@projects_bp.route('/<int:project_id>/spawn-bot', methods=['GET', 'POST'])
@login_required
def spawn_bot(project_id):
    """Spawn a new worker bot for the project."""
    project = Project.query.get_or_404(project_id)
    
    # Check ownership
    if project.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to spawn bots for this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        bot_type = request.form.get('type')
        ai_provider = request.form.get('ai_provider')
        ai_model = request.form.get('ai_model')
        
        if not all([name, bot_type, ai_provider, ai_model]):
            flash('All fields are required.', 'danger')
            return render_template('projects/spawn_bot.html', project=project)
        
        # Define bot capabilities based on type
        capabilities = {
            'architect': ['system_design', 'requirements_analysis', 'task_decomposition'],
            'developer': ['coding', 'refactoring', 'debugging'],
            'tester': ['test_creation', 'bug_finding', 'quality_assurance'],
            'devops': ['deployment', 'infrastructure', 'monitoring']
        }.get(bot_type, [])
        
        bot = WorkerBot(
            name=name,
            type=bot_type,
            ai_provider=ai_provider,
            ai_model=ai_model,
            capabilities=capabilities,
            status='idle',
            project_id=project.id,
            created_by_id=current_user.id
        )
        
        db.session.add(bot)
        db.session.commit()
        
        flash(f'Worker bot "{name}" spawned successfully.', 'success')
        return redirect(url_for('projects.detail', project_id=project.id))
    
    return render_template('projects/spawn_bot.html', project=project)