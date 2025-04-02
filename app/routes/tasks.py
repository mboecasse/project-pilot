"""
ProjectPilot - AI-powered project management system
Task management routes.
"""

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from app import db
from app.models.project import Project
from app.models.task import Task, TaskDependency
from app.models.worker_bot import WorkerBot
from datetime import datetime

tasks_bp = Blueprint('tasks', __name__, url_prefix='/tasks')

@tasks_bp.route('/project/<int:project_id>/new', methods=['GET', 'POST'])
@login_required
def new_task(project_id):
    """Create a new task for a project."""
    project = Project.query.get_or_404(project_id)
    
    # Check project ownership
    if project.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to add tasks to this project.', 'danger')
        return redirect(url_for('projects.index'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        task_type = request.form.get('type')
        priority = request.form.get('priority', type=int, default=2)
        weight = request.form.get('weight', type=int, default=1)
        due_date_str = request.form.get('due_date')
        parent_task_id = request.form.get('parent_task_id', type=int)
        dependencies = request.form.getlist('dependencies')
        
        if not name:
            flash('Task name is required.', 'danger')
            return render_template('tasks/new.html', project=project)
        
        task = Task(
            name=name,
            description=description,
            type=task_type,
            status='pending',
            priority=priority,
            weight=weight,
            project_id=project_id,
            parent_task_id=parent_task_id if parent_task_id else None
        )
        
        if due_date_str:
            try:
                task.due_date = datetime.strptime(due_date_str, '%Y-%m-%d')
            except ValueError:
                flash('Invalid due date format.', 'warning')
        
        db.session.add(task)
        db.session.flush()  # Get the task ID for dependencies
        
        # Add dependencies
        if dependencies:
            for dep_id in dependencies:
                dependency = TaskDependency(task_id=task.id, depends_on_id=int(dep_id))
                db.session.add(dependency)
        
        db.session.commit()
        project.update_progress()
        
        flash(f'Task "{name}" created successfully.', 'success')
        return redirect(url_for('projects.detail', project_id=project_id))
    
    # Get existing tasks for parent/dependency selection
    existing_tasks = project.tasks.all()
    
    return render_template('tasks/new.html', 
                         project=project, 
                         existing_tasks=existing_tasks)

@tasks_bp.route('/<int:task_id>')
@login_required
def detail(task_id):
    """Show task details."""
    task = Task.query.get_or_404(task_id)
    project = task.project
    
    # Check project ownership
    if project.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to view this task.', 'danger')
        return redirect(url_for('projects.index'))
    
    dependencies = [dep.depends_on for dep in task.dependencies]
    dependents = [dep.task for dep in task.dependents]
    
    return render_template('tasks/detail.html', 
                         task=task,
                         project=project,
                         dependencies=dependencies,
                         dependents=dependents)

@tasks_bp.route('/<int:task_id>/edit', methods=['GET', 'POST'])
@login_required
def edit(task_id):
    """Edit task details."""
    task = Task.query.get_or_404(task_id)
    project = task.project
    
    # Check project ownership
    if project.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to edit this task.', 'danger')
        return redirect(url_for('projects.index'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        task_type = request.form.get('type')
        status = request.form.get('status')
        priority = request.form.get('priority', type=int, default=2)
        weight = request.form.get('weight', type=int, default=1)
        due_date_str = request.form.get('due_date')
        parent_task_id = request.form.get('parent_task_id', type=int)
        
        if not name:
            flash('Task name is required.', 'danger')
            return render_template('tasks/edit.html', task=task, project=project)
        
        task.name = name
        task.description = description
        task.type = task_type
        task.priority = priority
        task.weight = weight
        task.parent_task_id = parent_task_id if parent_task_id else None
        
        # Update status with the helper method to track dates
        if status != task.status:
            task.update_status(status)
        
        if due_date_str:
            try:
                task.due_date = datetime.strptime(due_date_str, '%Y-%m-%d')
            except ValueError:
                flash('Invalid due date format.', 'warning')
        
        db.session.commit()
        project.update_progress()
        
        flash('Task updated successfully.', 'success')
        return redirect(url_for('tasks.detail', task_id=task.id))
    
    # Get existing tasks for parent/dependency selection
    existing_tasks = project.tasks.filter(Task.id != task_id).all()
    
    return render_template('tasks/edit.html', 
                         task=task,
                         project=project,
                         existing_tasks=existing_tasks)

@tasks_bp.route('/<int:task_id>/delete', methods=['POST'])
@login_required
def delete(task_id):
    """Delete a task."""
    task = Task.query.get_or_404(task_id)
    project = task.project
    
    # Check project ownership
    if project.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to delete this task.', 'danger')
        return redirect(url_for('projects.index'))
    
    task_name = task.name
    project_id = task.project_id
    
    db.session.delete(task)
    db.session.commit()
    project.update_progress()
    
    flash(f'Task "{task_name}" deleted successfully.', 'success')
    return redirect(url_for('projects.detail', project_id=project_id))

@tasks_bp.route('/<int:task_id>/assign', methods=['POST'])
@login_required
def assign_task(task_id):
    """Assign a task to a worker bot."""
    task = Task.query.get_or_404(task_id)
    project = task.project
    
    # Check project ownership
    if project.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to assign this task.', 'danger')
        return redirect(url_for('projects.index'))
    
    bot_id = request.form.get('bot_id', type=int)
    if not bot_id:
        flash('Worker bot selection is required.', 'danger')
        return redirect(url_for('tasks.detail', task_id=task_id))
    
    bot = WorkerBot.query.get_or_404(bot_id)
    
    # Check if bot is available
    if bot.status not in ['idle', 'inactive']:
        flash(f'Worker bot "{bot.name}" is not available for assignment.', 'warning')
        return redirect(url_for('tasks.detail', task_id=task_id))
    
    # Assign task to bot
    if bot.assign_task(task):
        task.update_status('in_progress')
        flash(f'Task assigned to worker bot "{bot.name}" successfully.', 'success')
    else:
        flash(f'Failed to assign task to worker bot "{bot.name}".', 'danger')
    
    return redirect(url_for('tasks.detail', task_id=task_id))

@tasks_bp.route('/<int:task_id>/update-status', methods=['POST'])
@login_required
def update_status(task_id):
    """Update task status."""
    task = Task.query.get_or_404(task_id)
    project = task.project
    
    # Check project ownership
    if project.owner_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to update this task.', 'danger')
        return redirect(url_for('projects.index'))
    
    status = request.form.get('status')
    if not status:
        flash('Status is required.', 'danger')
        return redirect(url_for('tasks.detail', task_id=task_id))
    
    task.update_status(status)
    db.session.commit()
    
    flash(f'Task status updated to "{status}" successfully.', 'success')
    return redirect(url_for('tasks.detail', task_id=task_id))