"""
ProjectPilot - AI-powered project management system
Task model for project tasks.
"""

from datetime import datetime
import uuid
from app import db

def generate_uuid():
    """Generate a UUID for tasks."""
    return str(uuid.uuid4())

class Task(db.Model):
    """Task model for project tasks."""
    __tablename__ = 'tasks'
    
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, default=generate_uuid)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    type = db.Column(db.String(64), index=True)  # e.g., "feature", "bug", "documentation", etc.
    status = db.Column(db.String(32), default='pending')  # pending, in_progress, review, completed, blocked
    priority = db.Column(db.Integer, default=2)  # 1=low, 2=medium, 3=high, 4=critical
    weight = db.Column(db.Integer, default=1)  # Relative weight for progress calculation
    due_date = db.Column(db.DateTime)
    start_date = db.Column(db.DateTime)
    completion_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    assigned_to_bot_id = db.Column(db.Integer, db.ForeignKey('worker_bots.id'))
    parent_task_id = db.Column(db.Integer, db.ForeignKey('tasks.id'))
    
    # Relationships
    dependencies = db.relationship('TaskDependency', 
                                  foreign_keys='TaskDependency.task_id',
                                  backref=db.backref('task', lazy='joined'),
                                  lazy='dynamic',
                                  cascade='all, delete-orphan')
    dependents = db.relationship('TaskDependency',
                                foreign_keys='TaskDependency.depends_on_id',
                                backref=db.backref('depends_on', lazy='joined'),
                                lazy='dynamic',
                                cascade='all, delete-orphan')
    subtasks = db.relationship('Task', 
                              backref=db.backref('parent', remote_side=[id]),
                              lazy='dynamic')
    
    def update_status(self, status):
        """Update task status and related data."""
        old_status = self.status
        self.status = status
        
        # Update dates based on status
        if status == 'in_progress' and not self.start_date:
            self.start_date = datetime.utcnow()
        elif status == 'completed' and not self.completion_date:
            self.completion_date = datetime.utcnow()
        
        # Update project progress if status changed
        if old_status != status:
            self.project.update_progress()
    
    def add_dependency(self, dependent_task):
        """Add a dependency to this task."""
        if dependent_task.id != self.id:  # Prevent self-dependency
            dependency = TaskDependency(task_id=self.id, depends_on_id=dependent_task.id)
            db.session.add(dependency)
            return dependency
        return None
    
    def to_dict(self):
        """Convert task to dictionary for API responses."""
        return {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
            'description': self.description,
            'type': self.type,
            'status': self.status,
            'priority': self.priority,
            'weight': self.weight,
            'project_id': self.project_id,
            'assigned_to_bot_id': self.assigned_to_bot_id,
            'parent_task_id': self.parent_task_id,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'completion_date': self.completion_date.isoformat() if self.completion_date else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'dependency_count': self.dependencies.count(),
            'subtask_count': self.subtasks.count()
        }
    
    def __repr__(self):
        return f'<Task {self.name}>'


class TaskDependency(db.Model):
    """Task dependency model to track dependencies between tasks."""
    __tablename__ = 'task_dependencies'
    
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('tasks.id'), nullable=False)
    depends_on_id = db.Column(db.Integer, db.ForeignKey('tasks.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('task_id', 'depends_on_id', name='unique_task_dependency'),
    )
    
    def __repr__(self):
        return f'<TaskDependency {self.task_id} depends on {self.depends_on_id}>'