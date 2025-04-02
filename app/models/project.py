"""
ProjectPilot - AI-powered project management system
Project model for project management.
"""

from datetime import datetime
import uuid
from app import db

def generate_uuid():
    """Generate a UUID for projects."""
    return str(uuid.uuid4())

class Project(db.Model):
    """Project model for project management."""
    __tablename__ = 'projects'
    
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, default=generate_uuid)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    type = db.Column(db.String(64), index=True)  # e.g., "web", "mobile", "api", etc.
    status = db.Column(db.String(32), default='planning')  # planning, active, completed, archived
    progress = db.Column(db.Integer, default=0)  # 0-100 percent
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    tasks = db.relationship('Task', backref='project', lazy='dynamic', cascade='all, delete-orphan')
    worker_bots = db.relationship('WorkerBot', backref='project', lazy='dynamic', cascade='all, delete-orphan')
    
    def update_progress(self):
        """Calculate and update project progress based on task completion."""
        tasks = self.tasks.all()
        if not tasks:
            self.progress = 0
            return
        
        completed_weight = sum(task.weight for task in tasks if task.status == 'completed')
        total_weight = sum(task.weight for task in tasks)
        
        if total_weight > 0:
            self.progress = int((completed_weight / total_weight) * 100)
        else:
            self.progress = 0
        
        db.session.commit()
    
    def to_dict(self):
        """Convert project to dictionary for API responses."""
        return {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
            'description': self.description,
            'type': self.type,
            'status': self.status,
            'progress': self.progress,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'owner_id': self.owner_id,
            'task_count': self.tasks.count(),
            'worker_bot_count': self.worker_bots.count()
        }
    
    def __repr__(self):
        return f'<Project {self.name}>'