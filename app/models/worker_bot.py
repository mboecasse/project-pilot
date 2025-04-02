"""
ProjectPilot - AI-powered project management system
WorkerBot model for AI workers.
"""

from datetime import datetime
import uuid
from app import db

def generate_uuid():
    """Generate a UUID for worker bots."""
    return str(uuid.uuid4())

class WorkerBot(db.Model):
    """WorkerBot model for AI workers."""
    __tablename__ = 'worker_bots'
    
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, default=generate_uuid)
    name = db.Column(db.String(128), nullable=False)
    type = db.Column(db.String(64), index=True)  # e.g., "architect", "developer", "tester", etc.
    status = db.Column(db.String(32), default='inactive')  # inactive, idle, working, paused, error
    capabilities = db.Column(db.JSON)  # JSON array of capabilities
    ai_provider = db.Column(db.String(32))  # e.g., "openai", "anthropic", "bedrock"
    ai_model = db.Column(db.String(64))  # e.g., "gpt-4", "claude-3-opus", etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_active = db.Column(db.DateTime)
    
    # Foreign keys
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'), nullable=False)
    created_by_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    assigned_tasks = db.relationship('Task', backref='assigned_bot', lazy='dynamic')
    
    def update_status(self, status):
        """Update bot status and last active timestamp."""
        self.status = status
        if status in ['working', 'idle']:
            self.last_active = datetime.utcnow()
        db.session.commit()
    
    def assign_task(self, task):
        """Assign a task to this bot."""
        if self.status in ['idle', 'inactive']:
            task.assigned_to_bot_id = self.id
            self.update_status('working')
            return True
        return False
    
    def complete_current_task(self):
        """Mark the current task as completed and update bot status."""
        current_task = self.assigned_tasks.filter_by(status='in_progress').first()
        if current_task:
            current_task.update_status('completed')
            self.update_status('idle')
            return current_task
        return None
    
    def to_dict(self):
        """Convert worker bot to dictionary for API responses."""
        return {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
            'type': self.type,
            'status': self.status,
            'capabilities': self.capabilities,
            'ai_provider': self.ai_provider,
            'ai_model': self.ai_model,
            'project_id': self.project_id,
            'created_by_id': self.created_by_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_active': self.last_active.isoformat() if self.last_active else None,
            'assigned_task_count': self.assigned_tasks.count()
        }
    
    def __repr__(self):
        return f'<WorkerBot {self.name}>'