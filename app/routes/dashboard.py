"""
ProjectPilot - AI-powered project management system
Dashboard routes for system overview and monitoring.
"""

from flask import Blueprint, render_template, jsonify, request, current_app
from flask_login import login_required, current_user
from app.models.project import Project
from app.models.task import Task
from app.models.worker_bot import WorkerBot
from app.services._provider import AIProvider
from app.services.bot_manager_lambda import BotManagerLambda
from datetime import datetime, timedelta
import random

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

@dashboard_bp.route('/')
@login_required
def index():
    """Dashboard home page with system overview."""
    return render_template('dashboard/system_overview.html', **get_dashboard_data())

@dashboard_bp.route('/api/metrics')
@login_required
def api_metrics():
    """API endpoint for dashboard metrics."""
    return jsonify(get_dashboard_data())

def get_dashboard_data():
    """Get all data needed for the dashboard."""
    # Projects statistics
    active_projects_count = Project.query.filter_by(status='active').count()
    max_projects = current_app.config.get('MAX_PROJECTS', 50)
    
    # Worker bots statistics
    worker_bots = WorkerBot.query.all()
    worker_bots_count = len(worker_bots)
    working_bots_count = sum(1 for bot in worker_bots if bot.status == 'working')
    idle_bots_count = sum(1 for bot in worker_bots if bot.status == 'idle')
    error_bots_count = sum(1 for bot in worker_bots if bot.status == 'error')
    
    # AI usage statistics
    # In a real app, you would get this from your AIProvider
    ai_tokens_used = random.randint(50000, 500000)  # Simulated token usage
    ai_token_budget = 1000000  # 1M tokens per day budget
    
    # Task statistics
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tasks_completed_today = Task.query.filter(
        Task.status == 'completed',
        Task.completion_date >= today
    ).count()
    
    completed_task_count = Task.query.filter_by(status='completed').count()
    in_progress_task_count = Task.query.filter_by(status='in_progress').count()
    pending_task_count = Task.query.filter_by(status='pending').count()
    blocked_task_count = Task.query.filter_by(status='blocked').count()
    
    # AI provider status (simulated)
    ai_providers = [
        {"name": "OpenAI", "status": "healthy", "response_time": 320, "uptime": "99.9%"},
        {"name": "Anthropic", "status": "healthy", "response_time": 380, "uptime": "99.7%"},
        {"name": "AWS Bedrock", "status": "healthy", "response_time": 340, "uptime": "99.8%"}
    ]
    
    # System components (simulated)
    system_components = [
        {"name": "Web Server", "status": "online", "icon": "fas fa-server", "uptime": "14d 6h", "response_time": 42},
        {"name": "Database", "status": "online", "icon": "fas fa-database", "uptime": "14d 6h", "response_time": 15},
        {"name": "Redis Cache", "status": "online", "icon": "fas fa-memory", "uptime": "7d 12h", "response_time": 3},
        {"name": "AI Workflow", "status": "online", "icon": "fas fa-brain", "uptime": "14d 6h", "response_time": 328},
        {"name": "Worker Manager", "status": "online", "icon": "fas fa-robot", "uptime": "14d 5h", "response_time": 108},
        {"name": "S3 Storage", "status": "online", "icon": "fas fa-hdd", "uptime": "30d+", "response_time": 87}
    ]
    
    # System logs (simulated)
    system_logs = [
        {"timestamp": "2025-04-03 12:42:01", "level": "INFO", "component": "AIProvider", "message": "Successfully initialized AI providers: OpenAI, Anthropic, AWS Bedrock"},
        {"timestamp": "2025-04-03 12:43:15", "level": "INFO", "component": "WorkerBot", "message": "Initialized 8 worker bots (2 architect, 3 developer, 2 tester, 1 devops)"},
        {"timestamp": "2025-04-03 12:44:32", "level": "WARNING", "component": "AIProvider", "message": "Rate limit approached for OpenAI provider (80% of quota)"},
        {"timestamp": "2025-04-03 12:45:10", "level": "INFO", "component": "TaskManager", "message": "Task #142 assigned to Developer Bot #2"},
        {"timestamp": "2025-04-03 12:46:58", "level": "ERROR", "component": "AIProvider", "message": "Failed to communicate with Anthropic API: Connection timeout"},
        {"timestamp": "2025-04-03 12:48:05", "level": "INFO", "component": "BotManager", "message": "Auto-remediation: Reset circuit breaker for Anthropic provider"},
        {"timestamp": "2025-04-03 12:49:22", "level": "INFO", "component": "TaskManager", "message": "Task #142 completed successfully by Developer Bot #2"},
        {"timestamp": "2025-04-03 12:51:45", "level": "INFO", "component": "ProjectManager", "message": "Project 'E-commerce Platform' progress updated to 68%"}
    ]
    total_logs = 128  # Simulated total log count
    
    # System alerts (simulated)
    system_alerts = [
        {"level": "warning", "title": "Rate Limit Warning", "message": "OpenAI API rate limit at 80%", "component": "AIProvider", "time": "15 min ago", "acknowledged": False},
        {"level": "critical", "title": "API Connection Failed", "message": "Anthropic API connection timeout", "component": "AIProvider", "time": "13 min ago", "acknowledged": True},
        {"level": "info", "title": "Auto-Remediation", "message": "Circuit breaker reset for Anthropic provider", "component": "BotManager", "time": "11 min ago", "acknowledged": True},
        {"level": "warning", "title": "Worker Bot Inactive", "message": "Tester Bot #1 inactive for 2 hours", "component": "WorkerBot", "time": "5 min ago", "acknowledged": False}
    ]
    
    return {
        "active_projects_count": active_projects_count,
        "max_projects": max_projects,
        "worker_bots": worker_bots,
        "worker_bots_count": worker_bots_count,
        "working_bots_count": working_bots_count,
        "idle_bots_count": idle_bots_count,
        "error_bots_count": error_bots_count,
        "ai_tokens_used": ai_tokens_used,
        "ai_token_budget": ai_token_budget,
        "tasks_completed_today": tasks_completed_today,
        "completed_task_count": completed_task_count,
        "in_progress_task_count": in_progress_task_count,
        "pending_task_count": pending_task_count,
        "blocked_task_count": blocked_task_count,
        "ai_providers": ai_providers,
        "system_components": system_components,
        "system_logs": system_logs,
        "total_logs": total_logs,
        "system_alerts": system_alerts
    }