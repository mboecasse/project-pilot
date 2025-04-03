"""
ProjectPilot - AI-powered project management system
Project Manager Bot for autonomous project orchestration.
"""

import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, Set
import threading
import queue
from enum import Enum

from app import db
from app.models.project import Project
from app.models.task import Task, TaskDependency
from app.models.worker_bot import WorkerBot
from app.services.task_execution_pipeline import task_pipeline, TaskStatus
from app.services.three_ai_workflow import ThreeAIWorkflow
from app.services.performance_analytics import PerformanceAnalytics
from app.services.ai_provider import AIProvider

logger = logging.getLogger(__name__)

class ProjectManagerBot:
    """
    Central orchestration system that autonomously manages software projects
    by coordinating worker bots, managing tasks, and optimizing resources.
    """
    
    def __init__(self, 
                 three_ai_workflow: Optional[ThreeAIWorkflow] = None,
                 task_pipeline = None,
                 analytics: Optional[PerformanceAnalytics] = None,
                 polling_interval: int = 60):
        """
        Initialize the Project Manager Bot.
        
        Args:
            three_ai_workflow: AI workflow for high-level decisions
            task_pipeline: Task execution pipeline 
            analytics: Performance analytics service
            polling_interval: Seconds between project management cycles
        """
        self.three_ai_workflow = three_ai_workflow or ThreeAIWorkflow()
        self.task_pipeline = task_pipeline or task_pipeline
        self.analytics = analytics or PerformanceAnalytics()
        self.polling_interval = polling_interval
        
        # Project management state
        self.managed_projects = set()
        self.project_locks = {}
        self.project_states = {}
        
        # Bot management state
        self.managed_bots = {}
        
        # Operation state
        self.running = False
        self.manager_thread = None
        self.exit_event = threading.Event()
        
        # Statistics
        self.stats = {
            'cycles_completed': 0,
            'tasks_created': 0,
            'tasks_scheduled': 0,
            'bots_spawned': 0,
            'projects_analyzed': 0,
            'decisions_made': 0
        }
        
        # Event log
        self.event_log = []
        self.max_event_log_size = 1000
        
        logger.info("Project Manager Bot initialized")
    
    def start(self) -> None:
        """Start the Project Manager Bot."""
        if self.running:
            logger.warning("Project Manager Bot already running")
            return
        
        self.running = True
        self.exit_event.clear()
        self.manager_thread = threading.Thread(target=self._manager_worker)
        self.manager_thread.daemon = True
        self.manager_thread.start()
        
        # Ensure the task pipeline is running
        if hasattr(self.task_pipeline, 'running') and not self.task_pipeline.running:
            self.task_pipeline.start()
        
        logger.info("Project Manager Bot started")
        self._log_event("system", "Project Manager Bot started")
    
    def stop(self) -> None:
        """Stop the Project Manager Bot."""
        if not self.running:
            logger.warning("Project Manager Bot not running")
            return
        
        self.running = False
        self.exit_event.set()
        
        if self.manager_thread:
            self.manager_thread.join(timeout=30)
            if self.manager_thread.is_alive():
                logger.warning("Project Manager Bot thread did not terminate cleanly")
            
        logger.info("Project Manager Bot stopped")
        self._log_event("system", "Project Manager Bot stopped")
    
    def manage_project(self, project_id: int) -> bool:
        """
        Add a project to be managed by the Project Manager Bot.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if project was added successfully
        """
        try:
            # Check if project exists
            project = Project.query.get(project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return False
            
            # Check if project is already being managed
            if project_id in self.managed_projects:
                logger.warning(f"Project {project_id} is already being managed")
                return False
            
            # Create lock for this project
            self.project_locks[project_id] = threading.RLock()
            
            # Add to managed projects
            self.managed_projects.add(project_id)
            
            # Initialize project state
            self.project_states[project_id] = {
                "last_analyzed": None,
                "last_task_created": None,
                "active_tasks": set(),
                "blocked_tasks": set(),
                "worker_bots": set()
            }
            
            logger.info(f"Project {project_id} added to managed projects")
            self._log_event("project", f"Started managing project: {project.name} (ID: {project_id})")
            
            # Perform initial analysis
            self._analyze_project(project_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding project {project_id} to managed projects: {str(e)}")
            return False
    
    def unmanage_project(self, project_id: int) -> bool:
        """
        Remove a project from being managed by the Project Manager Bot.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if project was removed successfully
        """
        try:
            # Check if project is being managed
            if project_id not in self.managed_projects:
                logger.warning(f"Project {project_id} is not being managed")
                return False
            
            # Remove from managed projects
            self.managed_projects.remove(project_id)
            
            # Clean up project state
            if project_id in self.project_states:
                del self.project_states[project_id]
            
            # Clean up project lock
            if project_id in self.project_locks:
                del self.project_locks[project_id]
            
            logger.info(f"Project {project_id} removed from managed projects")
            
            # Get project name for logging
            project = Project.query.get(project_id)
            project_name = project.name if project else f"Unknown ({project_id})"
            self._log_event("project", f"Stopped managing project: {project_name} (ID: {project_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing project {project_id} from managed projects: {str(e)}")
            return False
    
    def spawn_worker_bot(self, 
                        project_id: int, 
                        bot_type: str, 
                        ai_provider: str = 'auto', 
                        ai_model: Optional[str] = None) -> Optional[int]:
        """
        Spawn a new worker bot for a project.
        
        Args:
            project_id: Project ID
            bot_type: Type of worker bot to spawn
            ai_provider: AI provider to use
            ai_model: Specific AI model to use
            
        Returns:
            Worker bot ID or None if spawn failed
        """
        try:
            # Check if project exists and is being managed
            project = Project.query.get(project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return None
            
            if project_id not in self.managed_projects:
                logger.warning(f"Project {project_id} is not being managed")
                return None
            
            # Determine AI provider and model
            if ai_provider == 'auto':
                # Select based on bot type and availability
                ai_provider_util = AIProvider()
                providers = [p for p, available in ai_provider_util.providers.items() if available]
                
                if not providers:
                    logger.error("No AI providers available")
                    return None
                    
                # Select provider based on bot type
                provider_preferences = {
                    'architect': ['anthropic', 'openai', 'bedrock'],
                    'developer': ['openai', 'anthropic', 'bedrock'],
                    'tester': ['openai', 'anthropic', 'bedrock'],
                    'devops': ['anthropic', 'openai', 'bedrock']
                }
                
                preferences = provider_preferences.get(bot_type, ['openai', 'anthropic', 'bedrock'])
                
                for pref in preferences:
                    if pref in providers:
                        ai_provider = pref
                        break
                else:
                    # If no preferred provider is available, use the first available
                    ai_provider = providers[0]
            
            # Determine AI model if not specified
            if not ai_model:
                # Default models for each provider
                default_models = {
                    'openai': {
                        'architect': 'gpt-4',
                        'developer': 'gpt-4',
                        'tester': 'gpt-3.5-turbo',
                        'devops': 'gpt-4'
                    },
                    'anthropic': {
                        'architect': 'claude-3-opus-20240229',
                        'developer': 'claude-3-opus-20240229',
                        'tester': 'claude-3-sonnet-20240229',
                        'devops': 'claude-3-opus-20240229'
                    },
                    'bedrock': {
                        'architect': 'anthropic.claude-3-opus-20240229-v1:0',
                        'developer': 'anthropic.claude-3-opus-20240229-v1:0',
                        'tester': 'anthropic.claude-3-sonnet-20240229-v1:0',
                        'devops': 'anthropic.claude-3-opus-20240229-v1:0'
                    }
                }
                
                ai_model = default_models.get(ai_provider, {}).get(bot_type)
                
                if not ai_model:
                    # Fallback defaults
                    default_fallbacks = {
                        'openai': 'gpt-4',
                        'anthropic': 'claude-3-opus-20240229',
                        'bedrock': 'anthropic.claude-3-opus-20240229-v1:0'
                    }
                    ai_model = default_fallbacks.get(ai_provider)
            
            # Generate a name for the bot
            bot_names = {
                'architect': ["ArchitectBot", "DesignMaster", "BlueprintAI", "SchematicGenius", "ArchAngel"],
                'developer': ["DevBot", "CodeForge", "SyntaxWizard", "BuildMaster", "ByteCrafter"],
                'tester': ["TestBot", "QualityGuard", "BugHunter", "AssuranceAI", "ValidatorPro"],
                'devops': ["DevOpsBot", "DeployMaster", "InfraGenius", "PipelineAI", "CloudForge"]
            }
            
            # Count existing bots of this type for this project
            existing_count = WorkerBot.query.filter_by(
                project_id=project_id,
                type=bot_type
            ).count()
            
            # Get name options
            name_options = bot_names.get(bot_type, ["WorkerBot"])
            
            # Select name, or append number if we've used all options
            if existing_count < len(name_options):
                bot_name = name_options[existing_count]
            else:
                bot_name = f"{name_options[0]}-{existing_count + 1}"
            
            # Define bot capabilities based on type
            capabilities = {
                'architect': ['system_design', 'requirements_analysis', 'task_decomposition'],
                'developer': ['coding', 'refactoring', 'debugging'],
                'tester': ['test_creation', 'bug_finding', 'quality_assurance'],
                'devops': ['deployment', 'infrastructure', 'monitoring']
            }.get(bot_type, [])
            
            # Create the worker bot
            bot = WorkerBot(
                name=bot_name,
                type=bot_type,
                ai_provider=ai_provider,
                ai_model=ai_model,
                capabilities=capabilities,
                status='idle',
                project_id=project_id,
                created_by_id=None  # System-created
            )
            
            db.session.add(bot)
            db.session.commit()
            
            # Update statistics
            self.stats['bots_spawned'] += 1
            
            # Update project state
            with self.project_locks[project_id]:
                self.project_states[project_id]["worker_bots"].add(bot.id)
            
            logger.info(f"Worker bot {bot.id} ({bot_name}) spawned for project {project_id}")
            self._log_event("bot", f"Spawned {bot_type} bot: {bot_name} (ID: {bot.id}) for project {project.name}")
            
            return bot.id
            
        except Exception as e:
            logger.error(f"Error spawning worker bot for project {project_id}: {str(e)}")
            return None
    
    def analyze_requirements(self, project_id: int, requirements: str) -> Dict[str, Any]:
        """
        Analyze project requirements and generate tasks.
        
        Args:
            project_id: Project ID
            requirements: Project requirements text
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Check if project exists and is being managed
            project = Project.query.get(project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return {"error": f"Project {project_id} not found"}
            
            if project_id not in self.managed_projects:
                logger.warning(f"Project {project_id} is not being managed")
                # We'll allow analysis even if project is not being managed
            
            # Analyze requirements with ThreeAIWorkflow
            analysis = self.three_ai_workflow.analyze_project_requirements(requirements)
            
            if "error" in analysis:
                logger.error(f"Error analyzing requirements for project {project_id}: {analysis['error']}")
                return analysis
            
            # Check if we should automatically create tasks
            auto_create_tasks = project_id in self.managed_projects
            
            if auto_create_tasks:
                # Create tasks from the analysis
                created_tasks = self._create_tasks_from_analysis(project_id, analysis)
                analysis["created_tasks"] = created_tasks
                
                # Update statistics
                self.stats['tasks_created'] += len(created_tasks)
                
                # Update project state
                with self.project_locks[project_id]:
                    self.project_states[project_id]["last_task_created"] = datetime.utcnow()
                
                logger.info(f"Created {len(created_tasks)} tasks for project {project_id} from requirements analysis")
                self._log_event("project", f"Created {len(created_tasks)} tasks for project {project.name} from requirements analysis")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing requirements for project {project_id}: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def get_project_status(self, project_id: int) -> Dict[str, Any]:
        """
        Get detailed status for a managed project.
        
        Args:
            project_id: Project ID
            
        Returns:
            Dictionary with project status
        """
        try:
            # Check if project exists
            project = Project.query.get(project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return {"error": f"Project {project_id} not found"}
            
            # Get project analytics
            analytics = self.analytics.analyze_project(project_id)
            
            # Get task execution status
            tasks = Task.query.filter_by(project_id=project_id).all()
            task_statuses = {}
            
            for task in tasks:
                # Get execution status if being executed
                execution_status = None
                if hasattr(self.task_pipeline, 'get_execution_status'):
                    execution_status = self.task_pipeline.get_execution_status(task.id)
                
                task_statuses[task.id] = {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "priority": task.priority,
                    "assigned_to_bot_id": task.assigned_to_bot_id,
                    "execution_status": execution_status
                }
            
            # Get worker bot status
            bots = WorkerBot.query.filter_by(project_id=project_id).all()
            bot_statuses = {}
            
            for bot in bots:
                bot_statuses[bot.id] = {
                    "id": bot.id,
                    "name": bot.name,
                    "type": bot.type,
                    "status": bot.status,
                    "ai_provider": bot.ai_provider,
                    "ai_model": bot.ai_model,
                    "capabilities": bot.capabilities,
                    "last_active": bot.last_active.isoformat() if bot.last_active else None
                }
            
            # Get management status
            is_managed = project_id in self.managed_projects
            management_status = None
            
            if is_managed and project_id in self.project_states:
                management_status = {
                    "last_analyzed": self.project_states[project_id]["last_analyzed"].isoformat() 
                                   if self.project_states[project_id]["last_analyzed"] else None,
                    "last_task_created": self.project_states[project_id]["last_task_created"].isoformat()
                                       if self.project_states[project_id]["last_task_created"] else None,
                    "active_tasks": list(self.project_states[project_id]["active_tasks"]),
                    "blocked_tasks": list(self.project_states[project_id]["blocked_tasks"]),
                    "worker_bots": list(self.project_states[project_id]["worker_bots"])
                }
            
            # Assemble status response
            status = {
                "project": {
                    "id": project.id,
                    "name": project.name,
                    "type": project.type,
                    "status": project.status,
                    "progress": project.progress,
                    "start_date": project.start_date.isoformat() if project.start_date else None,
                    "end_date": project.end_date.isoformat() if project.end_date else None,
                    "created_at": project.created_at.isoformat(),
                    "updated_at": project.updated_at.isoformat()
                },
                "analytics": analytics,
                "tasks": task_statuses,
                "bots": bot_statuses,
                "is_managed": is_managed,
                "management_status": management_status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting status for project {project_id}: {str(e)}")
            return {"error": f"Status retrieval failed: {str(e)}"}
    
    def get_manager_status(self) -> Dict[str, Any]:
        """
        Get status of the Project Manager Bot.
        
        Returns:
            Dictionary with manager status
        """
        status = {
            "running": self.running,
            "managed_projects": list(self.managed_projects),
            "managed_projects_count": len(self.managed_projects),
            "statistics": self.stats,
            "recent_events": self.event_log[-10:] if self.event_log else [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return status
    
    def get_events(self, limit: int = 100, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent events from the event log.
        
        Args:
            limit: Maximum number of events to return
            event_type: Optional filter for event type
            
        Returns:
            List of event dictionaries
        """
        if event_type:
            filtered_events = [e for e in self.event_log if e["type"] == event_type]
            return filtered_events[-limit:] if filtered_events else []
        else:
            return self.event_log[-limit:] if self.event_log else []
    
    def _manager_worker(self) -> None:
        """Worker thread for project management."""
        logger.info("Project Manager Bot worker thread started")
        
        while self.running and not self.exit_event.is_set():
            try:
                # Process each managed project
                for project_id in list(self.managed_projects):
                    try:
                        self._process_project(project_id)
                    except Exception as e:
                        logger.error(f"Error processing project {project_id}: {str(e)}")
                
                # Update statistics
                self.stats['cycles_completed'] += 1
                
                # Sleep until next cycle
                time.sleep(self.polling_interval)
                
            except Exception as e:
                logger.error(f"Error in manager worker: {str(e)}")
                time.sleep(self.polling_interval)
        
        logger.info("Project Manager Bot worker thread stopped")
    
    def _process_project(self, project_id: int) -> None:
        """
        Process a managed project.
        
        Args:
            project_id: Project ID
        """
        # Skip if project has been removed
        if project_id not in self.managed_projects:
            return
            
        # Get lock for this project
        if project_id not in self.project_locks:
            self.project_locks[project_id] = threading.RLock()
            
        with self.project_locks[project_id]:
            # Check if project exists
            project = Project.query.get(project_id)
            if not project:
                logger.warning(f"Project {project_id} not found, removing from managed projects")
                self.managed_projects.remove(project_id)
                if project_id in self.project_states:
                    del self.project_states[project_id]
                return
            
            # Skip completed projects
            if project.status == 'completed':
                return
            
            # Analysis should run periodically
            should_analyze = True
            if project_id in self.project_states:
                last_analyzed = self.project_states[project_id]["last_analyzed"]
                if last_analyzed:
                    # Analyze at most once per hour
                    time_since_analysis = (datetime.utcnow() - last_analyzed).total_seconds()
                    should_analyze = time_since_analysis >= 3600
            
            if should_analyze:
                self._analyze_project(project_id)
            
            # Process tasks
            self._process_project_tasks(project_id)
            
            # Check worker bot needs
            self._check_worker_bot_needs(project_id)
            
            # Update project progress
            project.update_progress()
    
    def _analyze_project(self, project_id: int) -> None:
        """
        Analyze a project's status and needs.
        
        Args:
            project_id: Project ID
        """
        # Skip if project has been removed
        if project_id not in self.managed_projects:
            return
        
        # Get project
        project = Project.query.get(project_id)
        if not project:
            return
        
        # Get analytics
        analytics = self.analytics.analyze_project(project_id)
        
        # Update project state
        if project_id in self.project_states:
            self.project_states[project_id]["last_analyzed"] = datetime.utcnow()
        
        # Update statistics
        self.stats['projects_analyzed'] += 1
        
        # Make decisions based on analytics
        self._make_project_decisions(project_id, analytics)
        
        logger.info(f"Analyzed project {project_id}: {project.name}")
    
    def _make_project_decisions(self, project_id: int, analytics: Dict[str, Any]) -> None:
        """
        Make decisions based on project analytics.
        
        Args:
            project_id: Project ID
            analytics: Project analytics data
        """
        # Skip if project has been removed
        if project_id not in self.managed_projects:
            return
        
        # Get project
        project = Project.query.get(project_id)
        if not project:
            return
        
        decisions_made = 0
        
        # Check overall health
        health = analytics.get("health", {})
        health_score = health.get("score", 0)
        health_rating = health.get("rating", "unknown")
        
        # Decisions based on health
        if health_rating in ["poor", "critical"]:
            # Project is in trouble, add a note
            self._log_event(
                "decision", 
                f"Project {project.name} health is {health_rating.upper()} ({health_score:.1f}/10)."
            )
            decisions_made += 1
        
        # Check task distribution
        task_stats = analytics.get("task_stats", {})
        blocked_tasks = task_stats.get("status_distribution", {}).get("blocked", 0)
        
        if blocked_tasks > 0:
            # There are blocked tasks, try to address them
            self._log_event(
                "decision",
                f"Project {project.name} has {blocked_tasks} blocked tasks. Will attempt to unblock."
            )
            self._prioritize_unblocking_tasks(project_id)
            decisions_made += 1
        
        # Check bot utilization
        bot_stats = analytics.get("bot_stats", {})
        bot_metrics = bot_stats.get("bot_metrics", [])
        
        # Check for underutilized bots
        idle_bots = [bot for bot in bot_metrics if bot.get("tasks_completed", 0) < 2]
        if idle_bots:
            self._log_event(
                "decision",
                f"Project {project.name} has {len(idle_bots)} underutilized bots. Will assign tasks."
            )
            decisions_made += 1
        
        # Check for overburdened bots
        busy_bots = [bot for bot in bot_metrics if bot.get("tasks_assigned", 0) > 5]
        if busy_bots and len(busy_bots) / len(bot_metrics) > 0.5:
            # More than half the bots are overburdened, consider spawning more
            self._log_event(
                "decision",
                f"Project {project.name} has overburdened bots. Will spawn additional bots."
            )
            self._spawn_additional_bots(project_id, bot_metrics)
            decisions_made += 1
        
        # Check velocity
        velocity_metrics = analytics.get("velocity_metrics", {})
        velocity = velocity_metrics.get("velocity", 0)
        
        if velocity < 1:
            # Low velocity, consider adding more resources
            self._log_event(
                "decision",
                f"Project {project.name} has low velocity ({velocity:.1f} tasks/week). Will optimize resources."
            )
            decisions_made += 1
        
        # Update statistics
        self.stats['decisions_made'] += decisions_made
    
    def _prioritize_unblocking_tasks(self, project_id: int) -> None:
        """
        Prioritize unblocking tasks in a project.
        
        Args:
            project_id: Project ID
        """
        # Get blocked tasks
        blocked_tasks = Task.query.filter_by(
            project_id=project_id,
            status='blocked'
        ).all()
        
        if not blocked_tasks:
            return
        
        for task in blocked_tasks:
            # Check dependencies
            dependencies = task.dependencies.all()
            dependent_tasks = [Task.query.get(dep.depends_on_id) for dep in dependencies]
            
            # Find incomplete dependencies
            incomplete_deps = [t for t in dependent_tasks if t and t.status != 'completed']
            
            if incomplete_deps:
                # Prioritize the dependencies
                for dep in incomplete_deps:
                    # Increase priority to highest level
                    dep.priority = 4
                    
                    # If dependency is pending, schedule it
                    if dep.status == 'pending':
                        self.task_pipeline.add_task(dep.id, priority=1)
                        
                        # Update project state
                        if project_id in self.project_states:
                            self.project_states[project_id]["active_tasks"].add(dep.id)
                        
                        self._log_event(
                            "task",
                            f"Prioritized dependency task {dep.name} (ID: {dep.id}) to unblock task {task.name}"
                        )
            else:
                # No dependencies or all are complete, try to unblock
                task.update_status('pending')
                
                # Schedule the task
                self.task_pipeline.add_task(task.id, priority=1)
                
                # Update project state
                if project_id in self.project_states:
                    self.project_states[project_id]["active_tasks"].add(task.id)
                    self.project_states[project_id]["blocked_tasks"].discard(task.id)
                
                self._log_event(
                    "task",
                    f"Unblocked task {task.name} (ID: {task.id})"
                )
        
        # Commit changes
        db.session.commit()
    
    def _spawn_additional_bots(self, project_id: int, bot_metrics: List[Dict[str, Any]]) -> None:
        """
        Spawn additional bots based on workload.
        
        Args:
            project_id: Project ID
            bot_metrics: Bot metrics from analytics
        """
        # Determine which bot types are needed
        bot_types = {}
        for bot in bot_metrics:
            bot_type = bot.get("type", "unknown")
            if bot_type not in bot_types:
                bot_types[bot_type] = {"count": 0, "tasks_assigned": 0, "tasks_completed": 0}
            
            bot_types[bot_type]["count"] += 1
            bot_types[bot_type]["tasks_assigned"] += bot.get("tasks_assigned", 0)
            bot_types[bot_type]["tasks_completed"] += bot.get("tasks_completed", 0)
        
        # Check workload for each bot type
        for bot_type, metrics in bot_types.items():
            avg_workload = metrics["tasks_assigned"] / metrics["count"] if metrics["count"] > 0 else 0
            
            if avg_workload > 4:
                # High workload, spawn an additional bot of this type
                new_bot_id = self.spawn_worker_bot(project_id, bot_type)
                
                if new_bot_id:
                    self._log_event(
                        "bot",
                        f"Spawned additional {bot_type} bot (ID: {new_bot_id}) due to high workload"
                    )
    
    def _process_project_tasks(self, project_id: int) -> None:
        """
        Process tasks for a project, scheduling as appropriate.
        
        Args:
            project_id: Project ID
        """
        # Skip if project has been removed
        if project_id not in self.managed_projects:
            return
        
        # Get project
        project = Project.query.get(project_id)
        if not project:
            return
        
        # Get pending tasks
        pending_tasks = Task.query.filter_by(
            project_id=project_id,
            status='pending'
        ).all()
        
        # Sort tasks by priority (higher number = higher priority)
        pending_tasks.sort(key=lambda x: x.priority, reverse=True)
        
        # Get current task pipeline status
        pipeline_status = self.task_pipeline.get_pipeline_status() if hasattr(self.task_pipeline, 'get_pipeline_status') else None
        current_queue_size = pipeline_status.get('queue_size', 0) if pipeline_status else 0
        
        # Determine how many tasks to schedule
        tasks_to_schedule = max(0, 10 - current_queue_size)
        
        # Schedule highest priority tasks
        scheduled_count = 0
        for task in pending_tasks[:tasks_to_schedule]:
            # Check if dependencies are satisfied
            dependencies_met = True
            for dep in task.dependencies.all():
                dependent_task = Task.query.get(dep.depends_on_id)
                if not dependent_task or dependent_task.status != 'completed':
                    dependencies_met = False
                    break
            
            if dependencies_met:
                # Schedule this task
                success = self.task_pipeline.add_task(task.id, 5 - task.priority)  # Convert priority
                
                if success:
                    scheduled_count += 1
                    
                    # Update project state
                    if project_id in self.project_states:
                        self.project_states[project_id]["active_tasks"].add(task.id)
                    
                    self._log_event(
                        "task",
                        f"Scheduled task {task.name} (ID: {task.id}, Priority: {task.priority})"
                    )
        
        # Update statistics
        if scheduled_count > 0:
            self.stats['tasks_scheduled'] += scheduled_count
            logger.info(f"Scheduled {scheduled_count} tasks for project {project_id}")
    
    def _check_worker_bot_needs(self, project_id: int) -> None:
        """
        Check if a project needs more worker bots.
        
        Args:
            project_id: Project ID
        """
        # Skip if project has been removed
        if project_id not in self.managed_projects:
            return
        
        # Get project
        project = Project.query.get(project_id)
        if not project:
            return
        
        # Get existing bots
        bots = WorkerBot.query.filter_by(project_id=project_id).all()
        
        # Count bots by type
        bot_counts = {}
        for bot in bots:
            if bot.type not in bot_counts:
                bot_counts[bot.type] = 0
            bot_counts[bot.type] += 1
        
        # Check if we need more bots
        required_bots = {
            'architect': 1,
            'developer': 2,
            'tester': 1,
            'devops': 1
        }
        
        for bot_type, required_count in required_bots.items():
            current_count = bot_counts.get(bot_type, 0)
            
            if current_count < required_count:
                # Spawn more bots of this type
                for _ in range(required_count - current_count):
                    new_bot_id = self.spawn_worker_bot(project_id, bot_type)
                    
                    if new_bot_id:
                        self._log_event(
                            "bot",
                            f"Spawned required {bot_type} bot (ID: {new_bot_id})"
                        )
    
    def _create_tasks_from_analysis(self, project_id: int, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create tasks from a requirements analysis.
        
        Args:
            project_id: Project ID
            analysis: Requirements analysis result
            
        Returns:
            List of created task info
        """
        created_tasks = []
        
        # Get tasks from analysis
        tasks_to_create = analysis.get("tasks", [])
        
        # Track dependencies for later updates
        dependency_map = {}
        
        # First pass: Create all tasks
        for task_data in tasks_to_create:
            # Create task
            task = Task(
                name=task_data.get("name", "Unnamed Task"),
                description=task_data.get("description", ""),
                type=task_data.get("type", "feature"),
                status='pending',
                priority=task_data.get("priority", 2),
                weight=task_data.get("complexity", 1),
                project_id=project_id
            )
            
            db.session.add(task)
            db.session.flush()  # Get ID for task
            
            # Store in dependency map for second pass
            task_name = task_data.get("name")
            if task_name:
                dependency_map[task_name] = task.id
            
            # Add to created tasks list
            created_tasks.append({
                "id": task.id,
                "name": task.name,
                "type": task.type,
                "priority": task.priority,
                "weight": task.weight
            })
        
        # Second pass: Create dependencies
        for task_data in tasks_to_create:
            task_name = task_data.get("name")
            if not task_name or task_name not in dependency_map:
                continue
            
            task_id = dependency_map[task_name]
            dependencies = task_data.get("dependencies", [])
            
            for dep_name in dependencies:
                if dep_name in dependency_map:
                    dep_id = dependency_map[dep_name]
                    
                    # Create dependency
                    dependency = TaskDependency(
                        task_id=task_id,
                        depends_on_id=dep_id
                    )
                    
                    db.session.add(dependency)
        
        # Commit all changes
        db.session.commit()
        
        return created_tasks
    
    def _log_event(self, event_type: str, message: str) -> None:
        """
        Log an event to the event log.
        
        Args:
            event_type: Type of event
            message: Event message
        """
        event = {
            "type": event_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.event_log.append(event)
        
        # Trim event log if needed
        if len(self.event_log) > self.max_event_log_size:
            self.event_log = self.event_log[-self.max_event_log_size:]
            
        # Also log to application logger
        logger.info(f"[{event_type.upper()}] {message}")

# Initialize the global Project Manager Bot
project_manager = ProjectManagerBot()

# Start the manager automatically if not in testing mode
if not os.environ.get('TESTING'):
    project_manager.start()