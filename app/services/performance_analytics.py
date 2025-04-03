"""
ProjectPilot - AI-powered project management system
Performance analytics for tracking project and worker bot metrics.
"""

import logging
import json
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from flask import current_app
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from app import db
from app.models.project import Project
from app.models.task import Task
from app.models.worker_bot import WorkerBot

logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    """
    Analytics service for tracking and analyzing project performance metrics
    and worker bot efficiency.
    """
    
    def __init__(self):
        """Initialize the performance analytics service."""
        # Cache for analytics data
        self.analytics_cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.cache_timestamps = {}
        
        logger.info("Performance Analytics service initialized")
    
    def analyze_project(self, project_id: int) -> Dict[str, Any]:
        """
        Analyze a specific project's performance.
        
        Args:
            project_id: ID of the project to analyze
            
        Returns:
            Dictionary with project analytics
        """
        cache_key = f"project_{project_id}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.analytics_cache[cache_key]
        
        try:
            # Get project data
            project = Project.query.get(project_id)
            
            if not project:
                return {"error": f"Project with ID {project_id} not found"}
            
            # Get all tasks for this project
            tasks = Task.query.filter_by(project_id=project_id).all()
            
            # Get all worker bots for this project
            worker_bots = WorkerBot.query.filter_by(project_id=project_id).all()
            
            # Calculate project statistics
            project_stats = self._calculate_project_statistics(project, tasks)
            
            # Calculate task statistics
            task_stats = self._calculate_task_statistics(tasks)
            
            # Calculate worker bot statistics
            bot_stats = self._calculate_bot_statistics(worker_bots, tasks)
            
            # Calculate timeline metrics
            timeline_metrics = self._calculate_timeline_metrics(project, tasks)
            
            # Calculate velocity and burn rate
            velocity_metrics = self._calculate_velocity_metrics(project, tasks)
            
            # Generate overall health score
            health_score, health_factors = self._calculate_health_score(
                project_stats, task_stats, bot_stats, timeline_metrics, velocity_metrics
            )
            
            # Assemble analytics result
            analytics = {
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
                "project_stats": project_stats,
                "task_stats": task_stats,
                "bot_stats": bot_stats,
                "timeline_metrics": timeline_metrics,
                "velocity_metrics": velocity_metrics,
                "health": {
                    "score": health_score,
                    "rating": self._get_health_rating(health_score),
                    "factors": health_factors
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Cache the result
            self._update_cache(cache_key, analytics)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error analyzing project {project_id}: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def analyze_all_projects(self) -> Dict[str, Any]:
        """
        Analyze all projects in the system.
        
        Returns:
            Dictionary with analysis for all projects
        """
        cache_key = "all_projects"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.analytics_cache[cache_key]
        
        try:
            # Get all projects
            projects = Project.query.all()
            
            if not projects:
                return {"error": "No projects found"}
            
            # Analyze each project
            project_analyses = {}
            for project in projects:
                project_analyses[project.id] = self.analyze_project(project.id)
            
            # Calculate aggregate statistics
            total_projects = len(projects)
            active_projects = sum(1 for p in projects if p.status == 'active')
            completed_projects = sum(1 for p in projects if p.status == 'completed')
            
            # Calculate average health score
            avg_health_score = np.mean([
                analysis["health"]["score"] 
                for analysis in project_analyses.values() 
                if "health" in analysis and "score" in analysis["health"]
            ])
            
            # Get top performing projects
            top_projects = sorted(
                [(p.id, p.name, project_analyses[p.id]["health"]["score"] if "health" in project_analyses[p.id] else 0) 
                 for p in projects if p.id in project_analyses],
                key=lambda x: x[2],
                reverse=True
            )[:5]  # Top 5
            
            # Get most active bots
            all_bots = []
            for p in projects:
                if p.id in project_analyses and "bot_stats" in project_analyses[p.id]:
                    for bot in project_analyses[p.id]["bot_stats"]["bot_metrics"]:
                        all_bots.append({
                            "id": bot["id"],
                            "name": bot["name"],
                            "project_id": p.id,
                            "project_name": p.name,
                            "tasks_completed": bot["tasks_completed"],
                            "efficiency": bot["efficiency"]
                        })
            
            top_bots = sorted(all_bots, key=lambda x: x["tasks_completed"], reverse=True)[:5]  # Top 5
            
            # Assemble analytics result
            analytics = {
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total_projects": total_projects,
                    "active_projects": active_projects,
                    "completed_projects": completed_projects,
                    "avg_health_score": avg_health_score,
                    "health_rating": self._get_health_rating(avg_health_score)
                },
                "top_projects": top_projects,
                "top_bots": top_bots,
                "project_analyses": project_analyses
            }
            
            # Cache the result
            self._update_cache(cache_key, analytics)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error analyzing all projects: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def analyze_bot_performance(self, bot_id: int) -> Dict[str, Any]:
        """
        Analyze a specific worker bot's performance.
        
        Args:
            bot_id: ID of the worker bot to analyze
            
        Returns:
            Dictionary with bot analytics
        """
        cache_key = f"bot_{bot_id}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.analytics_cache[cache_key]
        
        try:
            # Get bot data
            bot = WorkerBot.query.get(bot_id)
            
            if not bot:
                return {"error": f"Worker bot with ID {bot_id} not found"}
            
            # Get all tasks assigned to this bot
            tasks = Task.query.filter_by(assigned_to_bot_id=bot_id).all()
            
            # Calculate bot metrics
            completed_tasks = [task for task in tasks if task.status == 'completed']
            in_progress_tasks = [task for task in tasks if task.status == 'in_progress']
            pending_tasks = [task for task in tasks if task.status == 'pending']
            
            # Calculate task durations
            task_durations = []
            for task in completed_tasks:
                if task.start_date and task.completion_date:
                    duration = (task.completion_date - task.start_date).total_seconds()
                    task_durations.append(duration)
            
            avg_task_duration = np.mean(task_durations) if task_durations else 0
            median_task_duration = np.median(task_durations) if task_durations else 0
            
            # Calculate task completion rate
            completion_rate = (len(completed_tasks) / len(tasks)) * 100 if tasks else 0
            
            # Calculate efficiency based on task complexity vs. duration
            efficiency_scores = []
            for task in completed_tasks:
                if task.start_date and task.completion_date:
                    duration = (task.completion_date - task.start_date).total_seconds()
                    complexity = task.weight  # Assuming weight represents complexity
                    
                    # Higher score means more efficient (completed complex tasks quickly)
                    if duration > 0:
                        efficiency = (complexity * 3600) / duration  # Normalize to hourly rate
                        efficiency_scores.append(min(efficiency, 10))  # Cap at 10
            
            avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0
            
            # Get performance over time
            time_performance = self._calculate_time_performance(bot, completed_tasks)
            
            # Assemble analytics result
            analytics = {
                "bot": {
                    "id": bot.id,
                    "name": bot.name,
                    "type": bot.type,
                    "status": bot.status,
                    "ai_provider": bot.ai_provider,
                    "ai_model": bot.ai_model,
                    "capabilities": bot.capabilities,
                    "project_id": bot.project_id,
                    "created_at": bot.created_at.isoformat(),
                    "updated_at": bot.updated_at.isoformat(),
                    "last_active": bot.last_active.isoformat() if bot.last_active else None
                },
                "task_metrics": {
                    "total_tasks": len(tasks),
                    "completed_tasks": len(completed_tasks),
                    "in_progress_tasks": len(in_progress_tasks),
                    "pending_tasks": len(pending_tasks),
                    "completion_rate": completion_rate,
                    "avg_task_duration": avg_task_duration,  # in seconds
                    "median_task_duration": median_task_duration,  # in seconds
                    "avg_efficiency": avg_efficiency
                },
                "time_performance": time_performance,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Calculate strengths and weaknesses
            analytics["insights"] = self._calculate_bot_insights(bot, analytics)
            
            # Cache the result
            self._update_cache(cache_key, analytics)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error analyzing bot {bot_id}: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def generate_project_dashboard(self, project_id: int) -> Dict[str, Any]:
        """
        Generate a dashboard for a project with visualizations.
        
        Args:
            project_id: ID of the project
            
        Returns:
            Dictionary with dashboard data and chart URLs
        """
        try:
            # Get project analytics
            analytics = self.analyze_project(project_id)
            
            if "error" in analytics:
                return analytics
            
            # Generate visualizations
            visualizations = {}
            
            # Task status distribution chart
            task_status_data = analytics["task_stats"]["status_distribution"]
            visualizations["task_status_chart"] = self._generate_pie_chart(
                title="Task Status Distribution",
                labels=list(task_status_data.keys()),
                values=list(task_status_data.values()),
                colors=['#28a745', '#ffc107', '#007bff', '#dc3545', '#6c757d']
            )
            
            # Burndown chart
            timeline = analytics["timeline_metrics"]
            if "burndown_data" in timeline:
                burndown = timeline["burndown_data"]
                visualizations["burndown_chart"] = self._generate_line_chart(
                    title="Project Burndown",
                    x_label="Date",
                    y_label="Remaining Work",
                    x_values=[item["date"] for item in burndown],
                    y_values=[item["remaining_work"] for item in burndown],
                    y_target=[item["ideal_burndown"] for item in burndown]
                )
            
            # Bot performance comparison
            bot_metrics = analytics["bot_stats"]["bot_metrics"]
            visualizations["bot_performance_chart"] = self._generate_bar_chart(
                title="Bot Performance Comparison",
                x_label="Bot",
                y_label="Tasks Completed",
                x_values=[bot["name"] for bot in bot_metrics],
                y_values=[bot["tasks_completed"] for bot in bot_metrics]
            )
            
            # Task completion over time
            if "completion_trend" in analytics["velocity_metrics"]:
                trend = analytics["velocity_metrics"]["completion_trend"]
                visualizations["completion_trend_chart"] = self._generate_line_chart(
                    title="Task Completion Trend",
                    x_label="Week",
                    y_label="Tasks Completed",
                    x_values=list(range(1, len(trend) + 1)),
                    y_values=trend
                )
            
            return {
                "project_id": project_id,
                "analytics": analytics,
                "visualizations": visualizations
            }
            
        except Exception as e:
            logger.error(f"Error generating project dashboard for {project_id}: {str(e)}")
            return {"error": f"Dashboard generation failed: {str(e)}"}
    
    def generate_system_health_report(self) -> Dict[str, Any]:
        """
        Generate a system-wide health report.
        
        Returns:
            Dictionary with health report data
        """
        try:
            # Get all projects analytics
            all_projects = self.analyze_all_projects()
            
            if "error" in all_projects:
                return all_projects
            
            # Get all bots
            all_bots = WorkerBot.query.all()
            
            # Calculate system metrics
            total_bots = len(all_bots)
            active_bots = sum(1 for bot in all_bots if bot.status in ['working', 'idle'])
            error_bots = sum(1 for bot in all_bots if bot.status == 'error')
            
            # Project health distribution
            health_distribution = {
                "excellent": 0,
                "good": 0,
                "fair": 0,
                "poor": 0,
                "critical": 0
            }
            
            for project_id, analysis in all_projects["project_analyses"].items():
                if "health" in analysis and "rating" in analysis["health"]:
                    rating = analysis["health"]["rating"]
                    health_distribution[rating] += 1
            
            # Calculate AI usage statistics
            ai_usage = self._calculate_ai_usage_statistics()
            
            # Calculate resource utilization
            resource_utilization = self._calculate_resource_utilization()
            
            # Assemble health report
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": {
                    "total_projects": all_projects["summary"]["total_projects"],
                    "active_projects": all_projects["summary"]["active_projects"],
                    "completed_projects": all_projects["summary"]["completed_projects"],
                    "total_bots": total_bots,
                    "active_bots": active_bots,
                    "error_bots": error_bots,
                    "bot_health_ratio": (active_bots / total_bots) * 100 if total_bots > 0 else 0
                },
                "health_distribution": health_distribution,
                "ai_usage": ai_usage,
                "resource_utilization": resource_utilization,
                "top_projects": all_projects["top_projects"],
                "top_bots": all_projects["top_bots"],
                "overall_health": {
                    "score": all_projects["summary"]["avg_health_score"],
                    "rating": all_projects["summary"]["health_rating"]
                }
            }
            
            # Generate visualizations
            visualizations = {}
            
            # Project health distribution chart
            visualizations["health_distribution_chart"] = self._generate_pie_chart(
                title="Project Health Distribution",
                labels=list(health_distribution.keys()),
                values=list(health_distribution.values()),
                colors=['#28a745', '#99c140', '#e7b416', '#db7b2b', '#cc3232']
            )
            
            # Resource utilization chart
            visualizations["resource_utilization_chart"] = self._generate_radar_chart(
                title="Resource Utilization",
                labels=list(resource_utilization.keys()),
                values=list(resource_utilization.values())
            )
            
            # AI provider usage chart
            visualizations["ai_provider_chart"] = self._generate_pie_chart(
                title="AI Provider Usage",
                labels=list(ai_usage["provider_distribution"].keys()),
                values=list(ai_usage["provider_distribution"].values()),
                colors=['#007bff', '#6f42c1', '#fd7e14', '#20c997', '#6c757d']
            )
            
            report["visualizations"] = visualizations
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating system health report: {str(e)}")
            return {"error": f"Health report generation failed: {str(e)}"}
    
    def _calculate_project_statistics(self, project: Project, tasks: List[Task]) -> Dict[str, Any]:
        """
        Calculate statistics for a project.
        
        Args:
            project: Project instance
            tasks: List of tasks for the project
            
        Returns:
            Dictionary with project statistics
        """
        # Calculate basic statistics
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task.status == 'completed')
        in_progress_tasks = sum(1 for task in tasks if task.status == 'in_progress')
        pending_tasks = sum(1 for task in tasks if task.status == 'pending')
        blocked_tasks = sum(1 for task in tasks if task.status == 'blocked')
        
        # Calculate completion percentage
        completion_percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Calculate project age in days
        age_days = (datetime.utcnow() - project.created_at).days
        
        # Calculate days since last update
        days_since_update = (datetime.utcnow() - project.updated_at).days
        
        # Calculate estimated completion date based on velocity
        estimated_completion = self._estimate_completion_date(project, tasks)
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "pending_tasks": pending_tasks,
            "blocked_tasks": blocked_tasks,
            "completion_percentage": completion_percentage,
            "age_days": age_days,
            "days_since_update": days_since_update,
            "estimated_completion": estimated_completion.isoformat() if estimated_completion else None
        }
    
    def _calculate_task_statistics(self, tasks: List[Task]) -> Dict[str, Any]:
        """
        Calculate statistics for tasks.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Dictionary with task statistics
        """
        # Calculate task status distribution
        status_distribution = {
            "completed": sum(1 for task in tasks if task.status == 'completed'),
            "in_progress": sum(1 for task in tasks if task.status == 'in_progress'),
            "pending": sum(1 for task in tasks if task.status == 'pending'),
            "blocked": sum(1 for task in tasks if task.status == 'blocked'),
            "review": sum(1 for task in tasks if task.status == 'review')
        }
        
        # Calculate task type distribution
        type_distribution = {}
        for task in tasks:
            task_type = task.type
            if task_type not in type_distribution:
                type_distribution[task_type] = 0
            type_distribution[task_type] += 1
        
        # Calculate task priority distribution
        priority_distribution = {
            1: sum(1 for task in tasks if task.priority == 1),
            2: sum(1 for task in tasks if task.priority == 2),
            3: sum(1 for task in tasks if task.priority == 3),
            4: sum(1 for task in tasks if task.priority == 4)
        }
        
        # Calculate average task completion time (in hours)
        completion_times = []
        for task in tasks:
            if task.status == 'completed' and task.start_date and task.completion_date:
                completion_time = (task.completion_date - task.start_date).total_seconds() / 3600  # hours
                completion_times.append(completion_time)
        
        avg_completion_time = np.mean(completion_times) if completion_times else 0
        median_completion_time = np.median(completion_times) if completion_times else 0
        
        # Count tasks with dependencies
        tasks_with_dependencies = sum(1 for task in tasks if task.dependencies.count() > 0)
        
        return {
            "status_distribution": status_distribution,
            "type_distribution": type_distribution,
            "priority_distribution": priority_distribution,
            "avg_completion_time": avg_completion_time,
            "median_completion_time": median_completion_time,
            "tasks_with_dependencies": tasks_with_dependencies,
            "dependency_ratio": (tasks_with_dependencies / len(tasks)) * 100 if tasks else 0
        }
    
    def _calculate_bot_statistics(self, bots: List[WorkerBot], tasks: List[Task]) -> Dict[str, Any]:
        """
        Calculate statistics for worker bots.
        
        Args:
            bots: List of worker bots
            tasks: List of tasks
            
        Returns:
            Dictionary with bot statistics
        """
        # Calculate bot metrics
        bot_metrics = []
        
        for bot in bots:
            # Get tasks assigned to this bot
            bot_tasks = [task for task in tasks if task.assigned_to_bot_id == bot.id]
            completed_tasks = [task for task in bot_tasks if task.status == 'completed']
            
            # Calculate efficiency
            efficiency = self._calculate_bot_efficiency(bot, completed_tasks)
            
            # Calculate average task duration
            durations = []
            for task in completed_tasks:
                if task.start_date and task.completion_date:
                    duration = (task.completion_date - task.start_date).total_seconds() / 3600  # hours
                    durations.append(duration)
            
            avg_duration = np.mean(durations) if durations else 0
            
            bot_metrics.append({
                "id": bot.id,
                "name": bot.name,
                "type": bot.type,
                "status": bot.status,
                "tasks_assigned": len(bot_tasks),
                "tasks_completed": len(completed_tasks),
                "completion_rate": (len(completed_tasks) / len(bot_tasks)) * 100 if bot_tasks else 0,
                "avg_task_duration": avg_duration,
                "efficiency": efficiency
            })
        
        # Calculate bot type distribution
        type_distribution = {}
        for bot in bots:
            bot_type = bot.type
            if bot_type not in type_distribution:
                type_distribution[bot_type] = 0
            type_distribution[bot_type] += 1
        
        # Calculate AI provider distribution
        provider_distribution = {}
        for bot in bots:
            provider = bot.ai_provider
            if provider not in provider_distribution:
                provider_distribution[provider] = 0
            provider_distribution[provider] += 1
        
        # Find most efficient bot
        most_efficient_bot = max(bot_metrics, key=lambda x: x["efficiency"]) if bot_metrics else None
        
        return {
            "total_bots": len(bots),
            "active_bots": sum(1 for bot in bots if bot.status in ['working', 'idle']),
            "type_distribution": type_distribution,
            "provider_distribution": provider_distribution,
            "bot_metrics": bot_metrics,
            "most_efficient_bot": most_efficient_bot["name"] if most_efficient_bot else None
        }
    
    def _calculate_timeline_metrics(self, project: Project, tasks: List[Task]) -> Dict[str, Any]:
        """
        Calculate timeline metrics for a project.
        
        Args:
            project: Project instance
            tasks: List of tasks
            
        Returns:
            Dictionary with timeline metrics
        """
        # Calculate project duration
        start_date = project.start_date or project.created_at
        end_date = project.end_date or datetime.utcnow()
        
        if project.status == 'completed':
            # For completed projects, use the actual end date
            duration_days = (project.end_date - start_date).days
            remaining_days = 0
        else:
            # For ongoing projects, calculate remaining time
            duration_days = (end_date - start_date).days
            
            # Estimate remaining time
            remaining_days = self._estimate_remaining_days(project, tasks)
        
        # Generate burndown data
        burndown_data = self._generate_burndown_data(project, tasks)
        
        # Check if project is on schedule
        on_schedule = self._is_project_on_schedule(project, tasks, burndown_data)
        
        # Calculate critical path tasks
        critical_path = self._find_critical_path_tasks(tasks)
        
        return {
            "start_date": start_date.isoformat(),
            "planned_end_date": project.end_date.isoformat() if project.end_date else None,
            "estimated_end_date": (datetime.utcnow() + timedelta(days=remaining_days)).isoformat(),
            "duration_days": duration_days,
            "remaining_days": remaining_days,
            "burndown_data": burndown_data,
            "on_schedule": on_schedule,
            "critical_path_tasks": critical_path
        }
    
    def _calculate_velocity_metrics(self, project: Project, tasks: List[Task]) -> Dict[str, Any]:
        """
        Calculate velocity metrics for a project.
        
        Args:
            project: Project instance
            tasks: List of tasks
            
        Returns:
            Dictionary with velocity metrics
        """
        # Collect completed tasks with dates
        completed_tasks_with_dates = [
            task for task in tasks 
            if task.status == 'completed' and task.completion_date
        ]
        
        # Sort by completion date
        completed_tasks_with_dates.sort(key=lambda x: x.completion_date)
        
        # Calculate completion velocity (tasks per week)
        if not completed_tasks_with_dates:
            return {
                "velocity": 0,
                "estimated_completion_weeks": None,
                "burn_rate": 0,
                "completion_trend": []
            }
        
        # Group tasks by week
        weeks = {}
        first_date = completed_tasks_with_dates[0].completion_date
        
        for task in completed_tasks_with_dates:
            week_number = ((task.completion_date - first_date).days // 7) + 1
            if week_number not in weeks:
                weeks[week_number] = []
            weeks[week_number].append(task)
        
        # Calculate tasks per week
        tasks_per_week = [len(weeks.get(i, [])) for i in range(1, max(weeks.keys()) + 1)]
        
        # Calculate velocity (average tasks per week)
        velocity = np.mean(tasks_per_week) if tasks_per_week else 0
        
        # Estimate remaining time
        remaining_tasks = len([task for task in tasks if task.status != 'completed'])
        estimated_completion_weeks = remaining_tasks / velocity if velocity > 0 else None
        
        # Calculate burn rate (story points per week)
        burn_rate = sum(task.weight for week in weeks.values() for task in week) / len(weeks) if weeks else 0
        
        return {
            "velocity": velocity,
            "estimated_completion_weeks": estimated_completion_weeks,
            "burn_rate": burn_rate,
            "completion_trend": tasks_per_week
        }
    
    def _calculate_health_score(self,
                              project_stats: Dict[str, Any],
                              task_stats: Dict[str, Any],
                              bot_stats: Dict[str, Any],
                              timeline_metrics: Dict[str, Any],
                              velocity_metrics: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate a health score for the project.
        
        Args:
            project_stats: Project statistics
            task_stats: Task statistics
            bot_stats: Bot statistics
            timeline_metrics: Timeline metrics
            velocity_metrics: Velocity metrics
            
        Returns:
            Tuple of (health score, health factors)
        """
        health_factors = {}
        
        # Factor 1: Progress vs. Time elapsed (30%)
        progress_score = 0
        if timeline_metrics.get("on_schedule"):
            progress_score = 10
        elif project_stats.get("completion_percentage", 0) > 0:
            # Calculate based on progress vs. time elapsed
            completion_pct = project_stats.get("completion_percentage", 0)
            
            # Calculate percentage of time elapsed
            start_date = datetime.fromisoformat(timeline_metrics.get("start_date"))
            planned_end_date = (datetime.fromisoformat(timeline_metrics.get("planned_end_date")) 
                              if timeline_metrics.get("planned_end_date") else None)
            
            if planned_end_date:
                total_days = (planned_end_date - start_date).days
                elapsed_days = (datetime.utcnow() - start_date).days
                time_elapsed_pct = (elapsed_days / total_days) * 100 if total_days > 0 else 100
                
                # Compare progress to time elapsed
                progress_ratio = completion_pct / time_elapsed_pct if time_elapsed_pct > 0 else 0
                progress_score = min(10, progress_ratio * 10)  # Scale to 0-10
        
        health_factors["progress_score"] = progress_score
        
        # Factor 2: Task distribution (20%)
        task_distribution_score = 0
        blocked_ratio = project_stats.get("blocked_tasks", 0) / project_stats.get("total_tasks", 1)
        in_progress_ratio = project_stats.get("in_progress_tasks", 0) / project_stats.get("total_tasks", 1)
        
        # Penalize for too many blocked tasks, reward for good in-progress ratio
        task_distribution_score = 10 - (blocked_ratio * 10)  # Reduce score for blocked tasks
        task_distribution_score += 5 * (0.2 - abs(in_progress_ratio - 0.2))  # Optimal in-progress ratio ~20%
        task_distribution_score = max(0, min(10, task_distribution_score))  # Clamp to 0-10
        
        health_factors["task_distribution_score"] = task_distribution_score
        
        # Factor 3: Bot performance (20%)
        bot_performance_score = 0
        active_bots_ratio = bot_stats.get("active_bots", 0) / bot_stats.get("total_bots", 1) if bot_stats.get("total_bots", 0) > 0 else 0
        
        # Average bot efficiency
        bot_efficiencies = [bot.get("efficiency", 0) for bot in bot_stats.get("bot_metrics", [])]
        avg_bot_efficiency = np.mean(bot_efficiencies) if bot_efficiencies else 0
        
        # Combine scores
        bot_performance_score = (active_bots_ratio * 5) + (avg_bot_efficiency / 2)  # Scale efficiency to 0-5
        bot_performance_score = max(0, min(10, bot_performance_score))  # Clamp to 0-10
        
        health_factors["bot_performance_score"] = bot_performance_score
        
        # Factor 4: Velocity stability (15%)
        velocity_score = 0
        completion_trend = velocity_metrics.get("completion_trend", [])
        
        if completion_trend and len(completion_trend) >= 2:
            # Calculate coefficient of variation (lower is better)
            cv = np.std(completion_trend) / np.mean(completion_trend) if np.mean(completion_trend) > 0 else 1
            velocity_stability = max(0, 1 - cv)  # 1 = perfectly stable, 0 = highly variable
            
            # Recent trend (is velocity increasing?)
            recent_velocity = np.mean(completion_trend[-2:])
            overall_velocity = np.mean(completion_trend)
            velocity_trend = recent_velocity / overall_velocity if overall_velocity > 0 else 1
            
            # Combine stability and trend
            velocity_score = (velocity_stability * 5) + (min(velocity_trend, 2) * 2.5)  # Scale to 0-10
            velocity_score = max(0, min(10, velocity_score))
        
        health_factors["velocity_score"] = velocity_score
        
        # Factor 5: Deadline risk (15%)
        deadline_score = 0
        if timeline_metrics.get("planned_end_date") and timeline_metrics.get("estimated_end_date"):
            planned_end = datetime.fromisoformat(timeline_metrics["planned_end_date"])
            estimated_end = datetime.fromisoformat(timeline_metrics["estimated_end_date"])
            
            # Calculate difference in days
            diff_days = (planned_end - estimated_end).days
            
            # Positive diff means ahead of schedule, negative means behind
            if diff_days >= 0:
                deadline_score = 10  # Ahead of schedule
            else:
                # Scale based on how far behind
                delay_ratio = abs(diff_days) / max(timeline_metrics.get("duration_days", 30), 30)
                deadline_score = max(0, 10 - (delay_ratio * 20))  # Higher penalty for longer delays
        
        health_factors["deadline_score"] = deadline_score
        
        # Calculate weighted score
        health_score = (
            progress_score * 0.3 +
            task_distribution_score * 0.2 +
            bot_performance_score * 0.2 +
            velocity_score * 0.15 +
            deadline_score * 0.15
        )
        
        return health_score, health_factors
    
    def _get_health_rating(self, score: float) -> str:
        """
        Convert a health score to a rating.
        
        Args:
            score: Health score (0-10)
            
        Returns:
            Rating string
        """
        if score >= 8:
            return "excellent"
        elif score >= 6.5:
            return "good"
        elif score >= 5:
            return "fair"
        elif score >= 3:
            return "poor"
        else:
            return "critical"
    
    def _calculate_bot_efficiency(self, bot: WorkerBot, completed_tasks: List[Task]) -> float:
        """
        Calculate efficiency score for a worker bot.
        
        Args:
            bot: Worker bot instance
            completed_tasks: List of completed tasks
            
        Returns:
            Efficiency score (0-10)
        """
        if not completed_tasks:
            return 0
        
        # Calculate based on task complexity vs. duration
        efficiency_scores = []
        for task in completed_tasks:
            if task.start_date and task.completion_date:
                duration = (task.completion_date - task.start_date).total_seconds()
                complexity = task.weight  # Assuming weight represents complexity
                
                # Higher score means more efficient (completed complex tasks quickly)
                if duration > 0:
                    efficiency = (complexity * 3600) / duration  # Normalize to hourly rate
                    efficiency_scores.append(min(efficiency, 10))  # Cap at 10
        
        return np.mean(efficiency_scores) if efficiency_scores else 0
    
    def _estimate_completion_date(self, project: Project, tasks: List[Task]) -> Optional[datetime]:
        """
        Estimate project completion date based on velocity.
        
        Args:
            project: Project instance
            tasks: List of tasks
            
        Returns:
            Estimated completion date or None
        """
        # If project is completed, return actual end date
        if project.status == 'completed' and project.end_date:
            return project.end_date
        
        # Calculate velocity based on completed tasks
        completed_tasks = [task for task in tasks if task.status == 'completed' and task.completion_date]
        
        if not completed_tasks:
            return None
        
        # Sort by completion date
        completed_tasks.sort(key=lambda x: x.completion_date)
        
        # Calculate average completion rate (tasks per day)
        first_completion = completed_tasks[0].completion_date
        last_completion = completed_tasks[-1].completion_date
        days_span = (last_completion - first_completion).days
        
        if days_span < 1:
            days_span = 1
        
        tasks_per_day = len(completed_tasks) / days_span
        
        # Count remaining tasks
        remaining_tasks = sum(1 for task in tasks if task.status != 'completed')
        
        # Estimate days needed
        if tasks_per_day > 0:
            days_needed = remaining_tasks / tasks_per_day
            return datetime.utcnow() + timedelta(days=days_needed)
        else:
            return None
    
    def _estimate_remaining_days(self, project: Project, tasks: List[Task]) -> int:
        """
        Estimate remaining days for a project.
        
        Args:
            project: Project instance
            tasks: List of tasks
            
        Returns:
            Estimated remaining days
        """
        estimated_end = self._estimate_completion_date(project, tasks)
        
        if estimated_end:
            remaining_days = (estimated_end - datetime.utcnow()).days
            return max(0, remaining_days)
        else:
            # Fallback: estimate based on completion percentage
            completed_pct = project.progress
            if completed_pct > 0:
                elapsed_days = (datetime.utcnow() - project.start_date).days if project.start_date else 0
                remaining_days = elapsed_days * ((100 - completed_pct) / completed_pct) if completed_pct > 0 else 0
                return max(0, int(remaining_days))
            else:
                return 0
    
    def _generate_burndown_data(self, project: Project, tasks: List[Task]) -> List[Dict[str, Any]]:
        """
        Generate burndown chart data.
        
        Args:
            project: Project instance
            tasks: List of tasks
            
        Returns:
            List of data points for burndown chart
        """
        # If no start date or no tasks, return empty list
        if not project.start_date or not tasks:
            return []
        
        # Define date range
        start_date = project.start_date
        if project.status == 'completed' and project.end_date:
            end_date = project.end_date
        else:
            # For ongoing projects, use estimated end date or today + 30 days
            estimated_end = self._estimate_completion_date(project, tasks)
            end_date = estimated_end if estimated_end else datetime.utcnow() + timedelta(days=30)
        
        # Calculate total work (sum of task weights)
        total_work = sum(task.weight for task in tasks)
        
        # Generate daily data points
        burndown_data = []
        
        # Get tasks completed by date
        tasks_by_date = {}
        for task in tasks:
            if task.status == 'completed' and task.completion_date:
                date_key = task.completion_date.date().isoformat()
                if date_key not in tasks_by_date:
                    tasks_by_date[date_key] = []
                tasks_by_date[date_key].append(task)
        
        # Generate daily data
        current_date = start_date.date()
        remaining_work = total_work
        
        # Calculate ideal burndown (linear)
        total_days = (end_date.date() - start_date.date()).days
        daily_ideal_decrease = total_work / total_days if total_days > 0 else 0
        
        while current_date <= end_date.date():
            date_key = current_date.isoformat()
            
            # Update remaining work based on completed tasks
            if date_key in tasks_by_date:
                work_completed = sum(task.weight for task in tasks_by_date[date_key])
                remaining_work -= work_completed
            
            # Calculate ideal burndown for this date
            days_elapsed = (current_date - start_date.date()).days
            ideal_burndown = max(0, total_work - (daily_ideal_decrease * days_elapsed))
            
            # Add data point
            burndown_data.append({
                "date": date_key,
                "remaining_work": max(0, remaining_work),
                "ideal_burndown": ideal_burndown
            })
            
            current_date += timedelta(days=1)
            
            # Stop if we've reached today for ongoing projects
            if project.status != 'completed' and current_date > datetime.utcnow().date():
                break
        
        return burndown_data
    
    def _is_project_on_schedule(self, project: Project, tasks: List[Task], burndown_data: List[Dict[str, Any]]) -> bool:
        """
        Check if a project is on schedule.
        
        Args:
            project: Project instance
            tasks: List of tasks
            burndown_data: Burndown chart data
            
        Returns:
            True if project is on schedule
        """
        # If no burndown data, can't determine
        if not burndown_data:
            return True
        
        # Get latest data point
        latest = burndown_data[-1]
        
        # Compare actual to ideal
        return latest["remaining_work"] <= latest["ideal_burndown"] * 1.1  # Allow 10% buffer
    
    def _find_critical_path_tasks(self, tasks: List[Task]) -> List[Dict[str, Any]]:
        """
        Find tasks on the critical path.
        
        Args:
            tasks: List of tasks
            
        Returns:
            List of critical tasks
        """
        # For a more accurate critical path, we would need to build a proper
        # dependency graph and run a critical path algorithm.
        # This is a simplified approximation.
        
        # Find tasks with dependencies
        tasks_with_deps = [task for task in tasks if task.dependencies.count() > 0]
        
        # Find tasks that are depended upon by many others
        dependency_counts = {}
        for task in tasks:
            for dep in task.dependencies:
                dep_id = dep.depends_on_id
                if dep_id not in dependency_counts:
                    dependency_counts[dep_id] = 0
                dependency_counts[dep_id] += 1
        
        # Find task IDs with highest dependency counts
        critical_ids = sorted(dependency_counts.keys(), key=lambda k: dependency_counts[k], reverse=True)[:5]
        
        # Get the actual task objects
        critical_tasks = []
        for task_id in critical_ids:
            task = next((t for t in tasks if t.id == task_id), None)
            if task:
                critical_tasks.append({
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "dependency_count": dependency_counts.get(task.id, 0)
                })
        
        return critical_tasks
    
    def _calculate_time_performance(self, bot: WorkerBot, completed_tasks: List[Task]) -> Dict[str, Any]:
        """
        Calculate performance over time for a bot.
        
        Args:
            bot: Worker bot instance
            completed_tasks: List of completed tasks
            
        Returns:
            Dictionary with time performance metrics
        """
        if not completed_tasks:
            return {
                "completion_trend": [],
                "efficiency_trend": [],
                "performance_improving": False
            }
        
        # Sort tasks by completion date
        completed_tasks.sort(key=lambda x: x.completion_date if x.completion_date else datetime.max)
        
        # Group by week
        weekly_tasks = {}
        first_date = completed_tasks[0].completion_date if completed_tasks[0].completion_date else bot.created_at
        
        for task in completed_tasks:
            if task.completion_date:
                week_num = ((task.completion_date - first_date).days // 7) + 1
                if week_num not in weekly_tasks:
                    weekly_tasks[week_num] = []
                weekly_tasks[week_num].append(task)
        
        # Calculate weekly metrics
        completion_trend = []
        efficiency_trend = []
        
        for week in sorted(weekly_tasks.keys()):
            tasks = weekly_tasks[week]
            completion_trend.append(len(tasks))
            
            # Calculate efficiency for this week
            efficiency = self._calculate_bot_efficiency(bot, tasks)
            efficiency_trend.append(efficiency)
        
        # Determine if performance is improving
        performance_improving = False
        if len(efficiency_trend) >= 2:
            recent_avg = np.mean(efficiency_trend[-2:])
            earlier_avg = np.mean(efficiency_trend[:-2]) if len(efficiency_trend) > 2 else efficiency_trend[0]
            performance_improving = recent_avg > earlier_avg
        
        return {
            "completion_trend": completion_trend,
            "efficiency_trend": efficiency_trend,
            "performance_improving": performance_improving
        }
    
    def _calculate_bot_insights(self, bot: WorkerBot, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate strengths and weaknesses for a bot.
        
        Args:
            bot: Worker bot instance
            analytics: Bot analytics data
            
        Returns:
            Dictionary with insights
        """
        task_metrics = analytics.get("task_metrics", {})
        time_performance = analytics.get("time_performance", {})
        
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Check completion rate
        completion_rate = task_metrics.get("completion_rate", 0)
        if completion_rate >= 80:
            strengths.append("High task completion rate")
        elif completion_rate <= 50:
            weaknesses.append("Low task completion rate")
            recommendations.append("Investigate why tasks are not being completed")
        
        # Check efficiency
        avg_efficiency = task_metrics.get("avg_efficiency", 0)
        if avg_efficiency >= 7:
            strengths.append("High efficiency rating")
        elif avg_efficiency <= 3:
            weaknesses.append("Low efficiency rating")
            recommendations.append("Consider optimizing the bot's AI model or capabilities")
        
        # Check performance trend
        if time_performance.get("performance_improving", False):
            strengths.append("Improving performance over time")
        elif len(time_performance.get("efficiency_trend", [])) >= 3:
            avg_trend = np.mean(np.diff(time_performance.get("efficiency_trend", [])))
            if avg_trend < 0:
                weaknesses.append("Declining performance over time")
                recommendations.append("Review recent task assignments for potential issues")
        
        # Check task duration
        avg_duration = task_metrics.get("avg_task_duration", 0)
        if avg_duration > 0:
            if avg_duration <= 4:  # 4 hours
                strengths.append("Fast task completion times")
            elif avg_duration >= 24:  # 24 hours
                weaknesses.append("Slow task completion times")
                recommendations.append("Consider assigning simpler tasks or improving the bot's capabilities")
        
        # Check inactivity
        if bot.status == 'idle' and bot.last_active:
            idle_time = (datetime.utcnow() - bot.last_active).total_seconds() / 3600  # hours
            if idle_time > 48:
                weaknesses.append(f"Bot has been idle for {idle_time:.1f} hours")
                recommendations.append("Assign new tasks to utilize this bot")
        
        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations
        }
    
    def _calculate_ai_usage_statistics(self) -> Dict[str, Any]:
        """
        Calculate AI usage statistics.
        
        Returns:
            Dictionary with AI usage statistics
        """
        # Get all bots
        bots = WorkerBot.query.all()
        
        # Calculate provider distribution
        provider_distribution = {}
        model_distribution = {}
        
        for bot in bots:
            # Count providers
            provider = bot.ai_provider
            if provider not in provider_distribution:
                provider_distribution[provider] = 0
            provider_distribution[provider] += 1
            
            # Count models
            model = bot.ai_model
            if model not in model_distribution:
                model_distribution[model] = 0
            model_distribution[model] += 1
        
        # TODO: In a real implementation, we would track actual token usage
        # For now, we'll simulate with some values
        
        # Example token usage data
        token_usage = {
            "openai": 1_500_000,
            "anthropic": 2_100_000,
            "bedrock": 800_000
        }
        
        # Calculate total and average
        total_tokens = sum(token_usage.values())
        avg_tokens_per_project = total_tokens / Project.query.count() if Project.query.count() > 0 else 0
        
        return {
            "provider_distribution": provider_distribution,
            "model_distribution": model_distribution,
            "token_usage": token_usage,
            "total_tokens": total_tokens,
            "avg_tokens_per_project": avg_tokens_per_project
        }
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """
        Calculate resource utilization metrics.
        
        Returns:
            Dictionary with resource utilization percentages
        """
        # TODO: In a real implementation, we would track actual resource usage
        # For now, we'll simulate with some values
        
        return {
            "cpu": 65.0,
            "memory": 72.0,
            "storage": 48.0,
            "network": 55.0,
            "api_rate": 42.0
        }
    
    def _is_cache_valid(self, key: str) -> bool:
        """
        Check if cache is valid for a key.
        
        Args:
            key: Cache key
            
        Returns:
            True if cache is valid
        """
        if key not in self.analytics_cache:
            return False
            
        if key not in self.cache_timestamps:
            return False
            
        # Check if cache has expired
        timestamp = self.cache_timestamps[key]
        cache_age = (datetime.utcnow() - timestamp).total_seconds()
        
        return cache_age < self.cache_ttl
    
    def _update_cache(self, key: str, value: Any) -> None:
        """
        Update cache for a key.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.analytics_cache[key] = value
        self.cache_timestamps[key] = datetime.utcnow()
    
    def _generate_pie_chart(self, 
                          title: str, 
                          labels: List[str], 
                          values: List[float],
                          colors: Optional[List[str]] = None) -> str:
        """
        Generate a pie chart and return as base64 image.
        
        Args:
            title: Chart title
            labels: Category labels
            values: Values for each category
            colors: Optional colors for each category
            
        Returns:
            Base64 encoded PNG image
        """
        plt.figure(figsize=(8, 6))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.axis('equal')
        plt.title(title)
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
    
    def _generate_bar_chart(self, 
                          title: str, 
                          x_label: str,
                          y_label: str,
                          x_values: List[str], 
                          y_values: List[float]) -> str:
        """
        Generate a bar chart and return as base64 image.
        
        Args:
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            x_values: X-axis values
            y_values: Y-axis values
            
        Returns:
            Base64 encoded PNG image
        """
        plt.figure(figsize=(10, 6))
        plt.bar(x_values, y_values, color='#007bff')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
    
    def _generate_line_chart(self, 
                           title: str, 
                           x_label: str,
                           y_label: str,
                           x_values: List[Any], 
                           y_values: List[float],
                           y_target: Optional[List[float]] = None) -> str:
        """
        Generate a line chart and return as base64 image.
        
        Args:
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            x_values: X-axis values
            y_values: Y-axis values
            y_target: Optional target line values
            
        Returns:
            Base64 encoded PNG image
        """
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, 'o-', color='#007bff', label='Actual')
        
        if y_target:
            plt.plot(x_values, y_target, '--', color='#dc3545', label='Target')
            plt.legend()
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
    
    def _generate_radar_chart(self, 
                            title: str, 
                            labels: List[str], 
                            values: List[float]) -> str:
        """
        Generate a radar chart and return as base64 image.
        
        Args:
            title: Chart title
            labels: Category labels
            values: Values for each category
            
        Returns:
            Base64 encoded PNG image
        """
        # Calculate angles for each category
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        
        # Close the polygon
        values = values + [values[0]]
        angles = angles + [angles[0]]
        labels = labels + [labels[0]]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2, color='#007bff')
        ax.fill(angles, values, alpha=0.25, color='#007bff')
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        
        # Set y-limits
        ax.set_ylim(0, 100)
        
        # Add title
        plt.title(title, size=14)
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"