"""
ProjectPilot - AI-powered project management system
Worker Bot Manager for monitoring and managing worker bots.
"""

import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
import boto3
from flask import current_app

from app.models.worker_bot import WorkerBot
from app.models.task import Task
from app import db

logger = logging.getLogger(__name__)

class BotManagerLambda:
    """
    Service for monitoring and managing worker bots, ensuring their availability
    and optimal performance. This service can be run as an AWS Lambda or as a
    background task.
    """
    
    def __init__(self, 
                region: Optional[str] = None,
                aws_access_key_id: Optional[str] = None,
                aws_secret_access_key: Optional[str] = None):
        """
        Initialize the Bot Manager.
        
        Args:
            region: AWS region for CloudWatch Metrics (if using AWS)
            aws_access_key_id: AWS access key ID (if using AWS)
            aws_secret_access_key: AWS secret access key (if using AWS)
        """
        self.region = region or os.environ.get('AWS_REGION') or current_app.config.get('AWS_REGION', 'eu-west-2')
        self.aws_access_key_id = aws_access_key_id or os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
        
        # Initialize AWS clients if credentials are provided
        self.cloudwatch = None
        if self.aws_access_key_id and self.aws_secret_access_key:
            try:
                session = boto3.Session(
                    region_name=self.region,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key
                )
                self.cloudwatch = session.client('cloudwatch')
                logger.info(f"AWS CloudWatch client initialized in region {self.region}")
            except Exception as e:
                logger.error(f"Failed to initialize AWS CloudWatch client: {str(e)}")
        
        # Bot status monitoring data
        self.bot_metrics = {}
        self.alert_thresholds = {
            'max_task_age': 24 * 60 * 60,  # 24 hours in seconds
            'max_error_count': 3,
            'max_bot_inactive_time': 48 * 60 * 60,  # 48 hours in seconds
            'high_cpu_threshold': 80,  # 80% CPU usage
            'low_memory_threshold': 20   # 20% memory remaining
        }
        
        logger.info("Bot Manager initialized")
    
    def monitor_all_bots(self) -> Dict[str, Any]:
        """
        Monitor all worker bots in the system and generate a report.
        
        Returns:
            Dictionary with monitoring report
        """
        try:
            # Get all worker bots
            bots = WorkerBot.query.all()
            
            if not bots:
                logger.info("No worker bots found")
                return {"status": "success", "message": "No worker bots found to monitor"}
            
            bot_reports = []
            bots_requiring_attention = []
            
            for bot in bots:
                bot_report = self.monitor_bot(bot)
                bot_reports.append(bot_report)
                
                # Check if this bot needs attention
                if bot_report.get("status") in ["error", "warning"]:
                    bots_requiring_attention.append({
                        "bot_id": bot.id, 
                        "name": bot.name,
                        "status": bot_report.get("status"),
                        "issues": bot_report.get("issues", [])
                    })
            
            # Generate summary report
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_bots": len(bots),
                "healthy_bots": len([r for r in bot_reports if r.get("status") == "healthy"]),
                "warning_bots": len([r for r in bot_reports if r.get("status") == "warning"]),
                "error_bots": len([r for r in bot_reports if r.get("status") == "error"]),
                "bots_requiring_attention": bots_requiring_attention,
                "bot_reports": bot_reports
            }
            
            logger.info(f"Bot monitoring summary: {len(bots)} bots, {len(bots_requiring_attention)} need attention")
            
            # Publish metrics to CloudWatch if available
            if self.cloudwatch:
                self._publish_metrics(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error monitoring bots: {str(e)}")
            return {"status": "error", "message": f"Monitoring failed: {str(e)}"}
    
    def monitor_bot(self, bot: WorkerBot) -> Dict[str, Any]:
        """
        Monitor a specific worker bot.
        
        Args:
            bot: WorkerBot instance to monitor
            
        Returns:
            Dictionary with monitoring report for this bot
        """
        try:
            issues = []
            
            # Check bot status
            if bot.status == "error":
                issues.append({"type": "status", "severity": "error", "message": "Bot is in error state"})
            elif bot.status == "inactive":
                issues.append({"type": "status", "severity": "warning", "message": "Bot is inactive"})
            
            # Check if bot has been stuck on the same task for too long
            if bot.status == "working":
                current_task = bot.assigned_tasks.filter_by(status='in_progress').first()
                if current_task and current_task.start_date:
                    task_age = (datetime.utcnow() - current_task.start_date).total_seconds()
                    if task_age > self.alert_thresholds['max_task_age']:
                        issues.append({
                            "type": "stuck_task", 
                            "severity": "error", 
                            "message": f"Bot has been working on task '{current_task.name}' for {task_age/3600:.1f} hours"
                        })
            
            # Check if idle bot has tasks waiting
            if bot.status == "idle":
                # Find pending tasks that match this bot's capabilities
                pending_tasks = Task.query.filter_by(
                    project_id=bot.project_id,
                    status='pending',
                    assigned_to_bot_id=None
                ).all()
                
                # Filter tasks that match this bot's capabilities
                matching_tasks = []
                for task in pending_tasks:
                    if self._task_matches_bot_capabilities(task, bot):
                        matching_tasks.append(task)
                
                if matching_tasks:
                    issues.append({
                        "type": "idle_with_tasks", 
                        "severity": "warning", 
                        "message": f"Bot is idle but {len(matching_tasks)} matching tasks are waiting"
                    })
            
            # Check if bot has been inactive for too long
            if bot.last_active:
                inactive_time = (datetime.utcnow() - bot.last_active).total_seconds()
                if inactive_time > self.alert_thresholds['max_bot_inactive_time']:
                    issues.append({
                        "type": "inactive", 
                        "severity": "warning", 
                        "message": f"Bot has been inactive for {inactive_time/3600:.1f} hours"
                    })
            
            # Collect performance metrics for this bot
            metrics = {
                "task_completion_rate": self._calculate_task_completion_rate(bot),
                "average_task_duration": self._calculate_average_task_duration(bot),
                "error_rate": self._calculate_error_rate(bot)
            }
            
            # Store metrics for trending
            self._update_bot_metrics(bot.id, metrics)
            
            # Determine overall status
            status = "healthy"
            if any(issue["severity"] == "error" for issue in issues):
                status = "error"
            elif any(issue["severity"] == "warning" for issue in issues):
                status = "warning"
            
            return {
                "bot_id": bot.id,
                "name": bot.name,
                "type": bot.type,
                "status": status,
                "current_status": bot.status,
                "project_id": bot.project_id,
                "capabilities": bot.capabilities,
                "ai_provider": bot.ai_provider,
                "ai_model": bot.ai_model,
                "last_active": bot.last_active.isoformat() if bot.last_active else None,
                "tasks_assigned": bot.assigned_tasks.count(),
                "tasks_completed": bot.assigned_tasks.filter_by(status='completed').count(),
                "issues": issues,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error monitoring bot {bot.id}: {str(e)}")
            return {
                "bot_id": bot.id,
                "name": bot.name,
                "status": "error",
                "issues": [{"type": "monitoring_error", "severity": "error", "message": f"Error monitoring bot: {str(e)}"}]
            }
    
    def auto_remediate_issues(self) -> Dict[str, Any]:
        """
        Automatically remediate issues with worker bots where possible.
        
        Returns:
            Dictionary with remediation report
        """
        try:
            # First get a monitoring report
            monitoring_report = self.monitor_all_bots()
            bots_requiring_attention = monitoring_report.get("bots_requiring_attention", [])
            
            if not bots_requiring_attention:
                logger.info("No bots requiring remediation")
                return {"status": "success", "message": "No bots requiring remediation"}
            
            remediation_actions = []
            
            for bot_issue in bots_requiring_attention:
                bot_id = bot_issue.get("bot_id")
                bot = WorkerBot.query.get(bot_id)
                
                if not bot:
                    logger.warning(f"Bot {bot_id} not found for remediation")
                    continue
                
                # Attempt to remediate issues
                for issue in bot_issue.get("issues", []):
                    remediation_result = self._remediate_issue(bot, issue)
                    if remediation_result:
                        remediation_actions.append(remediation_result)
            
            # Generate summary report
            summary = {
                "timestamp": datetime.now().isoformat(),
                "bots_remediated": len(set(action.get("bot_id") for action in remediation_actions)),
                "actions_taken": len(remediation_actions),
                "successful_actions": len([a for a in remediation_actions if a.get("success")]),
                "failed_actions": len([a for a in remediation_actions if not a.get("success")]),
                "remediation_actions": remediation_actions
            }
            
            logger.info(f"Auto-remediation summary: {summary['actions_taken']} actions, {summary['successful_actions']} successful")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error during auto-remediation: {str(e)}")
            return {"status": "error", "message": f"Auto-remediation failed: {str(e)}"}
    
    def _remediate_issue(self, bot: WorkerBot, issue: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to remediate a specific issue with a bot.
        
        Args:
            bot: WorkerBot instance with an issue
            issue: Issue dictionary from monitor_bot
            
        Returns:
            Dictionary describing remediation action taken
        """
        issue_type = issue.get("type")
        
        remediation_result = {
            "bot_id": bot.id,
            "bot_name": bot.name,
            "issue_type": issue_type,
            "timestamp": datetime.now().isoformat(),
            "action_taken": "none",
            "success": False,
            "message": ""
        }
        
        try:
            if issue_type == "status" and bot.status == "error":
                # Attempt to reset bot status to idle
                bot.update_status("idle")
                remediation_result["action_taken"] = "reset_status"
                remediation_result["success"] = True
                remediation_result["message"] = "Reset bot status from error to idle"
                
            elif issue_type == "stuck_task":
                # Find the stuck task
                stuck_task = bot.assigned_tasks.filter_by(status='in_progress').first()
                if stuck_task:
                    # Reset the task status and unassign from bot
                    stuck_task.update_status("pending")
                    stuck_task.assigned_to_bot_id = None
                    db.session.commit()
                    
                    # Reset bot status
                    bot.update_status("idle")
                    
                    remediation_result["action_taken"] = "reset_stuck_task"
                    remediation_result["success"] = True
                    remediation_result["message"] = f"Reset stuck task '{stuck_task.name}' and bot status"
                    
            elif issue_type == "idle_with_tasks":
                # Find appropriate pending tasks
                pending_tasks = Task.query.filter_by(
                    project_id=bot.project_id,
                    status='pending',
                    assigned_to_bot_id=None
                ).all()
                
                # Filter tasks that match this bot's capabilities
                matching_tasks = []
                for task in pending_tasks:
                    if self._task_matches_bot_capabilities(task, bot):
                        matching_tasks.append(task)
                
                if matching_tasks:
                    # Sort by priority (higher number = higher priority)
                    matching_tasks.sort(key=lambda x: x.priority, reverse=True)
                    
                    # Assign highest priority task
                    task = matching_tasks[0]
                    task.assigned_to_bot_id = bot.id
                    task.update_status("in_progress")
                    bot.update_status("working")
                    db.session.commit()
                    
                    remediation_result["action_taken"] = "assign_pending_task"
                    remediation_result["success"] = True
                    remediation_result["message"] = f"Assigned task '{task.name}' to idle bot"
                
            elif issue_type == "inactive":
                # For inactive bots, attempt to "ping" them by updating status
                old_status = bot.status
                # Temporarily set to another status and back
                bot.update_status("pinging")
                time.sleep(0.5)  # Small delay
                bot.update_status(old_status)
                
                remediation_result["action_taken"] = "ping_inactive_bot"
                remediation_result["success"] = True
                remediation_result["message"] = f"Pinged inactive bot with status {old_status}"
            
            logger.info(f"Remediation for bot {bot.id} ({issue_type}): {remediation_result['message']}")
            return remediation_result
            
        except Exception as e:
            logger.error(f"Error remediating issue {issue_type} for bot {bot.id}: {str(e)}")
            remediation_result["success"] = False
            remediation_result["message"] = f"Remediation failed: {str(e)}"
            return remediation_result
    
    def _task_matches_bot_capabilities(self, task: Task, bot: WorkerBot) -> bool:
        """
        Check if a task matches a bot's capabilities.
        
        Args:
            task: Task to check
            bot: Bot to check capabilities for
            
        Returns:
            True if the task matches the bot's capabilities
        """
        # Simple matching based on task type and bot type
        task_type_mapping = {
            "feature": ["developer"],
            "bug": ["developer", "tester"],
            "test": ["tester"],
            "documentation": ["developer", "architect"],
            "refactor": ["developer"],
            "design": ["architect"],
            "infrastructure": ["devops"],
            "deployment": ["devops"],
            "monitoring": ["devops"],
            "security": ["devops", "architect"]
        }
        
        # Get task types that match this bot type
        matching_task_types = [k for k, v in task_type_mapping.items() if bot.type in v]
        
        # Check if task type is in matching types
        return task.type in matching_task_types
    
    def _calculate_task_completion_rate(self, bot: WorkerBot) -> float:
        """
        Calculate task completion rate for a bot.
        
        Args:
            bot: WorkerBot instance
            
        Returns:
            Completion rate as percentage (0-100)
        """
        completed_tasks = bot.assigned_tasks.filter_by(status='completed').count()
        total_tasks = bot.assigned_tasks.count()
        
        if total_tasks == 0:
            return 100.0  # No tasks assigned, so technically 100% complete
            
        return (completed_tasks / total_tasks) * 100
    
    def _calculate_average_task_duration(self, bot: WorkerBot) -> float:
        """
        Calculate average task duration for completed tasks.
        
        Args:
            bot: WorkerBot instance
            
        Returns:
            Average duration in seconds or 0 if no completed tasks
        """
        completed_tasks = bot.assigned_tasks.filter_by(status='completed').all()
        
        if not completed_tasks:
            return 0.0
            
        durations = []
        for task in completed_tasks:
            if task.start_date and task.completion_date:
                duration = (task.completion_date - task.start_date).total_seconds()
                durations.append(duration)
        
        if not durations:
            return 0.0
            
        return sum(durations) / len(durations)
    
    def _calculate_error_rate(self, bot: WorkerBot) -> float:
        """
        Calculate error rate for a bot.
        
        Args:
            bot: WorkerBot instance
            
        Returns:
            Error rate as percentage (0-100)
        """
        # This would ideally be calculated from a bot_error_logs table
        # For now, we'll assume bots in error state have 100% error rate
        # and all others have 0% error rate
        return 100.0 if bot.status == "error" else 0.0
    
    def _update_bot_metrics(self, bot_id: int, metrics: Dict[str, float]) -> None:
        """
        Update metrics for a bot for trending analysis.
        
        Args:
            bot_id: Bot ID
            metrics: Dictionary of metrics
        """
        if bot_id not in self.bot_metrics:
            self.bot_metrics[bot_id] = []
        
        # Add timestamp to metrics
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Store metrics
        self.bot_metrics[bot_id].append(metrics)
        
        # Trim history if too long
        max_history = current_app.config.get('MAX_BOT_METRICS_HISTORY', 100)
        if len(self.bot_metrics[bot_id]) > max_history:
            self.bot_metrics[bot_id] = self.bot_metrics[bot_id][-max_history:]
    
    def _publish_metrics(self, summary: Dict[str, Any]) -> None:
        """
        Publish metrics to CloudWatch.
        
        Args:
            summary: Monitoring summary
        """
        if not self.cloudwatch:
            return
            
        try:
            timestamp = datetime.now()
            
            # Publish overall metrics
            self.cloudwatch.put_metric_data(
                Namespace='ProjectPilot/BotManager',
                MetricData=[
                    {
                        'MetricName': 'HealthyBots',
                        'Value': summary.get("healthy_bots", 0),
                        'Unit': 'Count',
                        'Timestamp': timestamp
                    },
                    {
                        'MetricName': 'WarningBots',
                        'Value': summary.get("warning_bots", 0),
                        'Unit': 'Count',
                        'Timestamp': timestamp
                    },
                    {
                        'MetricName': 'ErrorBots',
                        'Value': summary.get("error_bots", 0),
                        'Unit': 'Count',
                        'Timestamp': timestamp
                    }
                ]
            )
            
            # Publish bot-specific metrics
            for report in summary.get("bot_reports", []):
                bot_id = report.get("bot_id")
                bot_metrics = report.get("metrics", {})
                
                if bot_metrics:
                    self.cloudwatch.put_metric_data(
                        Namespace='ProjectPilot/BotManager',
                        MetricData=[
                            {
                                'MetricName': 'TaskCompletionRate',
                                'Value': bot_metrics.get("task_completion_rate", 0),
                                'Unit': 'Percent',
                                'Timestamp': timestamp,
                                'Dimensions': [
                                    {
                                        'Name': 'BotId',
                                        'Value': str(bot_id)
                                    }
                                ]
                            },
                            {
                                'MetricName': 'AverageTaskDuration',
                                'Value': bot_metrics.get("average_task_duration", 0),
                                'Unit': 'Seconds',
                                'Timestamp': timestamp,
                                'Dimensions': [
                                    {
                                        'Name': 'BotId',
                                        'Value': str(bot_id)
                                    }
                                ]
                            },
                            {
                                'MetricName': 'ErrorRate',
                                'Value': bot_metrics.get("error_rate", 0),
                                'Unit': 'Percent',
                                'Timestamp': timestamp,
                                'Dimensions': [
                                    {
                                        'Name': 'BotId',
                                        'Value': str(bot_id)
                                    }
                                ]
                            }
                        ]
                    )
            
            logger.debug(f"Published metrics to CloudWatch")
            
        except Exception as e:
            logger.error(f"Error publishing metrics to CloudWatch: {str(e)}")
    
    def get_bot_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of bot health across the system.
        
        Returns:
            Dictionary with health summary
        """
        try:
            # Get all worker bots
            bots = WorkerBot.query.all()
            
            if not bots:
                return {"status": "success", "message": "No worker bots found"}
            
            # Count bots by status
            status_counts = {}
            for bot in bots:
                if bot.status not in status_counts:
                    status_counts[bot.status] = 0
                status_counts[bot.status] += 1
            
            # Count bots by type
            type_counts = {}
            for bot in bots:
                if bot.type not in type_counts:
                    type_counts[bot.type] = 0
                type_counts[bot.type] += 1
            
            # Count bots by AI provider
            provider_counts = {}
            for bot in bots:
                if bot.ai_provider not in provider_counts:
                    provider_counts[bot.ai_provider] = 0
                provider_counts[bot.ai_provider] += 1
            
            # Calculate overall health
            healthy_bots = sum(1 for bot in bots if bot.status in ['idle', 'working'])
            health_percentage = (healthy_bots / len(bots) * 100) if bots else 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_bots": len(bots),
                "healthy_bots": healthy_bots,
                "health_percentage": health_percentage,
                "status_distribution": status_counts,
                "type_distribution": type_counts,
                "provider_distribution": provider_counts,
                "tasks_in_progress": sum(1 for bot in bots if bot.status == 'working'),
                "tasks_completed_today": self._get_tasks_completed_today(),
                "system_health": "good" if health_percentage >= 90 else 
                               "fair" if health_percentage >= 70 else "poor"
            }
            
        except Exception as e:
            logger.error(f"Error getting bot health summary: {str(e)}")
            return {"status": "error", "message": f"Failed to get health summary: {str(e)}"}
    
    def _get_tasks_completed_today(self) -> int:
        """
        Get count of tasks completed today.
        
        Returns:
            Number of tasks completed today
        """
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Count tasks completed today
        try:
            completed_today = Task.query.filter(
                Task.status == 'completed',
                Task.completion_date >= today_start
            ).count()
            
            return completed_today
            
        except Exception as e:
            logger.error(f"Error counting tasks completed today: {str(e)}")
            return 0
    
    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        AWS Lambda handler for scheduled monitoring and remediation.
        
        Args:
            event: Lambda event
            context: Lambda context
            
        Returns:
            Dictionary with execution results
        """
        try:
            action = event.get('action', 'monitor')
            
            if action == 'monitor':
                result = self.monitor_all_bots()
            elif action == 'remediate':
                result = self.auto_remediate_issues()
            elif action == 'health':
                result = self.get_bot_health_summary()
            else:
                result = {"status": "error", "message": f"Unknown action: {action}"}
            
            # Add execution metadata
            result["execution_timestamp"] = datetime.now().isoformat()
            result["execution_action"] = action
            
            return result
            
        except Exception as e:
            logger.error(f"Error in lambda_handler: {str(e)}")
            return {
                "status": "error",
                "message": f"Lambda execution failed: {str(e)}",
                "execution_timestamp": datetime.now().isoformat(),
                "execution_action": event.get('action', 'unknown')
            }