"""
ProjectPilot - AI-powered project management system
Task execution pipeline for managing the full lifecycle of tasks.
"""

import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import threading
import queue
from enum import Enum
from flask import current_app

from app import db
from app.models.project import Project
from app.models.task import Task, TaskDependency
from app.models.worker_bot import WorkerBot
from app.services.three_ai_workflow import ThreeAIWorkflow

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    REVIEW = 'review'
    COMPLETED = 'completed'
    BLOCKED = 'blocked'

class TaskExecutionStage(Enum):
    """Task execution stage enumeration."""
    PLANNING = 'planning'
    EXECUTION = 'execution'
    REVIEW = 'review'
    INTEGRATION = 'integration'
    VERIFICATION = 'verification'

class TaskExecutionResult:
    """Task execution result."""
    
    def __init__(self, 
                success: bool, 
                task_id: int, 
                message: str = "",
                output: Any = None,
                next_status: Optional[TaskStatus] = None,
                execution_time: float = 0.0):
        """
        Initialize task execution result.
        
        Args:
            success: Whether execution was successful
            task_id: Task ID
            message: Result message
            output: Execution output
            next_status: Status to update the task to
            execution_time: Time taken for execution in seconds
        """
        self.success = success
        self.task_id = task_id
        self.message = message
        self.output = output
        self.next_status = next_status
        self.execution_time = execution_time
        self.timestamp = datetime.utcnow()

class TaskExecutionPipeline:
    """
    Pipeline for executing tasks through their full lifecycle,
    from planning to completion.
    """
    
    def __init__(self, 
                three_ai_workflow: Optional[ThreeAIWorkflow] = None,
                max_concurrent_tasks: int = 5,
                polling_interval: int = 5):
        """
        Initialize the task execution pipeline.
        
        Args:
            three_ai_workflow: AI workflow for task execution
            max_concurrent_tasks: Maximum number of concurrently executing tasks
            polling_interval: Seconds between polling for new tasks
        """
        self.three_ai_workflow = three_ai_workflow or ThreeAIWorkflow()
        self.max_concurrent_tasks = max_concurrent_tasks
        self.polling_interval = polling_interval
        
        # Task queues and execution state
        self.pending_queue = queue.PriorityQueue()
        self.executing_tasks = {}  # task_id -> execution_info
        self.completed_tasks = {}  # task_id -> result
        self.task_locks = {}  # task_id -> lock
        
        # Pipeline stage handlers
        self.stage_handlers = {
            TaskExecutionStage.PLANNING: self._handle_planning_stage,
            TaskExecutionStage.EXECUTION: self._handle_execution_stage,
            TaskExecutionStage.REVIEW: self._handle_review_stage,
            TaskExecutionStage.INTEGRATION: self._handle_integration_stage,
            TaskExecutionStage.VERIFICATION: self._handle_verification_stage
        }
        
        # Pipeline state
        self.running = False
        self.worker_thread = None
        self.exit_event = threading.Event()
        
        # Task execution hooks
        self.pre_execution_hooks = []
        self.post_execution_hooks = []
        
        # Statistics
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'avg_execution_time': 0,
            'executions_by_bot_type': {}
        }
        
        logger.info("Task execution pipeline initialized")
    
    def start(self) -> None:
        """Start the task execution pipeline."""
        if self.running:
            logger.warning("Task execution pipeline already running")
            return
        
        self.running = True
        self.exit_event.clear()
        self.worker_thread = threading.Thread(target=self._pipeline_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        logger.info("Task execution pipeline started")
    
    def stop(self) -> None:
        """Stop the task execution pipeline."""
        if not self.running:
            logger.warning("Task execution pipeline not running")
            return
        
        self.running = False
        self.exit_event.set()
        
        if self.worker_thread:
            self.worker_thread.join(timeout=30)
            if self.worker_thread.is_alive():
                logger.warning("Task execution pipeline worker thread did not terminate cleanly")
            
        logger.info("Task execution pipeline stopped")
    
    def add_task(self, task_id: int, priority: int = 2) -> bool:
        """
        Add a task to the execution queue.
        
        Args:
            task_id: Task ID
            priority: Task priority (lower number = higher priority)
            
        Returns:
            True if task was added successfully
        """
        try:
            # Check if task exists
            task = Task.query.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return False
            
            # Check if task is already in queue or executing
            if task_id in self.executing_tasks or task_id in self.task_locks:
                logger.warning(f"Task {task_id} is already queued or executing")
                return False
            
            # Check if task is in a valid state
            if task.status != TaskStatus.PENDING.value and task.status != TaskStatus.BLOCKED.value:
                logger.warning(f"Task {task_id} is not in a valid state for execution")
                return False
            
            # Create lock for this task
            self.task_locks[task_id] = threading.RLock()
            
            # Add to priority queue (negative priority so lower numbers are higher priority)
            self.pending_queue.put((-priority, task_id))
            
            logger.info(f"Task {task_id} added to execution queue with priority {priority}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding task {task_id} to queue: {str(e)}")
            return False
    
    def execute_task(self, task_id: int) -> TaskExecutionResult:
        """
        Execute a task immediately (blocking).
        
        Args:
            task_id: Task ID
            
        Returns:
            Task execution result
        """
        try:
            # Check if task exists
            task = Task.query.get(task_id)
            if not task:
                return TaskExecutionResult(
                    success=False,
                    task_id=task_id,
                    message=f"Task {task_id} not found",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Check if dependencies are satisfied
            if not self._check_dependencies_satisfied(task):
                return TaskExecutionResult(
                    success=False,
                    task_id=task_id,
                    message=f"Task {task_id} has unsatisfied dependencies",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Execute each stage of the pipeline
            start_time = time.time()
            
            # Run pre-execution hooks
            for hook in self.pre_execution_hooks:
                hook(task)
            
            result = None
            current_stage = TaskExecutionStage.PLANNING
            
            # Process through all stages
            while current_stage:
                handler = self.stage_handlers[current_stage]
                stage_result = handler(task)
                
                if not stage_result.success:
                    # Stage failed, return result
                    result = stage_result
                    break
                
                # Determine next stage
                current_stage = self._get_next_stage(current_stage, stage_result)
                
                if not current_stage:
                    # Pipeline completed successfully
                    result = stage_result
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update result with execution time
            if result:
                result.execution_time = execution_time
            else:
                # This should never happen, but just in case
                result = TaskExecutionResult(
                    success=False,
                    task_id=task_id,
                    message="Pipeline execution failed with no result",
                    execution_time=execution_time,
                    next_status=TaskStatus.BLOCKED
                )
            
            # Update task status if needed
            if result.next_status:
                task.update_status(result.next_status.value)
                db.session.commit()
            
            # Run post-execution hooks
            for hook in self.post_execution_hooks:
                hook(task, result)
            
            # Update statistics
            self._update_statistics(task, result)
            
            # Store result in completed tasks
            self.completed_tasks[task_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {str(e)}")
            return TaskExecutionResult(
                success=False,
                task_id=task_id,
                message=f"Execution failed: {str(e)}",
                next_status=TaskStatus.BLOCKED,
                execution_time=0.0
            )
    
    def get_execution_status(self, task_id: int) -> Dict[str, Any]:
        """
        Get status of a task's execution.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dictionary with execution status
        """
        # Check if task is in queue
        in_queue = False
        queue_position = 0
        
        # Copy queue to check position without modifying it
        queue_copy = list(self.pending_queue.queue)
        for i, (_, queued_task_id) in enumerate(queue_copy):
            if queued_task_id == task_id:
                in_queue = True
                queue_position = i + 1
                break
        
        # Check if task is currently executing
        is_executing = task_id in self.executing_tasks
        execution_info = self.executing_tasks.get(task_id, {})
        
        # Check if task has completed
        is_completed = task_id in self.completed_tasks
        result = self.completed_tasks.get(task_id)
        
        # Get current task from database
        task = Task.query.get(task_id)
        
        status = {
            "task_id": task_id,
            "in_queue": in_queue,
            "queue_position": queue_position,
            "is_executing": is_executing,
            "current_stage": execution_info.get("current_stage") if is_executing else None,
            "execution_start_time": execution_info.get("start_time") if is_executing else None,
            "is_completed": is_completed,
            "result": {
                "success": result.success,
                "message": result.message,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat()
            } if result else None,
            "task_status": task.status if task else None,
            "assigned_bot": task.assigned_bot.name if task and task.assigned_bot else None
        }
        
        return status
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get status of the entire pipeline.
        
        Returns:
            Dictionary with pipeline status
        """
        status = {
            "running": self.running,
            "queue_size": self.pending_queue.qsize(),
            "executing_tasks": len(self.executing_tasks),
            "completed_tasks": len(self.completed_tasks),
            "statistics": self.stats,
            "executing_task_ids": list(self.executing_tasks.keys())
        }
        
        return status
    
    def register_pre_execution_hook(self, hook: Callable[[Task], None]) -> None:
        """
        Register a pre-execution hook.
        
        Args:
            hook: Function to call before task execution
        """
        self.pre_execution_hooks.append(hook)
    
    def register_post_execution_hook(self, hook: Callable[[Task, TaskExecutionResult], None]) -> None:
        """
        Register a post-execution hook.
        
        Args:
            hook: Function to call after task execution
        """
        self.post_execution_hooks.append(hook)
    
    def _pipeline_worker(self) -> None:
        """Worker thread for processing tasks in the pipeline."""
        logger.info("Pipeline worker thread started")
        
        while self.running and not self.exit_event.is_set():
            try:
                # Check if we can process more tasks
                if len(self.executing_tasks) >= self.max_concurrent_tasks:
                    # Wait and check again
                    time.sleep(self.polling_interval)
                    continue
                
                # Get next task from queue (non-blocking)
                try:
                    _, task_id = self.pending_queue.get(block=False)
                except queue.Empty:
                    # No tasks in queue, wait and try again
                    time.sleep(self.polling_interval)
                    continue
                
                # Start task execution in a separate thread
                task_thread = threading.Thread(
                    target=self._execute_task_thread,
                    args=(task_id,)
                )
                task_thread.daemon = True
                task_thread.start()
                
                # Brief pause between starting tasks
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in pipeline worker: {str(e)}")
                time.sleep(self.polling_interval)
        
        logger.info("Pipeline worker thread stopped")
    
    def _execute_task_thread(self, task_id: int) -> None:
        """
        Execute a task in a separate thread.
        
        Args:
            task_id: Task ID
        """
        try:
            # Mark task as executing
            task = Task.query.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found for execution")
                if task_id in self.task_locks:
                    del self.task_locks[task_id]
                return
            
            # Update task status to in_progress
            task.update_status(TaskStatus.IN_PROGRESS.value)
            db.session.commit()
            
            # Mark as executing
            self.executing_tasks[task_id] = {
                "start_time": datetime.utcnow().isoformat(),
                "current_stage": TaskExecutionStage.PLANNING.value
            }
            
            # Execute task
            result = self.execute_task(task_id)
            
            # Task completed (successfully or not)
            if task_id in self.executing_tasks:
                del self.executing_tasks[task_id]
            
            if task_id in self.task_locks:
                del self.task_locks[task_id]
            
            # Mark queue item as done
            self.pending_queue.task_done()
            
            logger.info(f"Task {task_id} execution completed with result: {result.success}")
            
        except Exception as e:
            logger.error(f"Error executing task {task_id} in thread: {str(e)}")
            
            # Clean up
            if task_id in self.executing_tasks:
                del self.executing_tasks[task_id]
            
            if task_id in self.task_locks:
                del self.task_locks[task_id]
            
            # Mark queue item as done
            self.pending_queue.task_done()
    
    def _handle_planning_stage(self, task: Task) -> TaskExecutionResult:
        """
        Handle the planning stage of task execution.
        
        Args:
            task: Task to process
            
        Returns:
            Task execution result
        """
        logger.info(f"Planning stage for task {task.id}: {task.name}")
        
        try:
            # Update execution tracking
            if task.id in self.executing_tasks:
                self.executing_tasks[task.id]["current_stage"] = TaskExecutionStage.PLANNING.value
            
            # Check if task has an assigned bot
            if not task.assigned_to_bot_id:
                # Assign a suitable bot
                assigned = self._assign_bot_to_task(task)
                if not assigned:
                    return TaskExecutionResult(
                        success=False,
                        task_id=task.id,
                        message="No suitable bot available for task",
                        next_status=TaskStatus.BLOCKED
                    )
            
            # Get the assigned bot
            bot = WorkerBot.query.get(task.assigned_to_bot_id)
            if not bot:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message="Assigned bot not found",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Create execution plan using the ThreeAIWorkflow
            project = task.project
            
            planning_result = self.three_ai_workflow.worker_bot_execute_task(
                bot_type=bot.type,
                task_description=f"Create an execution plan for task: {task.name}\n{task.description}",
                project_context=f"Project: {project.name}\n{project.description}",
                preferred_provider=bot.ai_provider
            )
            
            if "error" in planning_result:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message=f"Planning failed: {planning_result['error']}",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Extract plan from result
            execution_plan = planning_result.get("result", {})
            
            return TaskExecutionResult(
                success=True,
                task_id=task.id,
                message="Planning completed successfully",
                output={"execution_plan": execution_plan, "bot_id": bot.id},
                next_status=TaskStatus.IN_PROGRESS
            )
            
        except Exception as e:
            logger.error(f"Error in planning stage for task {task.id}: {str(e)}")
            return TaskExecutionResult(
                success=False,
                task_id=task.id,
                message=f"Planning error: {str(e)}",
                next_status=TaskStatus.BLOCKED
            )
    
    def _handle_execution_stage(self, task: Task) -> TaskExecutionResult:
        """
        Handle the execution stage of task execution.
        
        Args:
            task: Task to process
            
        Returns:
            Task execution result
        """
        logger.info(f"Execution stage for task {task.id}: {task.name}")
        
        try:
            # Update execution tracking
            if task.id in self.executing_tasks:
                self.executing_tasks[task.id]["current_stage"] = TaskExecutionStage.EXECUTION.value
            
            # Get the assigned bot
            bot = WorkerBot.query.get(task.assigned_to_bot_id)
            if not bot:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message="Assigned bot not found",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Get the project
            project = task.project
            
            # Execute the task using the ThreeAIWorkflow
            execution_result = self.three_ai_workflow.worker_bot_execute_task(
                bot_type=bot.type,
                task_description=f"Execute task: {task.name}\n{task.description}",
                project_context=f"Project: {project.name}\n{project.description}",
                preferred_provider=bot.ai_provider
            )
            
            if "error" in execution_result:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message=f"Execution failed: {execution_result['error']}",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Extract result
            task_output = execution_result.get("result", {})
            
            return TaskExecutionResult(
                success=True,
                task_id=task.id,
                message="Execution completed successfully",
                output={"task_output": task_output, "bot_id": bot.id},
                next_status=TaskStatus.REVIEW
            )
            
        except Exception as e:
            logger.error(f"Error in execution stage for task {task.id}: {str(e)}")
            return TaskExecutionResult(
                success=False,
                task_id=task.id,
                message=f"Execution error: {str(e)}",
                next_status=TaskStatus.BLOCKED
            )
    
    def _handle_review_stage(self, task: Task) -> TaskExecutionResult:
        """
        Handle the review stage of task execution.
        
        Args:
            task: Task to process
            
        Returns:
            Task execution result
        """
        logger.info(f"Review stage for task {task.id}: {task.name}")
        
        try:
            # Update execution tracking
            if task.id in self.executing_tasks:
                self.executing_tasks[task.id]["current_stage"] = TaskExecutionStage.REVIEW.value
            
            # Get the previous stage output
            prev_result = self.completed_tasks.get(task.id)
            if not prev_result or not prev_result.output:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message="No output from previous stage",
                    next_status=TaskStatus.BLOCKED
                )
            
            task_output = prev_result.output.get("task_output", {})
            
            # Get a different bot for review (preferably)
            review_bot = self._get_review_bot(task)
            if not review_bot:
                # Fall back to same bot
                review_bot = WorkerBot.query.get(task.assigned_to_bot_id)
            
            if not review_bot:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message="No bot available for review",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Get the project
            project = task.project
            
            # Review the task output using the ThreeAIWorkflow
            review_result = self.three_ai_workflow.worker_bot_execute_task(
                bot_type=review_bot.type,
                task_description=f"Review task output for: {task.name}\n{task.description}\nOutput: {json.dumps(task_output, indent=2)}",
                project_context=f"Project: {project.name}\n{project.description}",
                preferred_provider=review_bot.ai_provider
            )
            
            if "error" in review_result:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message=f"Review failed: {review_result['error']}",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Extract review result
            review_output = review_result.get("result", {})
            
            # Determine if review passed
            review_passed = self._check_review_passed(review_output)
            
            if not review_passed:
                # Review failed, go back to execution stage
                return TaskExecutionResult(
                    success=True,
                    task_id=task.id,
                    message="Review failed, retrying execution",
                    output={"review_output": review_output, "bot_id": review_bot.id},
                    next_status=TaskStatus.IN_PROGRESS
                )
            
            # Review passed, proceed to integration
            return TaskExecutionResult(
                success=True,
                task_id=task.id,
                message="Review passed",
                output={
                    "review_output": review_output, 
                    "bot_id": review_bot.id,
                    "task_output": task_output
                },
                next_status=TaskStatus.REVIEW
            )
            
        except Exception as e:
            logger.error(f"Error in review stage for task {task.id}: {str(e)}")
            return TaskExecutionResult(
                success=False,
                task_id=task.id,
                message=f"Review error: {str(e)}",
                next_status=TaskStatus.BLOCKED
            )
    
    def _handle_integration_stage(self, task: Task) -> TaskExecutionResult:
        """
        Handle the integration stage of task execution.
        
        Args:
            task: Task to process
            
        Returns:
            Task execution result
        """
        logger.info(f"Integration stage for task {task.id}: {task.name}")
        
        try:
            # Update execution tracking
            if task.id in self.executing_tasks:
                self.executing_tasks[task.id]["current_stage"] = TaskExecutionStage.INTEGRATION.value
            
            # Get the previous stage output
            prev_result = self.completed_tasks.get(task.id)
            if not prev_result or not prev_result.output:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message="No output from previous stage",
                    next_status=TaskStatus.BLOCKED
                )
            
            task_output = prev_result.output.get("task_output", {})
            
            # For integration, we'll use the original assigned bot
            bot = WorkerBot.query.get(task.assigned_to_bot_id)
            if not bot:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message="Assigned bot not found",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Get the project
            project = task.project
            
            # Integration is highly dependent on task type
            # For now, we'll treat it as a pass-through stage
            
            # In a real implementation, this would include:
            # - Committing code to repositories
            # - Updating documentation
            # - Merging changes
            # - Triggering builds or deployments
            
            integration_result = {
                "status": "success",
                "message": "Integration completed (simulated)",
                "details": f"Task {task.id} ({task.name}) integration simulated successfully"
            }
            
            return TaskExecutionResult(
                success=True,
                task_id=task.id,
                message="Integration completed successfully",
                output={
                    "integration_result": integration_result,
                    "task_output": task_output,
                    "bot_id": bot.id
                },
                next_status=TaskStatus.REVIEW
            )
            
        except Exception as e:
            logger.error(f"Error in integration stage for task {task.id}: {str(e)}")
            return TaskExecutionResult(
                success=False,
                task_id=task.id,
                message=f"Integration error: {str(e)}",
                next_status=TaskStatus.BLOCKED
            )
    
    def _handle_verification_stage(self, task: Task) -> TaskExecutionResult:
        """
        Handle the verification stage of task execution.
        
        Args:
            task: Task to process
            
        Returns:
            Task execution result
        """
        logger.info(f"Verification stage for task {task.id}: {task.name}")
        
        try:
            # Update execution tracking
            if task.id in self.executing_tasks:
                self.executing_tasks[task.id]["current_stage"] = TaskExecutionStage.VERIFICATION.value
            
            # Get the previous stage output
            prev_result = self.completed_tasks.get(task.id)
            if not prev_result or not prev_result.output:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message="No output from previous stage",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Get a tester bot if available
            verification_bot = self._get_verification_bot(task)
            if not verification_bot:
                # Fall back to original bot
                verification_bot = WorkerBot.query.get(task.assigned_to_bot_id)
            
            if not verification_bot:
                return TaskExecutionResult(
                    success=False,
                    task_id=task.id,
                    message="No bot available for verification",
                    next_status=TaskStatus.BLOCKED
                )
            
            # Get the project
            project = task.project
            
            # For verification, we'd normally run tests
            # For now, we'll simulate verification success
            
            # In a real implementation, this would include:
            # - Running automated tests
            # - Checking code quality
            # - Validating against requirements
            
            verification_result = {
                "status": "success",
                "message": "Verification completed (simulated)",
                "details": f"Task {task.id} ({task.name}) verification simulated successfully"
            }
            
            # Mark task as completed
            return TaskExecutionResult(
                success=True,
                task_id=task.id,
                message="Verification completed successfully, task complete",
                output={"verification_result": verification_result, "bot_id": verification_bot.id},
                next_status=TaskStatus.COMPLETED
            )
            
        except Exception as e:
            logger.error(f"Error in verification stage for task {task.id}: {str(e)}")
            return TaskExecutionResult(
                success=False,
                task_id=task.id,
                message=f"Verification error: {str(e)}",
                next_status=TaskStatus.BLOCKED
            )
    
    def _get_next_stage(self, 
                       current_stage: TaskExecutionStage, 
                       result: TaskExecutionResult) -> Optional[TaskExecutionStage]:
        """
        Determine the next pipeline stage based on current stage and result.
        
        Args:
            current_stage: Current execution stage
            result: Current stage execution result
            
        Returns:
            Next stage or None if pipeline is complete
        """
        if not result.success:
            # Failed stages don't progress
            return None
        
        # Standard stage progression
        stage_progression = {
            TaskExecutionStage.PLANNING: TaskExecutionStage.EXECUTION,
            TaskExecutionStage.EXECUTION: TaskExecutionStage.REVIEW,
            TaskExecutionStage.REVIEW: TaskExecutionStage.INTEGRATION,
            TaskExecutionStage.INTEGRATION: TaskExecutionStage.VERIFICATION,
            TaskExecutionStage.VERIFICATION: None  # End of pipeline
        }
        
        # Check for special cases
        if current_stage == TaskExecutionStage.REVIEW:
            # Check if review failed and needs to go back to execution
            review_output = result.output.get("review_output", {})
            if isinstance(review_output, dict) and review_output.get("status") == "failed":
                return TaskExecutionStage.EXECUTION
        
        # Return next stage from standard progression
        return stage_progression[current_stage]
    
    def _assign_bot_to_task(self, task: Task) -> bool:
        """
        Assign a suitable bot to a task.
        
        Args:
            task: Task to assign a bot to
            
        Returns:
            True if assignment was successful
        """
        # Get project
        project = task.project
        
        # Get available bots for this project
        available_bots = WorkerBot.query.filter_by(
            project_id=project.id,
            status='idle'
        ).all()
        
        if not available_bots:
            logger.warning(f"No available bots for task {task.id}")
            return False
        
        # Match bot to task based on type
        # This could be more sophisticated in a real implementation
        task_type_to_bot_type = {
            "feature": "developer",
            "bug": "developer",
            "documentation": "developer",
            "design": "architect",
            "infrastructure": "devops",
            "test": "tester"
        }
        
        preferred_bot_type = task_type_to_bot_type.get(task.type, "developer")
        
        # Try to find a bot of the preferred type
        matched_bot = next((bot for bot in available_bots if bot.type == preferred_bot_type), None)
        
        # If no match, take any available bot
        bot = matched_bot or available_bots[0]
        
        # Assign bot to task
        task.assigned_to_bot_id = bot.id
        bot.update_status('working')
        db.session.commit()
        
        logger.info(f"Assigned bot {bot.id} ({bot.name}) to task {task.id}")
        return True
    
    def _get_review_bot(self, task: Task) -> Optional[WorkerBot]:
        """
        Get a bot for reviewing a task, preferably different from the execution bot.
        
        Args:
            task: Task to review
            
        Returns:
            Worker bot for review or None if none available
        """
        # Get project
        project = task.project
        
        # Get available bots for this project
        available_bots = WorkerBot.query.filter_by(
            project_id=project.id,
            status='idle'
        ).all()
        
        if not available_bots:
            return None
        
        # Prefer a bot with the same type but different from the execution bot
        preferred_bots = [bot for bot in available_bots 
                         if bot.id != task.assigned_to_bot_id and bot.type == "tester"]
        
        # If no tester, try a bot of any type different from the execution bot
        if not preferred_bots:
            preferred_bots = [bot for bot in available_bots if bot.id != task.assigned_to_bot_id]
        
        # Return the preferred bot or None if none available
        return preferred_bots[0] if preferred_bots else None
    
    def _get_verification_bot(self, task: Task) -> Optional[WorkerBot]:
        """
        Get a bot for verifying a task, preferably a tester bot.
        
        Args:
            task: Task to verify
            
        Returns:
            Worker bot for verification or None if none available
        """
        # Get project
        project = task.project
        
        # Get available tester bots for this project
        tester_bots = WorkerBot.query.filter_by(
            project_id=project.id,
            status='idle',
            type='tester'
        ).all()
        
        if tester_bots:
            return tester_bots[0]
        
        # No tester bots available, try any available bot
        available_bots = WorkerBot.query.filter_by(
            project_id=project.id,
            status='idle'
        ).all()
        
        return available_bots[0] if available_bots else None
    
    def _check_review_passed(self, review_output: Any) -> bool:
        """
        Check if a review result indicates success.
        
        Args:
            review_output: Review output from AI
            
        Returns:
            True if review passed
        """
        # If review_output is a dict with a status field
        if isinstance(review_output, dict) and "status" in review_output:
            return review_output["status"].lower() == "success" or review_output["status"].lower() == "passed"
        
        # If review_output is a dict with a passed field
        if isinstance(review_output, dict) and "passed" in review_output:
            return bool(review_output["passed"])
        
        # If review_output is a string
        if isinstance(review_output, str):
            return "pass" in review_output.lower() or "success" in review_output.lower()
        
        # Default to passed (this could be customized based on your requirements)
        return True
    
    def _check_dependencies_satisfied(self, task: Task) -> bool:
        """
        Check if all dependencies for a task are satisfied.
        
        Args:
            task: Task to check
            
        Returns:
            True if all dependencies are satisfied
        """
        # Get dependencies for this task
        dependencies = task.dependencies.all()
        
        # Check if any dependencies are not completed
        for dependency in dependencies:
            dependent_task = Task.query.get(dependency.depends_on_id)
            if not dependent_task or dependent_task.status != TaskStatus.COMPLETED.value:
                logger.info(f"Task {task.id} has unsatisfied dependency: {dependency.depends_on_id}")
                return False
        
        return True
    
    def _update_statistics(self, task: Task, result: TaskExecutionResult) -> None:
        """
        Update execution statistics.
        
        Args:
            task: Task that was executed
            result: Execution result
        """
        # Get the assigned bot
        bot = WorkerBot.query.get(task.assigned_to_bot_id)
        bot_type = bot.type if bot else "unknown"
        
        # Update overall statistics
        self.stats['tasks_processed'] += 1
        
        if result.success:
            self.stats['tasks_successful'] += 1
        else:
            self.stats['tasks_failed'] += 1
        
        # Update average execution time (rolling average)
        current_avg = self.stats['avg_execution_time']
        current_count = self.stats['tasks_processed']
        self.stats['avg_execution_time'] = (current_avg * (current_count - 1) + result.execution_time) / current_count
        
        # Update bot type statistics
        if bot_type not in self.stats['executions_by_bot_type']:
            self.stats['executions_by_bot_type'][bot_type] = {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'avg_execution_time': 0
            }
        
        bot_stats = self.stats['executions_by_bot_type'][bot_type]
        bot_stats['total'] += 1
        
        if result.success:
            bot_stats['successful'] += 1
        else:
            bot_stats['failed'] += 1
        
        # Update bot type average execution time (rolling average)
        current_bot_avg = bot_stats['avg_execution_time']
        current_bot_count = bot_stats['total']
        bot_stats['avg_execution_time'] = (current_bot_avg * (current_bot_count - 1) + result.execution_time) / current_bot_count

# Initialize the global task execution pipeline
task_pipeline = TaskExecutionPipeline()

# Start the pipeline automatically if not in testing mode
if not os.environ.get('TESTING'):
    task_pipeline.start()