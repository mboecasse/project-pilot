"""
Test file for the BotManagerLambda service.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta

# Import the module to test
from app.services.bot_manager_lambda import BotManagerLambda

class TestBotManagerLambda:
    """Tests for the BotManagerLambda class."""
    
    @pytest.fixture
    def mock_boto3(self):
        """Mock boto3 for AWS services."""
        with patch('app.services.bot_manager_lambda.boto3') as mock_boto3:
            # Set up mock Session
            mock_session = MagicMock()
            mock_boto3.Session.return_value = mock_session
            
            # Set up mock CloudWatch client
            mock_cloudwatch = MagicMock()
            mock_session.client.return_value = mock_cloudwatch
            
            yield {
                "boto3": mock_boto3,
                "session": mock_session,
                "cloudwatch": mock_cloudwatch
            }
    
    @pytest.fixture
    def mock_flask_app(self):
        """Mock Flask app context."""
        mock_app = MagicMock()
        mock_app.config = {
            'MAX_BOT_METRICS_HISTORY': 50
        }
        with patch('app.services.bot_manager_lambda.current_app', mock_app):
            yield mock_app
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        with patch('app.services.bot_manager_lambda.db') as mock_db:
            yield mock_db
    
    @pytest.fixture
    def mock_worker_bot(self):
        """Create a mock WorkerBot class."""
        with patch('app.services.bot_manager_lambda.WorkerBot') as mock_worker_bot_class:
            # Create mock query
            mock_query = MagicMock()
            mock_worker_bot_class.query = mock_query
            
            # Create mock bot instances
            mock_bot1 = MagicMock()
            mock_bot1.id = 1
            mock_bot1.name = "Bot 1"
            mock_bot1.type = "developer"
            mock_bot1.status = "idle"
            mock_bot1.project_id = 1
            mock_bot1.capabilities = ["coding", "testing"]
            mock_bot1.ai_provider = "openai"
            mock_bot1.ai_model = "gpt-4"
            mock_bot1.last_active = datetime.now()
            
            mock_bot2 = MagicMock()
            mock_bot2.id = 2
            mock_bot2.name = "Bot 2"
            mock_bot2.type = "tester"
            mock_bot2.status = "error"
            mock_bot2.project_id = 1
            mock_bot2.capabilities = ["testing"]
            mock_bot2.ai_provider = "anthropic"
            mock_bot2.ai_model = "claude-3-opus-20240229"
            mock_bot2.last_active = datetime.now() - timedelta(hours=1)
            
            # Mock bots list
            mock_bots = [mock_bot1, mock_bot2]
            mock_query.all.return_value = mock_bots
            
            # Set up assigned_tasks for bots
            mock_tasks_query1 = MagicMock()
            mock_tasks_query1.count.return_value = 5
            mock_tasks_query1.filter_by.return_value.count.return_value = 3
            mock_bot1.assigned_tasks = mock_tasks_query1
            
            mock_tasks_query2 = MagicMock()
            mock_tasks_query2.count.return_value = 2
            mock_tasks_query2.filter_by.return_value.count.return_value = 0
            mock_bot2.assigned_tasks = mock_tasks_query2
            
            yield {
                "class": mock_worker_bot_class,
                "query": mock_query,
                "bots": mock_bots,
                "bot1": mock_bot1,
                "bot2": mock_bot2
            }
    
    @pytest.fixture
    def mock_task(self):
        """Create a mock Task class."""
        with patch('app.services.bot_manager_lambda.Task') as mock_task_class:
            # Create mock query
            mock_query = MagicMock()
            mock_task_class.query = mock_query
            
            # Create mock task instances
            mock_task1 = MagicMock()
            mock_task1.id = 1
            mock_task1.name = "Task 1"
            mock_task1.type = "feature"
            mock_task1.status = "in_progress"
            mock_task1.project_id = 1
            mock_task1.assigned_to_bot_id = 1
            mock_task1.priority = 1
            mock_task1.start_date = datetime.now() - timedelta(hours=2)
            mock_task1.completion_date = None
            
            mock_task2 = MagicMock()
            mock_task2.id = 2
            mock_task2.name = "Task 2"
            mock_task2.type = "bug"
            mock_task2.status = "pending"
            mock_task2.project_id = 1
            mock_task2.assigned_to_bot_id = None
            mock_task2.priority = 2
            mock_task2.start_date = None
            mock_task2.completion_date = None
            
            # Mock tasks list
            mock_tasks = [mock_task1, mock_task2]
            
            # Set up filter_by returns
            mock_query.filter_by.return_value = MagicMock()
            mock_query.filter_by.return_value.all.return_value = [mock_task2]
            
            # Set up filter returns
            mock_query.filter.return_value.count.return_value = 3  # Tasks completed today
            
            yield {
                "class": mock_task_class,
                "query": mock_query,
                "tasks": mock_tasks,
                "task1": mock_task1,
                "task2": mock_task2
            }
    
    @pytest.fixture
    def bot_manager(self, mock_boto3, mock_flask_app):
        """Create a BotManagerLambda instance."""
        manager = BotManagerLambda(
            region="us-west-2",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret"
        )
        return manager
    
    def test_init(self, mock_boto3):
        """Test initialization of BotManagerLambda."""
        manager = BotManagerLambda(
            region="us-west-2",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret"
        )
        
        assert manager.region == "us-west-2"
        assert manager.aws_access_key_id == "test-key"
        assert manager.aws_secret_access_key == "test-secret"
        assert manager.cloudwatch == mock_boto3["cloudwatch"]
        assert manager.bot_metrics == {}
        assert isinstance(manager.alert_thresholds, dict)
    
    def test_init_without_aws_credentials(self):
        """Test initialization without AWS credentials."""
        with patch('app.services.bot_manager_lambda.os') as mock_os, \
             patch('app.services.bot_manager_lambda.current_app') as mock_app:
            
            mock_os.environ.get.return_value = None
            mock_app.config.get.return_value = "eu-west-2"
            
            manager = BotManagerLambda()
            
            assert manager.region == "eu-west-2"
            assert manager.aws_access_key_id is None
            assert manager.aws_secret_access_key is None
            assert manager.cloudwatch is None
    
    def test_monitor_all_bots(self, bot_manager, mock_worker_bot):
        """Test monitoring all bots."""
        # Mock monitor_bot to return canned responses
        with patch.object(bot_manager, 'monitor_bot') as mock_monitor_bot:
            mock_monitor_bot.side_effect = [
                {"bot_id": 1, "name": "Bot 1", "status": "healthy", "issues": []},
                {"bot_id": 2, "name": "Bot 2", "status": "error", "issues": [{"type": "status", "severity": "error", "message": "Bot is in error state"}]}
            ]
            
            # Mock _publish_metrics
            with patch.object(bot_manager, '_publish_metrics') as mock_publish_metrics:
                result = bot_manager.monitor_all_bots()
                
                # Check that monitor_bot was called for each bot
                assert mock_monitor_bot.call_count == 2
                mock_monitor_bot.assert_any_call(mock_worker_bot["bot1"])
                mock_monitor_bot.assert_any_call(mock_worker_bot["bot2"])
                
                # Check that metrics were published
                mock_publish_metrics.assert_called_once()
                
                # Check the result
                assert "timestamp" in result
                assert result["total_bots"] == 2
                assert result["healthy_bots"] == 1
                assert result["error_bots"] == 1
                assert result["warning_bots"] == 0
                assert len(result["bots_requiring_attention"]) == 1
                assert result["bots_requiring_attention"][0]["bot_id"] == 2
    
    def test_monitor_all_bots_no_bots(self, bot_manager, mock_worker_bot):
        """Test monitoring when no bots are found."""
        # Set up mock to return empty list
        mock_worker_bot["query"].all.return_value = []
        
        result = bot_manager.monitor_all_bots()
        
        assert result["status"] == "success"
        assert "No worker bots found" in result["message"]
    
    def test_monitor_bot(self, bot_manager, mock_worker_bot, mock_task):
        """Test monitoring a specific bot."""
        # Set up mock for tasks
        mock_task_query = MagicMock()
        mock_task_query.filter_by.return_value.first.return_value = mock_task["task1"]
        mock_worker_bot["bot1"].assigned_tasks = mock_task_query
        
        # Mock helper methods
        with patch.object(bot_manager, '_calculate_task_completion_rate', return_value=60.0), \
             patch.object(bot_manager, '_calculate_average_task_duration', return_value=3600.0), \
             patch.object(bot_manager, '_calculate_error_rate', return_value=0.0), \
             patch.object(bot_manager, '_update_bot_metrics') as mock_update_metrics:
            
            # Set bot status to working
            mock_worker_bot["bot1"].status = "working"
            
            result = bot_manager.monitor_bot(mock_worker_bot["bot1"])
            
            # Check that bot metrics were updated
            mock_update_metrics.assert_called_once()
            
            # Check the result
            assert result["bot_id"] == 1
            assert result["name"] == "Bot 1"
            assert result["status"] == "healthy"  # No issues detected
            assert "metrics" in result
            assert result["metrics"]["task_completion_rate"] == 60.0
            assert result["metrics"]["average_task_duration"] == 3600.0
            assert result["metrics"]["error_rate"] == 0.0
    
    def test_monitor_bot_with_issues(self, bot_manager, mock_worker_bot, mock_task):
        """Test monitoring a bot with issues."""
        # Set up mock for tasks
        mock_task_query = MagicMock()
        old_task = mock_task["task1"]
        old_task.start_date = datetime.now() - timedelta(hours=48)  # Stuck task
        mock_task_query.filter_by.return_value.first.return_value = old_task
        mock_worker_bot["bot1"].assigned_tasks = mock_task_query
        
        # Mock helper methods
        with patch.object(bot_manager, '_calculate_task_completion_rate', return_value=60.0), \
             patch.object(bot_manager, '_calculate_average_task_duration', return_value=3600.0), \
             patch.object(bot_manager, '_calculate_error_rate', return_value=0.0), \
             patch.object(bot_manager, '_update_bot_metrics') as mock_update_metrics:
            
            # Set bot status to working
            mock_worker_bot["bot1"].status = "working"
            
            result = bot_manager.monitor_bot(mock_worker_bot["bot1"])
            
            # Check that bot metrics were updated
            mock_update_metrics.assert_called_once()
            
            # Check the result
            assert result["bot_id"] == 1
            assert result["name"] == "Bot 1"
            assert result["status"] == "error"  # Stuck task is an error
            assert len(result["issues"]) == 1
            assert result["issues"][0]["type"] == "stuck_task"
            assert result["issues"][0]["severity"] == "error"
    
    def test_auto_remediate_issues(self, bot_manager):
        """Test automatic remediation of issues."""
        # Mock monitor_all_bots
        mock_report = {
            "bots_requiring_attention": [
                {"bot_id": 1, "name": "Bot 1", "status": "warning", "issues": [{"type": "idle_with_tasks"}]},
                {"bot_id": 2, "name": "Bot 2", "status": "error", "issues": [{"type": "status"}]}
            ]
        }
        
        with patch.object(bot_manager, 'monitor_all_bots', return_value=mock_report), \
             patch.object(bot_manager, '_remediate_issue') as mock_remediate:
            
            # Set up remediation results
            mock_remediate.side_effect = [
                {"bot_id": 1, "success": True, "action_taken": "assign_pending_task"},
                {"bot_id": 2, "success": True, "action_taken": "reset_status"}
            ]
            
            # Mock WorkerBot.query.get
            with patch('app.services.bot_manager_lambda.WorkerBot.query.get') as mock_get:
                mock_get.side_effect = [MagicMock(), MagicMock()]
                
                result = bot_manager.auto_remediate_issues()
                
                # Check that _remediate_issue was called for both bots
                assert mock_remediate.call_count == 2
                
                # Check the result
                assert result["bots_remediated"] == 2
                assert result["actions_taken"] == 2
                assert result["successful_actions"] == 2
                assert len(result["remediation_actions"]) == 2
    
    def test_auto_remediate_issues_no_issues(self, bot_manager):
        """Test auto-remediation when no issues are found."""
        # Mock monitor_all_bots to return no issues
        mock_report = {
            "bots_requiring_attention": []
        }
        
        with patch.object(bot_manager, 'monitor_all_bots', return_value=mock_report):
            result = bot_manager.auto_remediate_issues()
            
            assert result["status"] == "success"
            assert "No bots requiring remediation" in result["message"]
    
    def test_remediate_issue_status_error(self, bot_manager, mock_db):
        """Test remediation of a bot in error status."""
        # Create a mock bot
        mock_bot = MagicMock()
        mock_bot.id = 1
        mock_bot.name = "Bot 1"
        mock_bot.status = "error"
        
        # Create a mock issue
        issue = {"type": "status", "severity": "error", "message": "Bot is in error state"}
        
        result = bot_manager._remediate_issue(mock_bot, issue)
        
        # Check that status was updated
        mock_bot.update_status.assert_called_once_with("idle")
        
        # Check the result
        assert result["bot_id"] == 1
        assert result["bot_name"] == "Bot 1"
        assert result["issue_type"] == "status"
        assert result["action_taken"] == "reset_status"
        assert result["success"] is True
    
    def test_remediate_issue_stuck_task(self, bot_manager, mock_db, mock_task):
        """Test remediation of a stuck task."""
        # Create a mock bot
        mock_bot = MagicMock()
        mock_bot.id = 1
        mock_bot.name = "Bot 1"
        mock_bot.status = "working"
        
        # Set up mock for tasks
        mock_task_query = MagicMock()
        stuck_task = mock_task["task1"]
        mock_task_query.filter_by.return_value.first.return_value = stuck_task
        mock_bot.assigned_tasks = mock_task_query
        
        # Create a mock issue
        issue = {"type": "stuck_task", "severity": "error", "message": "Bot has been working on task too long"}
        
        result = bot_manager._remediate_issue(mock_bot, issue)
        
        # Check that task was updated
        stuck_task.update_status.assert_called_once_with("pending")
        assert stuck_task.assigned_to_bot_id is None
        mock_db.session.commit.assert_called_once()
        
        # Check that bot status was updated
        mock_bot.update_status.assert_called_once_with("idle")
        
        # Check the result
        assert result["bot_id"] == 1
        assert result["bot_name"] == "Bot 1"
        assert result["issue_type"] == "stuck_task"
        assert result["action_taken"] == "reset_stuck_task"
        assert result["success"] is True
    
    def test_task_matches_bot_capabilities(self, bot_manager):
        """Test matching tasks to bot capabilities."""
        # Create mock task and bot
        mock_task = MagicMock()
        mock_task.type = "feature"
        
        mock_bot = MagicMock()
        mock_bot.type = "developer"
        
        # Test matching
        assert bot_manager._task_matches_bot_capabilities(mock_task, mock_bot) is True
        
        # Test non-matching
        mock_bot.type = "devops"
        assert bot_manager._task_matches_bot_capabilities(mock_task, mock_bot) is False
    
    def test_calculate_task_completion_rate(self, bot_manager):
        """Test calculation of task completion rate."""
        # Create mock bot
        mock_bot = MagicMock()
        
        # Set up task counts
        task_query = MagicMock()
        task_query.filter_by.return_value.count.return_value = 3  # Completed tasks
        task_query.count.return_value = 5  # Total tasks
        mock_bot.assigned_tasks = task_query
        
        rate = bot_manager._calculate_task_completion_rate(mock_bot)
        assert rate == 60.0
        
        # Test with no tasks
        task_query.count.return_value = 0
        rate = bot_manager._calculate_task_completion_rate(mock_bot)
        assert rate == 100.0
    
    def test_calculate_average_task_duration(self, bot_manager):
        """Test calculation of average task duration."""
        # Create mock bot
        mock_bot = MagicMock()
        
        # Create mock completed tasks
        task1 = MagicMock()
        task1.start_date = datetime.now() - timedelta(hours=2)
        task1.completion_date = datetime.now()
        
        task2 = MagicMock()
        task2.start_date = datetime.now() - timedelta(hours=1)
        task2.completion_date = datetime.now()
        
        # Set up task queries
        task_query = MagicMock()
        task_query.filter_by.return_value.all.return_value = [task1, task2]
        mock_bot.assigned_tasks = task_query
        
        duration = bot_manager._calculate_average_task_duration(mock_bot)
        assert 3600 <= duration <= 7200  # Between 1-2 hours in seconds
        
        # Test with no completed tasks
        task_query.filter_by.return_value.all.return_value = []
        duration = bot_manager._calculate_average_task_duration(mock_bot)
        assert duration == 0.0
    
    def test_update_bot_metrics(self, bot_manager, mock_flask_app):
        """Test updating bot metrics."""
        bot_id = 1
        metrics = {"task_completion_rate": 75.0, "average_task_duration": 3600.0}
        
        bot_manager._update_bot_metrics(bot_id, metrics)
        
        # Check that metrics were stored
        assert bot_id in bot_manager.bot_metrics
        assert len(bot_manager.bot_metrics[bot_id]) == 1
        assert "timestamp" in bot_manager.bot_metrics[bot_id][0]
        assert bot_manager.bot_metrics[bot_id][0]["task_completion_rate"] == 75.0
        
        # Test trimming
        for i in range(100):
            bot_manager._update_bot_metrics(bot_id, {"test": i})
        
        # Check that history was trimmed
        max_history = mock_flask_app.config['MAX_BOT_METRICS_HISTORY']
        assert len(bot_manager.bot_metrics[bot_id]) == max_history
        assert bot_manager.bot_metrics[bot_id][-1]["test"] == 99
    
    def test_publish_metrics(self, bot_manager):
        """Test publishing metrics to CloudWatch."""
        # Create summary data
        summary = {
            "healthy_bots": 2,
            "warning_bots": 1,
            "error_bots": 0,
            "bot_reports": [
                {
                    "bot_id": 1,
                    "metrics": {
                        "task_completion_rate": 75.0,
                        "average_task_duration": 3600.0,
                        "error_rate": 0.0
                    }
                }
            ]
        }
        
        with patch.object(bot_manager.cloudwatch, 'put_metric_data') as mock_put_metrics:
            bot_manager._publish_metrics(summary)
            
            # Check that put_metric_data was called twice (once for overall, once for bot)
            assert mock_put_metrics.call_count == 2
            
            # Check that correct namespace was used
            call_args = mock_put_metrics.call_args_list[0][1]
            assert call_args["Namespace"] == "ProjectPilot/BotManager"
            
            # Check that metrics were included
            metric_data = call_args["MetricData"]
            assert len(metric_data) == 3
            assert any(m["MetricName"] == "HealthyBots" and m["Value"] == 2 for m in metric_data)
    
    def test_get_bot_health_summary(self, bot_manager, mock_worker_bot, mock_task):
        """Test getting bot health summary."""
        # Mock _get_tasks_completed_today
        with patch.object(bot_manager, '_get_tasks_completed_today', return_value=10):
            result = bot_manager.get_bot_health_summary()
            
            # Check the result
            assert "timestamp" in result
            assert result["total_bots"] == 2
            assert result["healthy_bots"] == 1  # Bot 1 is idle
            assert result["health_percentage"] == 50.0
            assert "status_distribution" in result
            assert "type_distribution" in result
            assert "provider_distribution" in result
            assert result["tasks_completed_today"] == 10
            assert result["system_health"] == "fair"
    
    def test_lambda_handler(self, bot_manager):
        """Test AWS Lambda handler."""
        # Mock the methods that lambda_handler calls
        with patch.object(bot_manager, 'monitor_all_bots', return_value={"status": "success"}), \
             patch.object(bot_manager, 'auto_remediate_issues', return_value={"status": "success"}), \
             patch.object(bot_manager, 'get_bot_health_summary', return_value={"status": "success"}):
            
            # Test monitor action
            event = {"action": "monitor"}
            context = {}
            result = bot_manager.lambda_handler(event, context)
            
            assert "execution_timestamp" in result
            assert result["execution_action"] == "monitor"
            
            # Test remediate action
            event = {"action": "remediate"}
            result = bot_manager.lambda_handler(event, context)
            assert result["execution_action"] == "remediate"
            
            # Test health action
            event = {"action": "health"}
            result = bot_manager.lambda_handler(event, context)
            assert result["execution_action"] == "health"
            
            # Test unknown action
            event = {"action": "unknown"}
            result = bot_manager.lambda_handler(event, context)
            assert "error" in result["status"]
            assert "Unknown action" in result["message"]

if __name__ == "__main__":
    pytest.main(["-v", __file__])