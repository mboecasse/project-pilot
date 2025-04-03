"""
Test file for the SelfImprovementSystem utility.
"""

import os
import sys
import pytest
import tempfile
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the module to test
from app.services.self_improve_utils import SelfImprovementSystem

class TestSelfImprovementSystem:
    """Tests for the SelfImprovementSystem class."""
    
    @pytest.fixture
    def mock_three_ai_workflow(self):
        """Create a mock ThreeAIWorkflow."""
        mock_workflow = MagicMock()
        mock_workflow.worker_bot_execute_task.return_value = {
            "result": {
                "analysis": {
                    "code_quality": "Good",
                    "potential_bugs": ["None detected"],
                    "performance_issues": ["None detected"]
                },
                "improvement_suggestions": [],
                "improved_function": "def improved_function(): pass"
            },
            "provider": "anthropic"
        }
        return mock_workflow
    
    @pytest.fixture
    def mock_flask_app(self):
        """Create a mock Flask app context."""
        mock_app = MagicMock()
        mock_app.config = {
            'MAX_PERFORMANCE_HISTORY': 50
        }
        with patch('app.services.self_improve_utils.current_app', mock_app):
            yield mock_app
    
    @pytest.fixture
    def system(self, mock_three_ai_workflow):
        """Create a SelfImprovementSystem instance."""
        return SelfImprovementSystem(three_ai_workflow=mock_three_ai_workflow)
    
    def test_init(self, mock_three_ai_workflow):
        """Test initialization of SelfImprovementSystem."""
        system = SelfImprovementSystem(three_ai_workflow=mock_three_ai_workflow)
        
        assert system.three_ai_workflow == mock_three_ai_workflow
        assert system.performance_metrics == {}
        assert system.improvement_history == []
    
    def test_analyze_function(self, system):
        """Test analyzing a function."""
        def test_function(a, b):
            """Test function docstring."""
            return a + b
        
        result = system.analyze_function(test_function)
        
        assert "function_metadata" in result
        assert result["function_metadata"]["name"] == "test_function"
        assert system.three_ai_workflow.worker_bot_execute_task.called
        
        # Check the bot type and preferred provider
        call_args = system.three_ai_workflow.worker_bot_execute_task.call_args[1]
        assert call_args["bot_type"] == "developer"
        assert call_args["preferred_provider"] == "anthropic"
    
    def test_analyze_function_not_callable(self, system):
        """Test analyzing a non-callable object."""
        result = system.analyze_function("not_a_function")
        
        assert "error" in result
        assert "not a callable function" in result["error"]
        assert not system.three_ai_workflow.worker_bot_execute_task.called
    
    @patch('app.services.self_improve_utils.inspect.getsource')
    def test_analyze_function_source_error(self, mock_getsource, system):
        """Test handling an error when getting function source."""
        def test_function():
            pass
            
        mock_getsource.side_effect = Exception("Source error")
        
        result = system.analyze_function(test_function)
        
        assert "error" in result
        assert "Could not get source" in result["error"]
        assert not system.three_ai_workflow.worker_bot_execute_task.called
    
    @patch('app.services.self_improve_utils.os.path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    def test_improve_module(self, mock_open, mock_exists, system):
        """Test improving a module."""
        # Set up the mocks
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "def test(): pass"
        mock_open.return_value = mock_file
        
        result = system.improve_module('/path/to/module.py')
        
        assert "module_metadata" in result
        assert result["module_metadata"]["name"] == "module"
        assert system.three_ai_workflow.worker_bot_execute_task.called
        
        # Check the bot type and preferred provider
        call_args = system.three_ai_workflow.worker_bot_execute_task.call_args[1]
        assert call_args["bot_type"] == "developer"
        assert call_args["preferred_provider"] == "anthropic"
    
    def test_improve_module_invalid_path(self, system):
        """Test improving a module with an invalid path."""
        with patch('app.services.self_improve_utils.os.path.exists', return_value=False):
            result = system.improve_module('/path/to/nonexistent.py')
            
            assert "error" in result
            assert "Invalid Python module path" in result["error"]
            assert not system.three_ai_workflow.worker_bot_execute_task.called
    
    def test_track_performance(self, system, mock_flask_app):
        """Test tracking performance metrics."""
        component_name = "test_component"
        metrics = {"execution_time": 100, "success": True}
        
        system.track_performance(component_name, metrics)
        
        assert component_name in system.performance_metrics
        assert len(system.performance_metrics[component_name]) == 1
        assert "timestamp" in system.performance_metrics[component_name][0]
        assert system.performance_metrics[component_name][0]["execution_time"] == 100
        
        # Test adding multiple metrics
        for i in range(10):
            system.track_performance(component_name, {"execution_time": i})
            
        assert len(system.performance_metrics[component_name]) == 11
    
    def test_track_performance_history_limit(self, system, mock_flask_app):
        """Test that performance history is limited."""
        component_name = "test_component"
        
        # Add metrics beyond the limit
        limit = mock_flask_app.config['MAX_PERFORMANCE_HISTORY']
        for i in range(limit + 10):
            system.track_performance(component_name, {"iteration": i})
            
        # Check that history is trimmed
        assert len(system.performance_metrics[component_name]) == limit
        
        # Check that oldest entries were removed
        iterations = [m["iteration"] for m in system.performance_metrics[component_name]]
        assert iterations[0] == 10
        assert iterations[-1] == limit + 9
    
    def test_get_performance_trends(self, system):
        """Test getting performance trends."""
        component_name = "test_component"
        
        # Add some metrics with improving trend
        with patch('app.services.self_improve_utils.datetime') as mock_datetime:
            for i in range(5):
                mock_datetime.now.return_value.isoformat.return_value = f"2023-01-0{i+1}"
                system.track_performance(component_name, {"execution_time": 100 - i*10})
        
        result = system.get_performance_trends(component_name)
        
        assert result["component"] == component_name
        assert result["data_points"] == 5
        assert "trends" in result
        assert "execution_time" in result["trends"]
        assert result["trends"]["execution_time"]["trend"] == "improving"
        assert result["trends"]["execution_time"]["current"] == 60
    
    def test_get_performance_trends_no_data(self, system):
        """Test getting performance trends with no data."""
        result = system.get_performance_trends("nonexistent")
        
        assert "error" in result
        assert "No performance data" in result["error"]
    
    @patch('app.services.self_improve_utils.os.path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    def test_apply_improvements(self, mock_open, mock_exists, system):
        """Test applying improvements to a module."""
        # Set up the mocks
        mock_exists.return_value = True
        
        # Mock the file operations
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "original content"
        mock_open.return_value = mock_file
        
        improvements = {
            "improved_module": "improved content",
            "improvement_suggestions": [{"description": "Suggestion 1"}]
        }
        
        result = system.apply_improvements('/path/to/module.py', improvements)
        
        assert result["success"] is True
        assert result["module_path"] == '/path/to/module.py'
        assert result["improvements_applied"] == 1
        assert mock_open.called
        
        # Check that the backup and write operations were performed
        file_write_calls = [call[0][0] for call in mock_file.__enter__.return_value.write.call_args_list]
        assert "original content" in file_write_calls
        assert "improved content" in file_write_calls
    
    def test_apply_improvements_invalid_path(self, system):
        """Test applying improvements with an invalid path."""
        with patch('app.services.self_improve_utils.os.path.exists', return_value=False):
            result = system.apply_improvements('/path/to/nonexistent.py', {"improved_module": "content"})
            
            assert "error" in result
            assert "Invalid Python module path" in result["error"]
    
    def test_apply_improvements_missing_content(self, system):
        """Test applying improvements with missing content."""
        with patch('app.services.self_improve_utils.os.path.exists', return_value=True):
            result = system.apply_improvements('/path/to/module.py', {})
            
            assert "error" in result
            assert "No improved module content provided" in result["error"]
    
    @patch('app.services.self_improve_utils.inspect.getsource')
    def test_generate_test_cases_function(self, mock_getsource, system):
        """Test generating test cases for a function."""
        def test_function(a, b):
            """Test function."""
            return a + b
            
        mock_getsource.return_value = "def test_function(a, b):\n    return a + b"
        
        result = system.generate_test_cases(test_function)
        
        assert "function" in result
        assert result["function"] == "test_function"
        assert "test_code" in result
        assert system.three_ai_workflow.worker_bot_execute_task.called
        
        # Check the bot type and preferred provider
        call_args = system.three_ai_workflow.worker_bot_execute_task.call_args[1]
        assert call_args["bot_type"] == "tester"
        assert call_args["preferred_provider"] == "openai"
    
    @patch('app.services.self_improve_utils.os.path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    def test_generate_test_cases_module(self, mock_open, mock_exists, system):
        """Test generating test cases for a module."""
        # Set up the mocks
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "def test(): pass"
        mock_open.return_value = mock_file
        
        result = system.generate_test_cases('/path/to/module.py')
        
        assert "module" in result
        assert result["module"] == "module"
        assert "test_code" in result
        assert system.three_ai_workflow.worker_bot_execute_task.called
        
        # Check the bot type and preferred provider
        call_args = system.three_ai_workflow.worker_bot_execute_task.call_args[1]
        assert call_args["bot_type"] == "tester"
        assert call_args["preferred_provider"] == "openai"
    
    def test_generate_test_cases_invalid_input(self, system):
        """Test generating test cases with invalid input."""
        result = system.generate_test_cases(123)
        
        assert "error" in result
        assert "Input must be a callable function or a path" in result["error"]
    
    def test_analyze_workflow(self, system):
        """Test analyzing workflow logs."""
        workflow_logs = [
            {
                "workflow_name": "test_workflow",
                "timestamp": "2023-01-01T12:00:00",
                "duration": 5.2,
                "success": True
            },
            {
                "workflow_name": "test_workflow",
                "timestamp": "2023-01-02T12:00:00",
                "duration": 4.8,
                "success": True
            }
        ]
        
        result = system.analyze_workflow(workflow_logs)
        
        assert "workflow_metadata" in result
        assert result["workflow_metadata"]["name"] == "test_workflow"
        assert result["workflow_metadata"]["logs_analyzed"] == 2
        assert system.three_ai_workflow.worker_bot_execute_task.called
        
        # Check the bot type and preferred provider
        call_args = system.three_ai_workflow.worker_bot_execute_task.call_args[1]
        assert call_args["bot_type"] == "architect"
        assert call_args["preferred_provider"] == "anthropic"
    
    def test_analyze_workflow_empty_logs(self, system):
        """Test analyzing workflow with empty logs."""
        result = system.analyze_workflow([])
        
        assert "error" in result
        assert "No workflow logs provided" in result["error"]
        assert not system.three_ai_workflow.worker_bot_execute_task.called
    
    @patch('app.services.self_improve_utils.json.loads')
    def test_handle_json_response(self, mock_json_loads, system):
        """Test handling JSON response from AI."""
        # Mock the JSON parsing
        mock_json_loads.return_value = {"key": "value"}
        
        # Mock the AI response to return a string with JSON
        system.three_ai_workflow.worker_bot_execute_task.return_value = {
            "result": '{"key": "value"}'
        }
        
        # Test with a function that uses JSON parsing
        def test_function():
            pass
            
        result = system.analyze_function(test_function)
        
        assert "function_metadata" in result
        assert mock_json_loads.called

if __name__ == "__main__":
    pytest.main(["-v", __file__])