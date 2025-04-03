"""
ProjectPilot - AI-powered project management system
Self-improvement utilities for AI components.
"""

import logging
import json
import os
import re
import ast
import inspect
from typing import List, Dict, Any, Optional, Union, Callable
from flask import current_app

from app.services.three_ai_workflow import ThreeAIWorkflow

logger = logging.getLogger(__name__)

class SelfImprovementSystem:
    """
    System that enables AI components to analyze and improve their own code,
    algorithms, and performance based on feedback and results.
    """
    
    def __init__(self, three_ai_workflow: Optional[ThreeAIWorkflow] = None):
        """
        Initialize the self-improvement system.
        
        Args:
            three_ai_workflow: ThreeAIWorkflow instance for AI operations
        """
        self.three_ai_workflow = three_ai_workflow or ThreeAIWorkflow()
        self.performance_metrics = {}
        self.improvement_history = []
        logger.info("Self-improvement system initialized")
    
    def analyze_function(self, func: Callable) -> Dict[str, Any]:
        """
        Analyze a Python function for potential improvements.
        
        Args:
            func: Function to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not callable(func):
            return {"error": "Input is not a callable function"}
        
        # Get function source code
        try:
            source = inspect.getsource(func)
        except Exception as e:
            logger.error(f"Could not get source for function {func.__name__}: {str(e)}")
            return {"error": f"Could not get source: {str(e)}"}
        
        # Get function metadata
        module = func.__module__
        name = func.__name__
        doc = func.__doc__ or ""
        
        # Prepare prompt for AI analysis
        prompt = f"""
        Analyze the following Python function and suggest improvements:
        
        Module: {module}
        Function: {name}
        
        ```python
        {source}
        ```
        
        Please analyze this function for:
        1. Code quality and readability
        2. Potential bugs or edge cases
        3. Performance optimizations
        4. Better error handling
        5. Improved documentation
        
        Format your response as JSON with the following structure:
        {{
            "analysis": {{
                "code_quality": "Assessment of code quality",
                "potential_bugs": ["Bug risk 1", "Bug risk 2"],
                "performance_issues": ["Performance issue 1", "Performance issue 2"],
                "error_handling": "Assessment of error handling",
                "documentation": "Assessment of documentation"
            }},
            "improvement_suggestions": [
                {{
                    "description": "Description of the improvement",
                    "current_code": "Relevant code snippet that should be improved",
                    "improved_code": "Suggested improved code",
                    "explanation": "Explanation of why this is better",
                    "impact": "high|medium|low"
                }}
            ],
            "improved_function": "Complete improved version of the function"
        }}
        """
        
        # Use AI to analyze the function
        try:
            response = self.three_ai_workflow.worker_bot_execute_task(
                bot_type="developer",
                task_description="Analyze and improve function code",
                project_context=f"Function '{name}' in module '{module}'",
                preferred_provider="anthropic"  # Use Claude for code quality analysis
            )
            
            if "error" in response:
                return {"error": response["error"]}
            
            result = response.get("result", {})
            if isinstance(result, str):
                # Try to extract JSON from text response
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    try:
                        json_content = result[json_start:json_end]
                        result = json.loads(json_content)
                    except json.JSONDecodeError:
                        return {"error": "Failed to parse AI response as JSON"}
            
            # Add metadata to result
            result["function_metadata"] = {
                "module": module,
                "name": name,
                "original_source": source
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during function analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def improve_module(self, module_path: str) -> Dict[str, Any]:
        """
        Analyze and improve a Python module.
        
        Args:
            module_path: Path to the Python module file
            
        Returns:
            Dictionary with analysis and improvement suggestions
        """
        if not os.path.exists(module_path) or not module_path.endswith('.py'):
            return {"error": "Invalid Python module path"}
        
        try:
            with open(module_path, 'r') as f:
                module_content = f.read()
        except Exception as e:
            logger.error(f"Could not read module file {module_path}: {str(e)}")
            return {"error": f"Could not read module: {str(e)}"}
        
        # Extract module name from path
        module_name = os.path.basename(module_path).replace('.py', '')
        
        # Prepare prompt for AI analysis
        prompt = f"""
        Analyze the following Python module and suggest improvements:
        
        Module: {module_name}
        Path: {module_path}
        
        ```python
        {module_content}
        ```
        
        Please analyze this module for:
        1. Overall structure and organization
        2. Code quality and readability
        3. Potential bugs or edge cases
        4. Performance optimizations
        5. Better error handling
        6. Improved documentation
        7. Unused code or imports
        
        Format your response as JSON with the following structure:
        {{
            "module_analysis": {{
                "overall_quality": "Assessment of overall quality",
                "structure": "Assessment of module structure",
                "potential_bugs": ["Bug risk 1", "Bug risk 2"],
                "performance_issues": ["Performance issue 1", "Performance issue 2"],
                "code_smells": ["Code smell 1", "Code smell 2"]
            }},
            "function_analyses": [
                {{
                    "function_name": "Name of function",
                    "issues": ["Issue 1", "Issue 2"],
                    "suggestions": ["Suggestion 1", "Suggestion 2"]
                }}
            ],
            "improvement_suggestions": [
                {{
                    "description": "Description of the improvement",
                    "current_code": "Relevant code snippet that should be improved",
                    "improved_code": "Suggested improved code",
                    "explanation": "Explanation of why this is better",
                    "impact": "high|medium|low"
                }}
            ],
            "improved_module": "Complete improved version of the module"
        }}
        """
        
        # Use AI to analyze the module
        try:
            response = self.three_ai_workflow.worker_bot_execute_task(
                bot_type="developer",
                task_description="Analyze and improve Python module",
                project_context=f"Module '{module_name}' at path '{module_path}'",
                preferred_provider="anthropic"  # Use Claude for code quality analysis
            )
            
            if "error" in response:
                return {"error": response["error"]}
            
            result = response.get("result", {})
            if isinstance(result, str):
                # Try to extract JSON from text response
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    try:
                        json_content = result[json_start:json_end]
                        result = json.loads(json_content)
                    except json.JSONDecodeError:
                        return {"error": "Failed to parse AI response as JSON"}
            
            # Add metadata to result
            result["module_metadata"] = {
                "name": module_name,
                "path": module_path,
                "size": len(module_content),
                "line_count": module_content.count('\n') + 1
            }
            
            # Record this improvement analysis in history
            self.improvement_history.append({
                "timestamp": datetime.now().isoformat(),
                "module": module_name,
                "suggestions_count": len(result.get("improvement_suggestions", [])),
                "major_issues": [
                    issue for issue in result.get("module_analysis", {}).get("potential_bugs", [])
                    if "critical" in issue.lower() or "high" in issue.lower()
                ]
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error during module analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def track_performance(self, component_name: str, metrics: Dict[str, Any]) -> None:
        """
        Track performance metrics for a component.
        
        Args:
            component_name: Name of the component
            metrics: Performance metrics
        """
        if component_name not in self.performance_metrics:
            self.performance_metrics[component_name] = []
        
        # Add timestamp to metrics
        metrics["timestamp"] = datetime.now().isoformat()
        
        # Store metrics
        self.performance_metrics[component_name].append(metrics)
        
        # Trim history if too long
        max_history = current_app.config.get('MAX_PERFORMANCE_HISTORY', 100)
        if len(self.performance_metrics[component_name]) > max_history:
            self.performance_metrics[component_name] = self.performance_metrics[component_name][-max_history:]
        
        logger.debug(f"Tracked performance metrics for {component_name}: {metrics}")
    
    def get_performance_trends(self, component_name: str) -> Dict[str, Any]:
        """
        Get performance trends for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dictionary with performance trends
        """
        if component_name not in self.performance_metrics:
            return {"error": f"No performance data for {component_name}"}
        
        metrics = self.performance_metrics[component_name]
        if not metrics:
            return {"error": f"Empty performance data for {component_name}"}
        
        # Extract common metric keys
        metric_keys = set()
        for m in metrics:
            metric_keys.update(k for k in m.keys() if k != "timestamp")
        
        # Calculate trends
        trends = {}
        for key in metric_keys:
            values = [m.get(key) for m in metrics if key in m and isinstance(m.get(key), (int, float))]
            if values:
                trends[key] = {
                    "current": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "average": sum(values) / len(values),
                    "trend": "improving" if len(values) > 1 and values[-1] > values[0] else 
                             "declining" if len(values) > 1 and values[-1] < values[0] else "stable",
                    "change_percent": ((values[-1] - values[0]) / values[0] * 100) if len(values) > 1 and values[0] != 0 else 0
                }
        
        return {
            "component": component_name,
            "data_points": len(metrics),
            "time_range": {
                "start": metrics[0].get("timestamp"),
                "end": metrics[-1].get("timestamp")
            },
            "trends": trends
        }
    
    def apply_improvements(self, module_path: str, improvements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply suggested improvements to a module.
        
        Args:
            module_path: Path to the Python module file
            improvements: Improvement suggestions from analyze_module
            
        Returns:
            Dictionary with results of the improvement process
        """
        if not os.path.exists(module_path) or not module_path.endswith('.py'):
            return {"error": "Invalid Python module path"}
        
        if "improved_module" not in improvements:
            return {"error": "No improved module content provided"}
        
        # Create backup of the original file
        backup_path = f"{module_path}.bak"
        try:
            with open(module_path, 'r') as src:
                original_content = src.read()
                
            with open(backup_path, 'w') as dst:
                dst.write(original_content)
                
            # Write the improved module
            with open(module_path, 'w') as f:
                f.write(improvements["improved_module"])
                
            logger.info(f"Applied improvements to module {module_path} (backup at {backup_path})")
            
            return {
                "success": True,
                "module_path": module_path,
                "backup_path": backup_path,
                "improvements_applied": len(improvements.get("improvement_suggestions", [])),
                "module_size_before": len(original_content),
                "module_size_after": len(improvements["improved_module"])
            }
                
        except Exception as e:
            logger.error(f"Error applying improvements to {module_path}: {str(e)}")
            # Try to restore from backup if exists
            if os.path.exists(backup_path):
                try:
                    with open(backup_path, 'r') as src:
                        with open(module_path, 'w') as dst:
                            dst.write(src.read())
                    logger.info(f"Restored original module from backup")
                except Exception as restore_error:
                    logger.error(f"Error restoring from backup: {str(restore_error)}")
                    
            return {"error": f"Failed to apply improvements: {str(e)}"}
    
    def generate_test_cases(self, func_or_path: Union[Callable, str]) -> Dict[str, Any]:
        """
        Generate test cases for a function or module.
        
        Args:
            func_or_path: Function or path to a Python module
            
        Returns:
            Dictionary with generated test cases
        """
        if callable(func_or_path):
            # Generate tests for a function
            func = func_or_path
            try:
                source = inspect.getsource(func)
                module = func.__module__
                name = func.__name__
            except Exception as e:
                logger.error(f"Could not get source for function: {str(e)}")
                return {"error": f"Could not get source: {str(e)}"}
                
            prompt = f"""
            Generate comprehensive test cases for the following Python function:
            
            Module: {module}
            Function: {name}
            
            ```python
            {source}
            ```
            
            Please generate test cases that cover:
            1. Normal usage scenarios
            2. Edge cases
            3. Error conditions
            4. Boundary values
            
            Format your response as Python code using pytest framework.
            Include docstrings explaining each test case.
            """
            
            try:
                response = self.three_ai_workflow.worker_bot_execute_task(
                    bot_type="tester",
                    task_description=f"Generate test cases for function {name}",
                    project_context=f"Function '{name}' in module '{module}'",
                    preferred_provider="openai"  # OpenAI often does well with test case generation
                )
                
                if "error" in response:
                    return {"error": response["error"]}
                
                result = response.get("result", "")
                if isinstance(result, dict):
                    # Extract test cases if result is a dict
                    test_code = result.get("test_code", result.get("test_cases", ""))
                else:
                    # Assume result is the test code directly
                    test_code = result
                
                # Extract pytest code if embedded in explanation
                if isinstance(test_code, str):
                    code_blocks = re.findall(r'```python\s+(.*?)\s+```', test_code, re.DOTALL)
                    if code_blocks:
                        test_code = "\n\n".join(code_blocks)
                
                return {
                    "function": name,
                    "module": module,
                    "test_code": test_code,
                    "provider": response.get("provider")
                }
                
            except Exception as e:
                logger.error(f"Error generating test cases: {str(e)}")
                return {"error": f"Test generation failed: {str(e)}"}
                
        elif isinstance(func_or_path, str) and os.path.exists(func_or_path) and func_or_path.endswith('.py'):
            # Generate tests for a module
            module_path = func_or_path
            try:
                with open(module_path, 'r') as f:
                    module_content = f.read()
            except Exception as e:
                logger.error(f"Could not read module file {module_path}: {str(e)}")
                return {"error": f"Could not read module: {str(e)}"}
                
            # Extract module name from path
            module_name = os.path.basename(module_path).replace('.py', '')
                
            prompt = f"""
            Generate comprehensive test cases for the following Python module:
            
            Module: {module_name}
            Path: {module_path}
            
            ```python
            {module_content}
            ```
            
            Please generate test cases that cover all key functions in this module.
            For each function, include tests for:
            1. Normal usage scenarios
            2. Edge cases
            3. Error conditions
            4. Boundary values
            
            Format your response as Python code using pytest framework.
            Include docstrings explaining each test case.
            Organize tests into test classes based on functionality.
            """
            
            try:
                response = self.three_ai_workflow.worker_bot_execute_task(
                    bot_type="tester",
                    task_description=f"Generate test cases for module {module_name}",
                    project_context=f"Module at path '{module_path}'",
                    preferred_provider="openai"
                )
                
                if "error" in response:
                    return {"error": response["error"]}
                
                result = response.get("result", "")
                if isinstance(result, dict):
                    # Extract test cases if result is a dict
                    test_code = result.get("test_code", result.get("test_cases", ""))
                else:
                    # Assume result is the test code directly
                    test_code = result
                
                # Extract pytest code if embedded in explanation
                if isinstance(test_code, str):
                    code_blocks = re.findall(r'```python\s+(.*?)\s+```', test_code, re.DOTALL)
                    if code_blocks:
                        test_code = "\n\n".join(code_blocks)
                
                return {
                    "module": module_name,
                    "path": module_path,
                    "test_code": test_code,
                    "provider": response.get("provider")
                }
                
            except Exception as e:
                logger.error(f"Error generating test cases: {str(e)}")
                return {"error": f"Test generation failed: {str(e)}"}
        else:
            return {"error": "Input must be a callable function or a path to a Python module"}
    
    def analyze_workflow(self, workflow_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze workflow execution logs to identify improvements.
        
        Args:
            workflow_logs: List of workflow execution logs
            
        Returns:
            Dictionary with analysis and improvement suggestions
        """
        if not workflow_logs:
            return {"error": "No workflow logs provided"}
        
        # Extract workflow name and other metadata
        workflow_name = workflow_logs[0].get("workflow_name", "Unknown")
        
        # Prepare prompt for AI analysis
        log_sample = json.dumps(workflow_logs[:min(5, len(workflow_logs))], indent=2)
        
        prompt = f"""
        Analyze the following workflow execution logs and suggest improvements:
        
        Workflow: {workflow_name}
        Log Sample:
        ```
        {log_sample}
        ```
        
        Total logs provided: {len(workflow_logs)}
        
        Please analyze these workflow logs for:
        1. Efficiency issues (slow steps, bottlenecks)
        2. Error patterns
        3. Resource usage patterns
        4. Success rate
        5. Potential optimizations
        
        Format your response as JSON with the following structure:
        {{
            "workflow_analysis": {{
                "efficiency": "Assessment of workflow efficiency",
                "error_patterns": ["Error pattern 1", "Error pattern 2"],
                "resource_usage": "Assessment of resource usage",
                "success_rate": "Calculated success rate",
                "bottlenecks": ["Bottleneck 1", "Bottleneck 2"]
            }},
            "step_analyses": [
                {{
                    "step_name": "Name of workflow step",
                    "average_duration": "Average duration in seconds",
                    "failure_rate": "Failure rate percentage",
                    "common_errors": ["Error 1", "Error 2"],
                    "improvement_suggestions": ["Suggestion 1", "Suggestion 2"]
                }}
            ],
            "overall_suggestions": [
                {{
                    "description": "Description of the improvement",
                    "implementation": "Suggested implementation approach",
                    "expected_impact": "Expected impact of this improvement",
                    "priority": "high|medium|low"
                }}
            ]
        }}
        """
        
        # Use AI to analyze the workflow logs
        try:
            response = self.three_ai_workflow.worker_bot_execute_task(
                bot_type="architect",
                task_description=f"Analyze workflow logs for {workflow_name}",
                project_context=f"Workflow with {len(workflow_logs)} execution logs",
                preferred_provider="anthropic"  # Claude is good at long-context analysis
            )
            
            if "error" in response:
                return {"error": response["error"]}
            
            result = response.get("result", {})
            if isinstance(result, str):
                # Try to extract JSON from text response
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    try:
                        json_content = result[json_start:json_end]
                        result = json.loads(json_content)
                    except json.JSONDecodeError:
                        return {"error": "Failed to parse AI response as JSON"}
            
            # Add metadata to result
            result["workflow_metadata"] = {
                "name": workflow_name,
                "logs_analyzed": len(workflow_logs),
                "time_range": {
                    "start": workflow_logs[0].get("timestamp", "Unknown"),
                    "end": workflow_logs[-1].get("timestamp", "Unknown")
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during workflow analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

# Import this at the end to avoid circular imports
from datetime import datetime