"""
ProjectPilot - AI-powered project management system
Three-AI Workflow integration for combining multiple AI models.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional, Union
from flask import current_app

from app.services.openai_utils import OpenAIService
from app.services.anthropic_utils import AnthropicService
from app.services.aws_bedrock_utils import AWSBedrockService

logger = logging.getLogger(__name__)

class ThreeAIWorkflow:
    """
    Service for orchestrating workflows using multiple AI models for improved
    results and redundancy.
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 aws_region: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None):
        """
        Initialize the Three-AI Workflow service.
        
        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            aws_region: AWS region for Bedrock
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        # Initialize services
        self.openai_service = OpenAIService(api_key=openai_api_key)
        self.anthropic_service = AnthropicService(api_key=anthropic_api_key)
        self.bedrock_service = AWSBedrockService(
            region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # Store availability of services
        self.openai_available = bool(openai_api_key or os.environ.get('OPENAI_API_KEY') or 
                                    current_app.config.get('OPENAI_API_KEY'))
        self.anthropic_available = bool(anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY') or 
                                       current_app.config.get('ANTHROPIC_API_KEY'))
        self.bedrock_available = self.bedrock_service.is_available()
        
        # Log service initialization
        logger.info(f"Three-AI Workflow initialized with services: " +
                    f"OpenAI={'Available' if self.openai_available else 'Unavailable'}, " +
                    f"Anthropic={'Available' if self.anthropic_available else 'Unavailable'}, " +
                    f"AWS Bedrock={'Available' if self.bedrock_available else 'Unavailable'}")
    
    def analyze_project_requirements(self, requirements: str) -> Dict[str, Any]:
        """
        Analyze project requirements using multiple AI models for better results.
        
        Args:
            requirements: Project requirements text
            
        Returns:
            Dictionary with merged analysis from available AI models
        """
        results = []
        errors = []
        
        # Get responses from available services
        if self.openai_available:
            try:
                openai_response = self.openai_service.analyze_project_requirements(requirements)
                if "error" not in openai_response:
                    results.append(("openai", openai_response))
                else:
                    errors.append(f"OpenAI error: {openai_response.get('error')}")
            except Exception as e:
                logger.error(f"Error with OpenAI analysis: {str(e)}")
                errors.append(f"OpenAI exception: {str(e)}")
        
        if self.anthropic_available:
            try:
                anthropic_response = self.anthropic_service.analyze_project_requirements(requirements)
                if "error" not in anthropic_response:
                    results.append(("anthropic", anthropic_response))
                else:
                    errors.append(f"Anthropic error: {anthropic_response.get('error')}")
            except Exception as e:
                logger.error(f"Error with Anthropic analysis: {str(e)}")
                errors.append(f"Anthropic exception: {str(e)}")
        
        if self.bedrock_available:
            try:
                bedrock_response = self.bedrock_service.analyze_project_requirements(requirements)
                if "error" not in bedrock_response:
                    results.append(("bedrock", bedrock_response))
                else:
                    errors.append(f"AWS Bedrock error: {bedrock_response.get('error')}")
            except Exception as e:
                logger.error(f"Error with AWS Bedrock analysis: {str(e)}")
                errors.append(f"AWS Bedrock exception: {str(e)}")
        
        # Handle case where no services provided valid results
        if not results:
            error_msg = "; ".join(errors) if errors else "No AI services available"
            return {
                "error": f"Failed to analyze requirements: {error_msg}",
                "tasks": []
            }
        
        # Merge results if we have multiple responses
        if len(results) > 1:
            return self._merge_project_analyses(results)
        else:
            # Return the single successful result
            return results[0][1]
    
    def _merge_project_analyses(self, analyses: List[tuple]) -> Dict[str, Any]:
        """
        Merge project analyses from multiple AI models.
        
        Args:
            analyses: List of (provider, response) tuples
            
        Returns:
            Merged analysis
        """
        merged_tasks = []
        seen_task_names = set()
        all_risks = set()
        timeline_estimates = []
        suggested_architectures = []
        
        # Process each analysis
        for provider, analysis in analyses:
            # Add tasks avoiding duplicates by name
            for task in analysis.get("tasks", []):
                task_name = task.get("name", "").strip()
                if task_name and task_name not in seen_task_names:
                    task["provider"] = provider  # Track which provider suggested this task
                    merged_tasks.append(task)
                    seen_task_names.add(task_name)
            
            # Collect risks
            for risk in analysis.get("risks", []):
                all_risks.add(risk)
            
            # Collect timeline estimates
            if "timeline_estimate" in analysis:
                timeline_estimates.append(analysis["timeline_estimate"])
            
            # Collect architecture suggestions
            if "suggested_architecture" in analysis:
                suggested_architectures.append({
                    "provider": provider,
                    "architecture": analysis["suggested_architecture"]
                })
        
        # Construct merged result
        merged_result = {
            "tasks": merged_tasks,
            "risks": list(all_risks),
            "providers_used": [provider for provider, _ in analyses]
        }
        
        # Add timeline estimate (average if multiple)
        if timeline_estimates:
            merged_result["timeline_estimate"] = timeline_estimates[0]
            if len(timeline_estimates) > 1:
                merged_result["all_timeline_estimates"] = timeline_estimates
        
        # Add architecture suggestions
        if suggested_architectures:
            merged_result["suggested_architecture"] = suggested_architectures[0]["architecture"]
            merged_result["all_architecture_suggestions"] = suggested_architectures
        
        return merged_result
    
    def worker_bot_execute_task(self, 
                               bot_type: str, 
                               task_description: str, 
                               project_context: str,
                               preferred_provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a task using worker bot AI workflows.
        
        Args:
            bot_type: Type of worker bot (architect, developer, tester, devops)
            task_description: Description of the task
            project_context: Context information about the project
            preferred_provider: Preferred AI provider (openai, anthropic, bedrock)
            
        Returns:
            Dictionary with task results
        """
        if preferred_provider == "openai" and self.openai_available:
            return self._execute_with_openai(bot_type, task_description, project_context)
        elif preferred_provider == "anthropic" and self.anthropic_available:
            return self._execute_with_anthropic(bot_type, task_description, project_context)
        elif preferred_provider == "bedrock" and self.bedrock_available:
            return self._execute_with_bedrock(bot_type, task_description, project_context)
        else:
            # No preferred provider or preferred provider not available, try in order
            if self.openai_available:
                return self._execute_with_openai(bot_type, task_description, project_context)
            elif self.anthropic_available:
                return self._execute_with_anthropic(bot_type, task_description, project_context)
            elif self.bedrock_available:
                return self._execute_with_bedrock(bot_type, task_description, project_context)
            else:
                return {"error": "No AI services available", "result": None}
    
    def _execute_with_openai(self, 
                           bot_type: str, 
                           task_description: str, 
                           project_context: str) -> Dict[str, Any]:
        """Execute task using OpenAI."""
        try:
            if bot_type == "architect":
                # For architecture tasks, use specific prompt
                prompt = self._create_architect_prompt(task_description, project_context)
                response = self.openai_service.generate_text(prompt, model="gpt-4", temperature=0.2)
            elif bot_type == "developer":
                # For developer tasks, use task implementation method
                response = self.openai_service.generate_task_implementation(task_description, project_context)
                return {
                    "provider": "openai",
                    "result": response
                }
            else:
                # Generic task execution
                prompt = self._create_generic_bot_prompt(bot_type, task_description, project_context)
                response = self.openai_service.generate_text(prompt, model="gpt-4", temperature=0.4)
            
            if "error" in response:
                return {"error": response["error"], "provider": "openai"}
            
            # Parse JSON if possible
            try:
                result_text = response.get("text", "")
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = result_text[json_start:json_end]
                    parsed_result = json.loads(json_content)
                    return {
                        "provider": "openai",
                        "result": parsed_result,
                        "usage": response.get("usage", {})
                    }
                else:
                    return {
                        "provider": "openai",
                        "result": result_text,
                        "usage": response.get("usage", {})
                    }
            except json.JSONDecodeError:
                return {
                    "provider": "openai",
                    "result": response.get("text", ""),
                    "usage": response.get("usage", {})
                }
                
        except Exception as e:
            logger.error(f"Error executing task with OpenAI: {str(e)}")
            return {"error": str(e), "provider": "openai"}
    
    def _execute_with_anthropic(self, 
                              bot_type: str, 
                              task_description: str, 
                              project_context: str) -> Dict[str, Any]:
        """Execute task using Anthropic Claude."""
        try:
            if bot_type == "architect":
                # For architecture tasks, use specific prompt
                prompt = self._create_architect_prompt(task_description, project_context)
                response = self.anthropic_service.generate_text(prompt, model="claude-3-opus-20240229", temperature=0.2)
            elif bot_type == "developer":
                # For developer tasks, use task implementation method
                response = self.anthropic_service.generate_task_implementation(task_description, project_context)
                return {
                    "provider": "anthropic",
                    "result": response
                }
            else:
                # Generic task execution
                prompt = self._create_generic_bot_prompt(bot_type, task_description, project_context)
                response = self.anthropic_service.generate_text(prompt, model="claude-3-opus-20240229", temperature=0.4)
            
            if "error" in response:
                return {"error": response["error"], "provider": "anthropic"}
            
            # Parse JSON if possible
            try:
                result_text = response.get("text", "")
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = result_text[json_start:json_end]
                    parsed_result = json.loads(json_content)
                    return {
                        "provider": "anthropic",
                        "result": parsed_result,
                        "usage": response.get("usage", {})
                    }
                else:
                    return {
                        "provider": "anthropic",
                        "result": result_text,
                        "usage": response.get("usage", {})
                    }
            except json.JSONDecodeError:
                return {
                    "provider": "anthropic",
                    "result": response.get("text", ""),
                    "usage": response.get("usage", {})
                }
                
        except Exception as e:
            logger.error(f"Error executing task with Anthropic: {str(e)}")
            return {"error": str(e), "provider": "anthropic"}
    
    def _execute_with_bedrock(self, 
                            bot_type: str, 
                            task_description: str, 
                            project_context: str) -> Dict[str, Any]:
        """Execute task using AWS Bedrock."""
        try:
            if bot_type == "architect":
                # For architecture tasks, use specific prompt
                prompt = self._create_architect_prompt(task_description, project_context)
                response = self.bedrock_service.generate_text_claude(
                    prompt, 
                    model_id="anthropic.claude-3-opus-20240229-v1:0", 
                    temperature=0.2
                )
            elif bot_type == "developer" and task_description.lower().find("code") >= 0:
                # For developer tasks requiring code, use code generation
                language = self._extract_language_from_task(task_description)
                response = self.bedrock_service.generate_code(task_description, language, project_context)
                return {
                    "provider": "bedrock",
                    "result": response
                }
            else:
                # Generic task execution
                prompt = self._create_generic_bot_prompt(bot_type, task_description, project_context)
                response = self.bedrock_service.generate_text_claude(
                    prompt, 
                    model_id="anthropic.claude-3-opus-20240229-v1:0", 
                    temperature=0.4
                )
            
            if "error" in response:
                return {"error": response["error"], "provider": "bedrock"}
            
            # Parse JSON if possible
            try:
                result_text = response.get("text", "")
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = result_text[json_start:json_end]
                    parsed_result = json.loads(json_content)
                    return {
                        "provider": "bedrock",
                        "result": parsed_result,
                        "usage": response.get("usage", {})
                    }
                else:
                    return {
                        "provider": "bedrock",
                        "result": result_text,
                        "usage": response.get("usage", {})
                    }
            except json.JSONDecodeError:
                return {
                    "provider": "bedrock",
                    "result": response.get("text", ""),
                    "usage": response.get("usage", {})
                }
                
        except Exception as e:
            logger.error(f"Error executing task with AWS Bedrock: {str(e)}")
            return {"error": str(e), "provider": "bedrock"}
    
    def _create_architect_prompt(self, task_description: str, project_context: str) -> str:
        """Create prompt for architect bot tasks."""
        return f"""
        As an AI architect bot, your task is to provide system design and architectural guidance for the following task:
        
        TASK: {task_description}
        
        PROJECT CONTEXT: {project_context}
        
        Please provide a comprehensive architectural design including:
        1. Overall system architecture
        2. Component breakdown
        3. Data flow diagrams
        4. API specifications
        5. Database schema
        6. Technology stack recommendations
        7. Scalability considerations
        8. Security recommendations
        
        Format your response as JSON with the following structure:
        {{
            "architecture_overview": "High-level description of the architecture",
            "components": [
                {{
                    "name": "Component name",
                    "purpose": "Component purpose",
                    "technologies": ["Technology 1", "Technology 2"],
                    "interfaces": ["Interface 1", "Interface 2"]
                }}
            ],
            "data_flow": "Description of data flow",
            "api_endpoints": [
                {{
                    "path": "/api/path",
                    "method": "HTTP method",
                    "purpose": "Endpoint purpose",
                    "request_params": "Request parameters",
                    "response_format": "Response format"
                }}
            ],
            "database_schema": [
                {{
                    "table": "Table name",
                    "fields": [
                        {{
                            "name": "Field name",
                            "type": "Field type",
                            "description": "Field description"
                        }}
                    ],
                    "relationships": ["Relationship 1", "Relationship 2"]
                }}
            ],
            "tech_stack": ["Technology 1", "Technology 2"],
            "scaling_strategy": "Description of scaling strategy",
            "security_measures": ["Security measure 1", "Security measure 2"]
        }}

        Ensure your response is valid JSON.
        """
    
    def _create_generic_bot_prompt(self, bot_type: str, task_description: str, project_context: str) -> str:
        """Create prompt for generic bot tasks."""
        bot_descriptions = {
            "developer": "You are an AI developer bot responsible for implementing features, writing code, refactoring, and debugging.",
            "tester": "You are an AI tester bot responsible for creating test plans, test cases, and identifying potential issues.",
            "devops": "You are an AI DevOps bot responsible for infrastructure, deployment, monitoring, and operations."
        }
        
        return f"""
        {bot_descriptions.get(bot_type, "You are an AI worker bot")}
        
        TASK: {task_description}
        
        PROJECT CONTEXT: {project_context}
        
        Please provide a comprehensive solution to this task based on your role. 
        Format your response as JSON with relevant sections for your role.
        Ensure your response is valid JSON.
        """
    
    def _extract_language_from_task(self, task_description: str) -> str:
        """Extract programming language from task description or default to Python."""
        languages = ["python", "javascript", "typescript", "java", "c#", "go", "rust", "php", "ruby", "kotlin", "swift"]
        
        lower_desc = task_description.lower()
        for language in languages:
            if language in lower_desc:
                return language
        
        # Default to Python if no language specified
        return "python"