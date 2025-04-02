"""
ProjectPilot - AI-powered project management system
OpenAI integration utilities.
"""

import logging
import json
import os
import openai
from typing import List, Dict, Any, Optional, Union
from flask import current_app

logger = logging.getLogger(__name__)

class OpenAIService:
    """Service for interacting with OpenAI APIs."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI service.
        
        Args:
            api_key: OpenAI API key. If not provided, will use environment variable.
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY') or current_app.config.get('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Service will not function.")
        else:
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info("OpenAI service initialized")
    
    def generate_text(self, 
                      prompt: str, 
                      model: str = "gpt-4", 
                      temperature: float = 0.7, 
                      max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate text using OpenAI's models.
        
        Args:
            prompt: Text prompt to generate from
            model: Model to use (e.g., gpt-4, gpt-3.5-turbo)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        if not self.api_key:
            return {"error": "OpenAI API key not configured", "text": ""}
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "text": response.choices[0].message.content,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return {"error": str(e), "text": ""}
    
    def analyze_project_requirements(self, requirements: str) -> Dict[str, Any]:
        """
        Analyze project requirements and suggest tasks.
        
        Args:
            requirements: Project requirements text
            
        Returns:
            Dictionary with analyzed requirements and suggested tasks
        """
        prompt = f"""
        Analyze the following project requirements and decompose them into specific, 
        actionable tasks appropriate for an AI-driven development team:

        REQUIREMENTS:
        {requirements}

        For each task, provide:
        1. Task name
        2. Task description
        3. Task type (e.g., "feature", "infrastructure", "design", "documentation")
        4. Priority (1-4 where 1=low, 4=critical)
        5. Estimated complexity (1-3 where 1=simple, 3=complex)
        6. Dependencies (what other tasks must be completed first)

        Format your response as JSON following this structure:
        {{
            "tasks": [
                {{
                    "name": "Task name",
                    "description": "Task description",
                    "type": "Task type",
                    "priority": 1-4,
                    "complexity": 1-3,
                    "dependencies": ["Name of dependent task", "Name of another dependent task"]
                }}
            ],
            "suggested_architecture": "High-level description of suggested architecture",
            "risks": ["Potential risk 1", "Potential risk 2"],
            "timeline_estimate": "Estimated timeline in weeks"
        }}
        """
        
        response = self.generate_text(prompt, model="gpt-4", temperature=0.2, max_tokens=2000)
        
        if "error" in response:
            return {"error": response["error"], "tasks": []}
        
        try:
            # Extract JSON from response
            result_text = response["text"]
            # Find JSON content if embedded in other text
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = result_text[json_start:json_end]
                return json.loads(json_content)
            else:
                return {"error": "No valid JSON found in response", "tasks": []}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return {"error": "Failed to parse AI response", "text": response["text"], "tasks": []}
    
    def generate_task_implementation(self, 
                                    task_description: str, 
                                    project_context: str) -> Dict[str, Any]:
        """
        Generate implementation details for a specific task.
        
        Args:
            task_description: Description of the task
            project_context: Broader context about the project
            
        Returns:
            Dictionary with implementation details
        """
        prompt = f"""
        I need a detailed implementation plan for the following task in my project:
        
        TASK: {task_description}
        
        PROJECT CONTEXT: {project_context}
        
        Provide a detailed implementation plan including:
        1. Step-by-step approach
        2. Key components or files that need to be created or modified
        3. Any libraries or technologies that should be used
        4. Potential challenges and how to address them
        5. Testing strategies
        
        Format your response as JSON with the following structure:
        {{
            "implementation_steps": [
                {{
                    "step": "Step description",
                    "details": "Implementation details"
                }}
            ],
            "components": ["Component 1", "Component 2"],
            "technologies": ["Technology 1", "Technology 2"],
            "challenges": ["Challenge 1", "Challenge 2"],
            "testing": ["Test approach 1", "Test approach 2"]
        }}
        """
        
        response = self.generate_text(prompt, model="gpt-4", temperature=0.3, max_tokens=2000)
        
        if "error" in response:
            return {"error": response["error"]}
        
        try:
            # Extract JSON from response
            result_text = response["text"]
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = result_text[json_start:json_end]
                return json.loads(json_content)
            else:
                return {"error": "No valid JSON found in response", "text": response["text"]}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return {"error": "Failed to parse AI response", "text": response["text"]}
    
    def review_code(self, code: str, language: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """
        Review code and provide feedback.
        
        Args:
            code: Code to review
            language: Programming language
            focus_areas: Specific areas to focus the review on
            
        Returns:
            Dictionary with review comments
        """
        focus_areas_str = ", ".join(focus_areas) if focus_areas else "general quality, bugs, best practices"
        
        prompt = f"""
        Review the following {language} code focusing on {focus_areas_str}:
        
        ```{language}
        {code}
        ```
        
        Provide your feedback as JSON with the following structure:
        {{
            "issues": [
                {{
                    "line": "Line number or range",
                    "severity": "high|medium|low",
                    "description": "Issue description",
                    "suggestion": "Suggested fix"
                }}
            ],
            "overall_assessment": "Overall code quality assessment",
            "improvements": ["Improvement 1", "Improvement 2"],
            "strengths": ["Strength 1", "Strength 2"]
        }}
        """
        
        response = self.generate_text(prompt, model="gpt-4", temperature=0.2, max_tokens=2000)
        
        if "error" in response:
            return {"error": response["error"]}
        
        try:
            # Extract JSON from response
            result_text = response["text"]
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = result_text[json_start:json_end]
                return json.loads(json_content)
            else:
                return {"error": "No valid JSON found in response", "text": response["text"]}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            return {"error": "Failed to parse AI response", "text": response["text"]}