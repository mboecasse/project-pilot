"""
ProjectPilot - AI-powered project management system
Anthropic Claude integration utilities.
"""

import logging
import json
import os
import anthropic
from typing import List, Dict, Any, Optional, Union
from flask import current_app

logger = logging.getLogger(__name__)

class AnthropicService:
    """Service for interacting with Anthropic Claude APIs."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Anthropic service.
        
        Args:
            api_key: Anthropic API key. If not provided, will use environment variable.
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY') or current_app.config.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            logger.warning("Anthropic API key not provided. Service will not function.")
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("Anthropic service initialized")
    
    def generate_text(self, 
                     prompt: str, 
                     model: str = "claude-3-opus-20240229", 
                     temperature: float = 0.7, 
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate text using Anthropic's Claude models.
        
        Args:
            prompt: Text prompt to generate from
            model: Model to use (e.g., claude-3-opus-20240229, claude-3-sonnet-20240229)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        if not self.api_key:
            return {"error": "Anthropic API key not configured", "text": ""}
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "text": response.content[0].text,
                "model": model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            return {"error": str(e), "text": ""}
    
    def analyze_project_requirements(self, requirements: str) -> Dict[str, Any]:
        """
        Analyze project requirements and suggest tasks using Claude.
        
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

        Ensure your response is valid JSON.
        """
        
        response = self.generate_text(
            prompt, 
            model="claude-3-opus-20240229", 
            temperature=0.2, 
            max_tokens=2000
        )
        
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
        Generate implementation details for a specific task using Claude.
        
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

        Ensure your response is valid JSON.
        """
        
        response = self.generate_text(
            prompt, 
            model="claude-3-opus-20240229", 
            temperature=0.3, 
            max_tokens=2000
        )
        
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
    
    def architectural_review(self, 
                             system_design: str, 
                             requirements: str) -> Dict[str, Any]:
        """
        Review system architecture using Claude.
        
        Args:
            system_design: System architecture design text
            requirements: Project requirements
            
        Returns:
            Dictionary with architectural review
        """
        prompt = f"""
        Review the following system architecture design against the project requirements:
        
        SYSTEM DESIGN:
        {system_design}
        
        PROJECT REQUIREMENTS:
        {requirements}
        
        Provide a comprehensive review including:
        1. Alignment with requirements
        2. Scalability assessment
        3. Security considerations
        4. Performance considerations
        5. Potential bottlenecks
        6. Suggested improvements
        
        Format your response as JSON with the following structure:
        {{
            "alignment_score": 0-10,
            "alignment_notes": "Notes on requirement alignment",
            "scalability_score": 0-10,
            "scalability_notes": "Notes on scalability",
            "security_score": 0-10,
            "security_notes": "Notes on security",
            "performance_score": 0-10,
            "performance_notes": "Notes on performance",
            "bottlenecks": ["Bottleneck 1", "Bottleneck 2"],
            "suggested_improvements": ["Improvement 1", "Improvement 2"],
            "overall_assessment": "Overall assessment of the architecture"
        }}

        Ensure your response is valid JSON.
        """
        
        response = self.generate_text(
            prompt, 
            model="claude-3-opus-20240229", 
            temperature=0.2, 
            max_tokens=2000
        )
        
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