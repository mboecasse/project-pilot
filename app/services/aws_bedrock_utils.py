"""
ProjectPilot - AI-powered project management system
AWS Bedrock integration utilities.
"""

import logging
import json
import os
import boto3
from typing import List, Dict, Any, Optional, Union
from flask import current_app

logger = logging.getLogger(__name__)

class AWSBedrockService:
    """Service for interacting with AWS Bedrock foundational models."""
    
    def __init__(self, 
                 region: Optional[str] = None,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None):
        """
        Initialize the AWS Bedrock service.
        
        Args:
            region: AWS region. If not provided, will use environment variable.
            aws_access_key_id: AWS access key ID. If not provided, will use environment variable.
            aws_secret_access_key: AWS secret access key. If not provided, will use environment variable.
        """
        self.region = region or os.environ.get('AWS_REGION') or current_app.config.get('AWS_REGION', 'eu-west-2')
        self.aws_access_key_id = aws_access_key_id or os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
        
        try:
            # Initialize Bedrock client
            session = boto3.Session(
                region_name=self.region,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            self.client = session.client('bedrock-runtime')
            logger.info(f"AWS Bedrock service initialized in region {self.region}")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Bedrock service is available."""
        return self.client is not None
    
    def list_foundation_models(self) -> List[Dict[str, Any]]:
        """
        List available foundation models in AWS Bedrock.
        
        Returns:
            List of available model information
        """
        if not self.client:
            return []
        
        try:
            # Create a separate bedrock client (not bedrock-runtime)
            session = boto3.Session(
                region_name=self.region,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            bedrock_client = session.client('bedrock')
            
            response = bedrock_client.list_foundation_models()
            return response.get('modelSummaries', [])
        except Exception as e:
            logger.error(f"Error listing foundation models: {str(e)}")
            return []
    
    def generate_text_claude(self, 
                           prompt: str, 
                           model_id: str = "anthropic.claude-3-opus-20240229-v1:0", 
                           temperature: float = 0.7, 
                           max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate text using Claude models via AWS Bedrock.
        
        Args:
            prompt: Text prompt to generate from
            model_id: Claude model ID on Bedrock
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        if not self.client:
            return {"error": "AWS Bedrock client not initialized", "text": ""}
        
        try:
            # Format request body for Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response.get('body').read())
            
            return {
                "text": response_body.get('content', [{}])[0].get('text', ''),
                "model": model_id,
                "usage": {
                    "input_tokens": response_body.get('usage', {}).get('input_tokens', 0),
                    "output_tokens": response_body.get('usage', {}).get('output_tokens', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error calling AWS Bedrock Claude API: {str(e)}")
            return {"error": str(e), "text": ""}
    
    def generate_text_titan(self, 
                           prompt: str, 
                           model_id: str = "amazon.titan-text-express-v1", 
                           temperature: float = 0.7, 
                           max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate text using Amazon Titan models via AWS Bedrock.
        
        Args:
            prompt: Text prompt to generate from
            model_id: Titan model ID on Bedrock
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing the generated text and metadata
        """
        if not self.client:
            return {"error": "AWS Bedrock client not initialized", "text": ""}
        
        try:
            # Format request body for Titan
            request_body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            }
            
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response.get('body').read())
            
            return {
                "text": response_body.get('results', [{}])[0].get('outputText', ''),
                "model": model_id
            }
        except Exception as e:
            logger.error(f"Error calling AWS Bedrock Titan API: {str(e)}")
            return {"error": str(e), "text": ""}
    
    def analyze_project_requirements(self, requirements: str) -> Dict[str, Any]:
        """
        Analyze project requirements and suggest tasks using Bedrock.
        
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
        
        response = self.generate_text_claude(
            prompt, 
            model_id="anthropic.claude-3-opus-20240229-v1:0", 
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
    
    def generate_code(self, 
                     description: str, 
                     language: str,
                     project_context: str) -> Dict[str, Any]:
        """
        Generate code using Bedrock models.
        
        Args:
            description: Description of what the code should do
            language: Programming language for the code
            project_context: Additional context about the project
            
        Returns:
            Dictionary with generated code
        """
        prompt = f"""
        Generate {language} code based on the following requirements:
        
        DESCRIPTION:
        {description}
        
        PROJECT CONTEXT:
        {project_context}
        
        Provide the code with:
        1. Clear commenting
        2. Error handling
        3. Best practices for {language}
        4. Explanation of key design decisions
        
        Format your response as JSON with the following structure:
        {{
            "code": "The complete code implementation",
            "explanation": "Explanation of how the code works",
            "usage_example": "Example of how to use the code",
            "dependencies": ["Dependency 1", "Dependency 2"],
            "considerations": ["Consideration 1", "Consideration 2"]
        }}

        Ensure your response is valid JSON with properly escaped code blocks.
        """
        
        response = self.generate_text_claude(
            prompt, 
            model_id="anthropic.claude-3-opus-20240229-v1:0", 
            temperature=0.3, 
            max_tokens=2500
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