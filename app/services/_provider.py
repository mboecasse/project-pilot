"""
ProjectPilot - AI-powered project management system
Unified AI provider interface with circuit breaker and fallback mechanisms.
"""

import logging
import json
import os
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable
import re
import threading
from flask import current_app

# Import AI service providers
from app.services.openai_utils import OpenAIService
from app.services.anthropic_utils import AnthropicService
from app.services.aws_bedrock_utils import AWSBedrockService

# Planned future providers - commented out until implemented
# from app.services.google_gemini_utils import GeminiService
# from app.services.azure_openai_utils import AzureOpenAIService

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern implementation to prevent repeated failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_failure_time = None
        self.lock = threading.RLock()
    
    def record_success(self) -> None:
        """Record a successful operation."""
        with self.lock:
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                
            self.failures = 0
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        with self.lock:
            self.failures += 1
            self.last_failure_time = datetime.now()
            
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
    
    def is_closed(self) -> bool:
        """Check if the circuit is closed or can be attempted."""
        with self.lock:
            if self.state == "CLOSED":
                return True
            
            if self.state == "OPEN":
                # Check if recovery timeout has elapsed
                if self.last_failure_time and \
                   (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout:
                    self.state = "HALF-OPEN"
                    return True
                
                return False
            
            # HALF-OPEN state allows one attempt
            return True
    
    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self.lock:
            self.failures = 0
            self.state = "CLOSED"
            self.last_failure_time = None

class TokenBudget:
    """Token usage tracking and budget management."""
    
    def __init__(self, daily_budget: int = 1000000):
        """
        Initialize token budget tracker.
        
        Args:
            daily_budget: Maximum tokens to use per day
        """
        self.daily_budget = daily_budget
        self.usage = {}
        self.lock = threading.RLock()
    
    def record_usage(self, provider: str, tokens: int) -> None:
        """
        Record token usage for a provider.
        
        Args:
            provider: AI provider name
            tokens: Number of tokens used
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        with self.lock:
            if today not in self.usage:
                self.usage[today] = {}
                
            if provider not in self.usage[today]:
                self.usage[today][provider] = 0
                
            self.usage[today][provider] += tokens
    
    def get_usage_today(self) -> Dict[str, int]:
        """
        Get token usage for today.
        
        Returns:
            Dictionary of provider -> tokens used
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        with self.lock:
            return self.usage.get(today, {}).copy()
    
    def get_total_usage_today(self) -> int:
        """
        Get total token usage for today.
        
        Returns:
            Total tokens used today
        """
        usage = self.get_usage_today()
        return sum(usage.values())
    
    def is_within_budget(self) -> bool:
        """
        Check if usage is within the daily budget.
        
        Returns:
            True if usage is within budget
        """
        return self.get_total_usage_today() < self.daily_budget
    
    def get_remaining_budget(self) -> int:
        """
        Get remaining token budget for today.
        
        Returns:
            Remaining token budget
        """
        return max(0, self.daily_budget - self.get_total_usage_today())

class AIProvider:
    """
    Unified interface for multiple AI services with circuit breaker
    pattern, token management, and fallback capabilities.
    """
    
    def __init__(self,
                openai_api_key: Optional[str] = None,
                anthropic_api_key: Optional[str] = None,
                aws_region: Optional[str] = None,
                aws_access_key_id: Optional[str] = None,
                aws_secret_access_key: Optional[str] = None,
                daily_token_budget: int = 1000000):
        """
        Initialize the AI provider.
        
        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            aws_region: AWS region for Bedrock
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            daily_token_budget: Maximum tokens to use per day
        """
        # Initialize all service providers
        self.openai_service = OpenAIService(api_key=openai_api_key)
        self.anthropic_service = AnthropicService(api_key=anthropic_api_key)
        self.bedrock_service = AWSBedrockService(
            region=aws_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # Future providers - commented out until implemented
        # self.gemini_service = None  # To be implemented
        # self.azure_openai_service = None  # To be implemented
        
        # Track available providers
        self.providers = {
            'openai': bool(openai_api_key or os.environ.get('OPENAI_API_KEY') or 
                          current_app.config.get('OPENAI_API_KEY')),
            'anthropic': bool(anthropic_api_key or os.environ.get('ANTHROPIC_API_KEY') or 
                             current_app.config.get('ANTHROPIC_API_KEY')),
            'bedrock': self.bedrock_service.is_available(),
            # 'gemini': False,  # To be implemented
            # 'azure_openai': False  # To be implemented
        }
        
        # Circuit breakers for each provider
        self.circuit_breakers = {
            provider: CircuitBreaker() for provider in self.providers
        }
        
        # Token budget tracker
        self.token_budget = TokenBudget(daily_budget=daily_token_budget)
        
        # Provider preference order (fallback sequence)
        self.preference_order = ['openai', 'anthropic', 'bedrock']  # Add others as implemented
        
        # Model mappings between providers for fallback
        self.model_mappings = {
            'openai': {
                'gpt-4': {'anthropic': 'claude-3-opus-20240229', 'bedrock': 'anthropic.claude-3-opus-20240229-v1:0'},
                'gpt-3.5-turbo': {'anthropic': 'claude-3-sonnet-20240229', 'bedrock': 'anthropic.claude-3-sonnet-20240229-v1:0'},
            },
            'anthropic': {
                'claude-3-opus-20240229': {'openai': 'gpt-4', 'bedrock': 'anthropic.claude-3-opus-20240229-v1:0'},
                'claude-3-sonnet-20240229': {'openai': 'gpt-3.5-turbo', 'bedrock': 'anthropic.claude-3-sonnet-20240229-v1:0'},
                'claude-3-haiku-20240307': {'openai': 'gpt-3.5-turbo', 'bedrock': 'anthropic.claude-3-haiku-20240307-v1:0'},
            },
            'bedrock': {
                'anthropic.claude-3-opus-20240229-v1:0': {'openai': 'gpt-4', 'anthropic': 'claude-3-opus-20240229'},
                'anthropic.claude-3-sonnet-20240229-v1:0': {'openai': 'gpt-3.5-turbo', 'anthropic': 'claude-3-sonnet-20240229'},
                'amazon.titan-text-express-v1': {'openai': 'gpt-3.5-turbo', 'anthropic': 'claude-3-haiku-20240307'},
            }
        }
        
        # Performance metrics
        self.performance_metrics = {}
        
        logger.info(f"AI Provider initialized with available providers: " + 
                   f"{', '.join(p for p, available in self.providers.items() if available)}")
    
    def generate_text(self, 
                     prompt: str,
                     provider: str = 'auto',
                     model: Optional[str] = None,
                     temperature: float = 0.7,
                     max_tokens: int = 1000,
                     retry_count: int = 2,
                     fallback: bool = True) -> Dict[str, Any]:
        """
        Generate text using the specified provider or the best available provider.
        
        Args:
            prompt: Text prompt
            provider: AI provider ('openai', 'anthropic', 'bedrock', 'auto')
            model: Model name for the provider (or None for default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            retry_count: Number of retries
            fallback: Whether to fallback to other providers on failure
            
        Returns:
            Dictionary with generated text and metadata
        """
        start_time = time.time()
        
        # Check if we're within budget
        if not self.token_budget.is_within_budget():
            return {
                "error": "Daily token budget exceeded",
                "text": "",
                "provider": provider,
                "model": model,
                "within_budget": False
            }
        
        # Track original request
        original_provider = provider
        original_model = model
        
        # Default models if not specified
        default_models = {
            'openai': 'gpt-4',
            'anthropic': 'claude-3-opus-20240229',
            'bedrock': 'anthropic.claude-3-opus-20240229-v1:0'
        }
        
        # If auto, select the first available provider
        if provider == 'auto':
            for p in self.preference_order:
                if self.providers.get(p, False) and self.circuit_breakers[p].is_closed():
                    provider = p
                    break
            
            # If no providers available, return error
            if provider == 'auto':
                return {
                    "error": "No AI providers available",
                    "text": "",
                    "provider": provider,
                }
        
        # If provider not available, fallback or error
        if not self.providers.get(provider, False):
            if fallback:
                for p in self.preference_order:
                    if p != provider and self.providers.get(p, False) and self.circuit_breakers[p].is_closed():
                        logger.info(f"Provider {provider} not available, falling back to {p}")
                        provider = p
                        break
            
            if not self.providers.get(provider, False):
                return {
                    "error": f"Provider {provider} not available and no fallbacks found",
                    "text": "",
                    "provider": original_provider,
                }
        
        # Use default model if not specified
        if not model:
            model = default_models.get(provider, '')
        
        # Generate text with retry loop
        for attempt in range(retry_count + 1):
            # Check if circuit breaker is closed
            if not self.circuit_breakers[provider].is_closed():
                logger.warning(f"Circuit breaker open for {provider}, attempting fallback")
                if fallback:
                    # Try to find an alternative provider
                    fallback_provider = None
                    for p in self.preference_order:
                        if p != provider and self.providers.get(p, False) and self.circuit_breakers[p].is_closed():
                            fallback_provider = p
                            
                            # Map the model to equivalent in fallback provider
                            provider_mappings = self.model_mappings.get(provider, {}).get(model, {})
                            model = provider_mappings.get(fallback_provider, default_models.get(fallback_provider))
                            
                            logger.info(f"Falling back to {fallback_provider} with model {model}")
                            provider = fallback_provider
                            break
                    
                    if not fallback_provider:
                        return {
                            "error": "All providers circuit breakers open",
                            "text": "",
                            "provider": original_provider,
                            "model": original_model,
                        }
                else:
                    return {
                        "error": f"Circuit breaker open for {provider}",
                        "text": "",
                        "provider": provider,
                        "model": model,
                    }
            
            try:
                # Generate text using selected provider
                if provider == 'openai':
                    response = self.openai_service.generate_text(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                elif provider == 'anthropic':
                    response = self.anthropic_service.generate_text(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                elif provider == 'bedrock':
                    if 'claude' in model.lower():
                        response = self.bedrock_service.generate_text_claude(
                            prompt=prompt,
                            model_id=model,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    else:
                        response = self.bedrock_service.generate_text_titan(
                            prompt=prompt,
                            model_id=model,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                else:
                    return {
                        "error": f"Unknown provider: {provider}",
                        "text": "",
                        "provider": provider,
                        "model": model,
                    }
                
                # Check for errors in response
                if "error" in response:
                    logger.warning(f"Error from {provider}: {response['error']}")
                    self.circuit_breakers[provider].record_failure()
                    
                    if attempt < retry_count:
                        logger.info(f"Retrying with {provider}, attempt {attempt+1} of {retry_count}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    
                    if fallback and attempt >= retry_count:
                        # Try to find an alternative provider
                        fallback_provider = None
                        for p in self.preference_order:
                            if p != provider and self.providers.get(p, False) and self.circuit_breakers[p].is_closed():
                                fallback_provider = p
                                
                                # Map the model to equivalent in fallback provider
                                provider_mappings = self.model_mappings.get(provider, {}).get(model, {})
                                model = provider_mappings.get(fallback_provider, default_models.get(fallback_provider))
                                
                                logger.info(f"Falling back to {fallback_provider} with model {model}")
                                provider = fallback_provider
                                break
                        
                        if fallback_provider:
                            continue  # Try with the fallback provider
                    
                    return response
                
                # Calculate token usage
                usage = response.get('usage', {})
                total_tokens = usage.get('total_tokens', 0)
                if not total_tokens:
                    # Try to calculate from input/output tokens
                    input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
                    total_tokens = input_tokens + output_tokens
                
                # Record token usage
                self.token_budget.record_usage(provider, total_tokens)
                
                # Record successful completion
                self.circuit_breakers[provider].record_success()
                
                # Record performance metrics
                elapsed_time = time.time() - start_time
                self._record_performance(provider, model, elapsed_time, total_tokens, prompt)
                
                # Add additional metadata
                response.update({
                    'provider': provider,
                    'model': model,
                    'elapsed_time': elapsed_time,
                    'original_provider': original_provider,
                    'original_model': original_model,
                    'within_budget': True,
                    'remaining_budget': self.token_budget.get_remaining_budget()
                })
                
                return response
                
            except Exception as e:
                logger.error(f"Exception calling {provider}: {str(e)}")
                self.circuit_breakers[provider].record_failure()
                
                if attempt < retry_count:
                    logger.info(f"Retrying with {provider}, attempt {attempt+1} of {retry_count}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                if fallback and attempt >= retry_count:
                    # Try to find an alternative provider
                    fallback_provider = None
                    for p in self.preference_order:
                        if p != provider and self.providers.get(p, False) and self.circuit_breakers[p].is_closed():
                            fallback_provider = p
                            
                            # Map the model to equivalent in fallback provider
                            provider_mappings = self.model_mappings.get(provider, {}).get(model, {})
                            model = provider_mappings.get(fallback_provider, default_models.get(fallback_provider))
                            
                            logger.info(f"Falling back to {fallback_provider} with model {model}")
                            provider = fallback_provider
                            break
                    
                    if fallback_provider:
                        continue  # Try with the fallback provider
                
                return {
                    "error": f"Failed to generate text: {str(e)}",
                    "text": "",
                    "provider": provider,
                    "model": model,
                }
        
        # Should never reach here, but just in case
        return {
            "error": "Exhausted all retries and fallbacks",
            "text": "",
            "provider": provider,
            "model": model,
        }
    
    def structured_generate(self, 
                          prompt: str, 
                          output_schema: Dict[str, Any],
                          provider: str = 'auto',
                          model: Optional[str] = None,
                          temperature: float = 0.2,
                          max_tokens: int = 2000,
                          retry_count: int = 2,
                          fallback: bool = True) -> Dict[str, Any]:
        """
        Generate structured output according to a schema.
        
        Args:
            prompt: Text prompt
            output_schema: JSON schema defining the expected output structure
            provider: AI provider ('openai', 'anthropic', 'bedrock', 'auto')
            model: Model name for the provider (or None for default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            retry_count: Number of retries
            fallback: Whether to fallback to other providers on failure
            
        Returns:
            Dictionary with generated structured output and metadata
        """
        schema_str = json.dumps(output_schema, indent=2)
        
        enhanced_prompt = f"""
        {prompt}

        FORMAT YOUR RESPONSE AS VALID JSON MATCHING THIS SCHEMA:
        {schema_str}

        The response must be valid JSON that conforms exactly to the schema above.
        Do not include any explanations, notes, or anything else outside the JSON.
        """
        
        response = self.generate_text(
            prompt=enhanced_prompt,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            retry_count=retry_count,
            fallback=fallback
        )
        
        if "error" in response:
            return response
        
        # Try to parse the response as JSON
        result_text = response.get("text", "")
        
        # Extract JSON from the response if needed
        json_content = self._extract_json(result_text)
        
        try:
            parsed_result = json.loads(json_content)
            
            # Add the parsed result to the response
            response["parsed_result"] = parsed_result
            
            return response
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response as JSON: {str(e)}")
            return {
                "error": f"Failed to parse response as JSON: {str(e)}",
                "text": result_text,
                "provider": response.get("provider"),
                "model": response.get("model"),
            }
    
    def analyze_code(self, 
                    code: str, 
                    language: str, 
                    analysis_type: str = 'general',
                    provider: str = 'auto',
                    model: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze code for bugs, improvements, etc.
        
        Args:
            code: Code to analyze
            language: Programming language
            analysis_type: Type of analysis ('general', 'security', 'performance', 'style')
            provider: AI provider ('openai', 'anthropic', 'bedrock', 'auto')
            model: Model name for the provider (or None for default)
            
        Returns:
            Dictionary with analysis results
        """
        analysis_prompts = {
            'general': f"Analyze this {language} code and identify any bugs, best practices violations, or efficiency improvements:",
            'security': f"Analyze this {language} code for security vulnerabilities and suggest fixes:",
            'performance': f"Analyze this {language} code for performance improvements:",
            'style': f"Analyze this {language} code for style improvements and adherence to {language} best practices:"
        }
        
        prompt = f"""
        {analysis_prompts.get(analysis_type, analysis_prompts['general'])}

        ```{language}
        {code}
        ```
        
        Provide your analysis as JSON with the following structure:
        {{
            "summary": "Brief summary of the code and major findings",
            "issues": [
                {{
                    "line": "Line number or range",
                    "severity": "high|medium|low",
                    "type": "bug|security|performance|style",
                    "description": "Description of the issue",
                    "suggestion": "Suggested fix with code example"
                }}
            ],
            "improvements": [
                {{
                    "description": "Description of the improvement",
                    "suggested_code": "Suggested improved code"
                }}
            ],
            "overall_assessment": "Overall assessment of the code quality",
            "metrics": {{
                "complexity": "1-10 score",
                "maintainability": "1-10 score",
                "security": "1-10 score"
            }}
        }}
        """
        
        output_schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {"type": "string"},
                            "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                            "suggestion": {"type": "string"}
                        }
                    }
                },
                "improvements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "suggested_code": {"type": "string"}
                        }
                    }
                },
                "overall_assessment": {"type": "string"},
                "metrics": {
                    "type": "object",
                    "properties": {
                        "complexity": {"type": "string"},
                        "maintainability": {"type": "string"},
                        "security": {"type": "string"}
                    }
                }
            }
        }
        
        return self.structured_generate(
            prompt=prompt,
            output_schema=output_schema,
            provider=provider,
            model=model,
            temperature=0.2
        )
    
    def get_provider_status(self) -> Dict[str, Any]:
        """
        Get status information for all providers.
        
        Returns:
            Dictionary with provider status information
        """
        status = {}
        
        for provider in self.providers:
            if not self.providers[provider]:
                status[provider] = {
                    "available": False,
                    "circuit_state": "N/A",
                    "message": "Provider not configured"
                }
            else:
                circuit = self.circuit_breakers[provider]
                status[provider] = {
                    "available": True,
                    "circuit_state": circuit.state,
                    "failures": circuit.failures,
                    "last_failure": circuit.last_failure_time.isoformat() if circuit.last_failure_time else None,
                    "message": "Ready" if circuit.state == "CLOSED" else
                               "Recovering" if circuit.state == "HALF-OPEN" else
                               "Unavailable due to failures"
                }
                
                # Add performance metrics if available
                if provider in self.performance_metrics:
                    status[provider].update({
                        "avg_response_time": self._calculate_average(
                            [m["elapsed_time"] for m in self.performance_metrics[provider]]
                        ),
                        "calls_count": len(self.performance_metrics[provider]),
                        "success_rate": self._calculate_success_rate(self.performance_metrics[provider])
                    })
        
        # Add token budget information
        status["token_budget"] = {
            "total_budget": self.token_budget.daily_budget,
            "used_today": self.token_budget.get_total_usage_today(),
            "remaining": self.token_budget.get_remaining_budget(),
            "usage_by_provider": self.token_budget.get_usage_today()
        }
        
        return status
    
    def reset_circuit_breaker(self, provider: str) -> Dict[str, Any]:
        """
        Reset the circuit breaker for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary with reset result
        """
        if provider not in self.circuit_breakers:
            return {
                "success": False,
                "message": f"Provider {provider} not found"
            }
        
        self.circuit_breakers[provider].reset()
        
        return {
            "success": True,
            "provider": provider,
            "message": f"Circuit breaker for {provider} reset",
            "new_state": "CLOSED"
        }
    
    def _record_performance(self, 
                          provider: str, 
                          model: str, 
                          elapsed_time: float,
                          tokens: int,
                          prompt: str) -> None:
        """
        Record performance metrics for a provider.
        
        Args:
            provider: Provider name
            model: Model name
            elapsed_time: Request time in seconds
            tokens: Total tokens used
            prompt: The prompt that was used
        """
        if provider not in self.performance_metrics:
            self.performance_metrics[provider] = []
        
        # Limit history length
        max_history = 100
        if len(self.performance_metrics[provider]) >= max_history:
            self.performance_metrics[provider] = self.performance_metrics[provider][-max_history+1:]
        
        # Estimate prompt complexity
        complexity = self._estimate_prompt_complexity(prompt)
        
        # Record metrics
        self.performance_metrics[provider].append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "elapsed_time": elapsed_time,
            "tokens": tokens,
            "prompt_length": len(prompt),
            "prompt_complexity": complexity,
            "tokens_per_second": tokens / elapsed_time if elapsed_time > 0 else 0,
            "success": True
        })
    
    def _calculate_average(self, values: List[float]) -> float:
        """
        Calculate average of a list of values.
        
        Args:
            values: List of values
            
        Returns:
            Average value or 0 if list is empty
        """
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def _calculate_success_rate(self, metrics: List[Dict[str, Any]]) -> float:
        """
        Calculate success rate from metrics.
        
        Args:
            metrics: List of metric dictionaries
            
        Returns:
            Success rate as a percentage
        """
        if not metrics:
            return 0.0
        
        success_count = sum(1 for m in metrics if m.get("success", False))
        return (success_count / len(metrics)) * 100
    
    def _estimate_prompt_complexity(self, prompt: str) -> int:
        """
        Estimate the complexity of a prompt on a scale of 1-10.
        
        Args:
            prompt: The prompt text
            
        Returns:
            Complexity estimate (1-10)
        """
        # Simple heuristic based on length, structure, and special tokens
        length_factor = min(10, len(prompt) / 500)  # Length factor (max 10)
        
        # Structure factors
        has_code = "```" in prompt
        has_json = "{" in prompt and "}" in prompt
        has_lists = re.search(r'\n\s*[-*]\s+', prompt) is not None
        has_headings = re.search(r'\n\s*#{1,6}\s+', prompt) is not None
        
        structure_count = sum([has_code, has_json, has_lists, has_headings])
        structure_factor = min(5, structure_count * 1.25)
        
        # Instruction clarity
        instruction_words = ["write", "create", "list", "explain", "analyze", "generate", "summarize"]
        has_clear_instructions = any(word in prompt.lower() for word in instruction_words)
        instruction_factor = 1 if has_clear_instructions else 3
        
        # Combined score
        complexity = (length_factor * 0.5) + (structure_factor * 0.3) + instruction_factor
        return min(10, max(1, round(complexity)))
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text, handling cases where JSON is embedded in other text.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON string
        """
        # Try to find JSON block delimited by ``` markers
        json_block_match = re.search(r'```(?:json)?\s*(.+?)```', text, re.DOTALL)
        if json_block_match:
            return json_block_match.group(1).strip()
        
        # Try to find content between outermost braces
        brace_match = re.search(r'(\{.+\})', text, re.DOTALL)
        if brace_match:
            return brace_match.group(1).strip()
            
        # Fall back to the original text
        return text.strip()
    
    def sanitize_prompt(self, prompt: str, provider: str) -> str:
        """
        Sanitize a prompt for a specific provider to avoid known issues.
        
        Args:
            prompt: Original prompt
            provider: Target provider
            
        Returns:
            Sanitized prompt
        """
        if provider == 'anthropic':
            # Claude sometimes has issues with specific formats
            # Remove any "<", ">" XML/HTML-like tags that aren't in code blocks
            def replace_outside_code_blocks(text, pattern, replacement):
                # Split by code blocks
                parts = re.split(r'(```.*?```)', text, flags=re.DOTALL)
                for i in range(0, len(parts), 2):  # Every other part is outside code blocks
                    parts[i] = re.sub(pattern, replacement, parts[i])
                return ''.join(parts)
            
            # Replace XML/HTML tags outside code blocks with their entity references
            prompt = replace_outside_code_blocks(prompt, r'<', '&lt;')
            prompt = replace_outside_code_blocks(prompt, r'>', '&gt;')
            
        elif provider == 'openai':
            # OpenAI has issues with certain Unicode characters
            # Replace zero-width spaces with regular spaces
            prompt = prompt.replace('\u200b', ' ').replace('\u200c', ' ')
            
        return prompt
    
    def sanitize_response(self, response: str, provider: str) -> str:
        """
        Sanitize a response from a specific provider.
        
        Args:
            response: Original response
            provider: Source provider
            
        Returns:
            Sanitized response
        """
        if provider == 'anthropic':
            # Claude sometimes adds "I'll help you..." preambles before JSON
            json_start = response.find('{')
            if json_start > 0:
                # Check if it seems to be introducing JSON
                preamble = response[:json_start].lower()
                if ('json' in preamble or 'here' in preamble) and ('{' in response and '}' in response):
                    response = response[json_start:]
            
        elif provider == 'openai':
            # OpenAI sometimes adds markdown formatting
            if response.startswith('```json') and response.endswith('```'):
                response = response[7:-3].strip()
                
        return response