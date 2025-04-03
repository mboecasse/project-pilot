"""
Test file for the AIProvider component.
"""

import os
import sys
import pytest
import json
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta
import threading

# Import the module to test
from app.services._provider import AIProvider, CircuitBreaker, TokenBudget

class TestCircuitBreaker:
    """Test the CircuitBreaker class."""
    
    def test_init(self):
        """Test initialization of CircuitBreaker."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30
        assert cb.failures == 0
        assert cb.state == "CLOSED"
        assert cb.last_failure_time is None
        assert isinstance(cb.lock, threading.RLock)
    
    def test_record_success(self):
        """Test recording a successful operation."""
        cb = CircuitBreaker()
        
        # Set some failures and state
        cb.failures = 2
        cb.state = "HALF-OPEN"
        
        # Record success
        cb.record_success()
        
        # Check that state was reset
        assert cb.failures == 0
        assert cb.state == "CLOSED"
    
    def test_record_failure(self):
        """Test recording a failed operation."""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Record failures
        cb.record_failure()
        assert cb.failures == 1
        assert cb.state == "CLOSED"
        assert cb.last_failure_time is not None
        
        cb.record_failure()
        assert cb.failures == 2
        assert cb.state == "CLOSED"
        
        # This should trip the circuit breaker
        cb.record_failure()
        assert cb.failures == 3
        assert cb.state == "OPEN"
    
    def test_is_closed_when_closed(self):
        """Test is_closed when state is CLOSED."""
        cb = CircuitBreaker()
        cb.state = "CLOSED"
        
        assert cb.is_closed() is True
    
    def test_is_closed_when_open_before_timeout(self):
        """Test is_closed when state is OPEN and timeout hasn't elapsed."""
        cb = CircuitBreaker(recovery_timeout=60)
        cb.state = "OPEN"
        cb.last_failure_time = datetime.now()
        
        assert cb.is_closed() is False
    
    def test_is_closed_when_open_after_timeout(self):
        """Test is_closed when state is OPEN and timeout has elapsed."""
        cb = CircuitBreaker(recovery_timeout=1)
        cb.state = "OPEN"
        cb.last_failure_time = datetime.now() - timedelta(seconds=2)
        
        assert cb.is_closed() is True
        assert cb.state == "HALF-OPEN"
    
    def test_is_closed_when_half_open(self):
        """Test is_closed when state is HALF-OPEN."""
        cb = CircuitBreaker()
        cb.state = "HALF-OPEN"
        
        assert cb.is_closed() is True
    
    def test_reset(self):
        """Test resetting the circuit breaker."""
        cb = CircuitBreaker()
        
        # Set some state
        cb.failures = 3
        cb.state = "OPEN"
        cb.last_failure_time = datetime.now()
        
        # Reset
        cb.reset()
        
        # Check that state was reset
        assert cb.failures == 0
        assert cb.state == "CLOSED"
        assert cb.last_failure_time is None

class TestTokenBudget:
    """Test the TokenBudget class."""
    
    def test_init(self):
        """Test initialization of TokenBudget."""
        tb = TokenBudget(daily_budget=500000)
        
        assert tb.daily_budget == 500000
        assert tb.usage == {}
        assert isinstance(tb.lock, threading.RLock)
    
    def test_record_usage(self):
        """Test recording token usage."""
        tb = TokenBudget()
        
        with patch('app.services._provider.datetime') as mock_datetime:
            # Set up mock datetime to return a fixed date
            mock_datetime.now.return_value.strftime.return_value = "2023-01-01"
            
            # Record usage
            tb.record_usage("openai", 100)
            
            # Check that usage was recorded
            assert tb.usage == {"2023-01-01": {"openai": 100}}
            
            # Record more usage
            tb.record_usage("openai", 50)
            tb.record_usage("anthropic", 200)
            
            # Check that usage was accumulated
            assert tb.usage == {"2023-01-01": {"openai": 150, "anthropic": 200}}
    
    def test_get_usage_today(self):
        """Test getting usage for today."""
        tb = TokenBudget()
        
        with patch('app.services._provider.datetime') as mock_datetime:
            # Set up mock datetime to return a fixed date
            mock_datetime.now.return_value.strftime.return_value = "2023-01-01"
            
            # Record usage
            tb.record_usage("openai", 100)
            tb.record_usage("anthropic", 200)
            
            # Get usage for today
            usage = tb.get_usage_today()
            
            # Check that usage was returned
            assert usage == {"openai": 100, "anthropic": 200}
            
            # Ensure that a copy was returned
            usage["openai"] = 999
            assert tb.usage["2023-01-01"]["openai"] == 100
    
    def test_get_total_usage_today(self):
        """Test getting total usage for today."""
        tb = TokenBudget()
        
        with patch('app.services._provider.datetime') as mock_datetime:
            # Set up mock datetime to return a fixed date
            mock_datetime.now.return_value.strftime.return_value = "2023-01-01"
            
            # Record usage
            tb.record_usage("openai", 100)
            tb.record_usage("anthropic", 200)
            
            # Get total usage for today
            total = tb.get_total_usage_today()
            
            # Check that total was calculated correctly
            assert total == 300
    
    def test_is_within_budget(self):
        """Test checking if usage is within budget."""
        tb = TokenBudget(daily_budget=500)
        
        with patch('app.services._provider.datetime') as mock_datetime:
            # Set up mock datetime to return a fixed date
            mock_datetime.now.return_value.strftime.return_value = "2023-01-01"
            
            # Record usage within budget
            tb.record_usage("openai", 200)
            assert tb.is_within_budget() is True
            
            # Record usage exceeding budget
            tb.record_usage("anthropic", 400)
            assert tb.is_within_budget() is False
    
    def test_get_remaining_budget(self):
        """Test getting remaining budget."""
        tb = TokenBudget(daily_budget=1000)
        
        with patch('app.services._provider.datetime') as mock_datetime:
            # Set up mock datetime to return a fixed date
            mock_datetime.now.return_value.strftime.return_value = "2023-01-01"
            
            # Record usage
            tb.record_usage("openai", 300)
            tb.record_usage("anthropic", 200)
            
            # Get remaining budget
            remaining = tb.get_remaining_budget()
            
            # Check that remaining budget was calculated correctly
            assert remaining == 500
            
            # Record usage exceeding budget
            tb.record_usage("bedrock", 800)
            assert tb.get_remaining_budget() == 0

class TestAIProvider:
    """Test the AIProvider class."""
    
    @pytest.fixture
    def mock_services(self):
        """Mock the AI service providers."""
        with patch('app.services._provider.OpenAIService') as mock_openai, \
             patch('app.services._provider.AnthropicService') as mock_anthropic, \
             patch('app.services._provider.AWSBedrockService') as mock_bedrock, \
             patch('app.services._provider.current_app') as mock_app:
            
            # Set up mock OpenAI service
            mock_openai_instance = MagicMock()
            mock_openai_instance.generate_text.return_value = {
                "text": "OpenAI response",
                "usage": {"total_tokens": 100}
            }
            mock_openai.return_value = mock_openai_instance
            
            # Set up mock Anthropic service
            mock_anthropic_instance = MagicMock()
            mock_anthropic_instance.generate_text.return_value = {
                "text": "Anthropic response",
                "usage": {"input_tokens": 50, "output_tokens": 75}
            }
            mock_anthropic.return_value = mock_anthropic_instance
            
            # Set up mock Bedrock service
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.is_available.return_value = True
            mock_bedrock_instance.generate_text_claude.return_value = {
                "text": "Bedrock Claude response",
                "usage": {"input_tokens": 40, "output_tokens": 60}
            }
            mock_bedrock_instance.generate_text_titan.return_value = {
                "text": "Bedrock Titan response",
                "usage": {"input_tokens": 30, "output_tokens": 50}
            }
            mock_bedrock.return_value = mock_bedrock_instance
            
            # Set up mock app
            mock_app.config = {}
            
            yield {
                "openai": mock_openai_instance,
                "anthropic": mock_anthropic_instance,
                "bedrock": mock_bedrock_instance,
                "app": mock_app
            }
    
    def test_init(self, mock_services):
        """Test initialization of AIProvider."""
        with patch('app.services._provider.os') as mock_os:
            mock_os.environ.get.return_value = None
            
            provider = AIProvider(
                openai_api_key="openai_key",
                anthropic_api_key="anthropic_key",
                aws_region="us-west-2",
                aws_access_key_id="aws_id",
                aws_secret_access_key="aws_secret"
            )
            
            # Check that services were initialized
            assert provider.openai_service == mock_services["openai"]
            assert provider.anthropic_service == mock_services["anthropic"]
            assert provider.bedrock_service == mock_services["bedrock"]
            
            # Check that providers were detected
            assert provider.providers["openai"] is True
            assert provider.providers["anthropic"] is True
            assert provider.providers["bedrock"] is True
            
            # Check that circuit breakers were created
            assert "openai" in provider.circuit_breakers
            assert "anthropic" in provider.circuit_breakers
            assert "bedrock" in provider.circuit_breakers
            
            # Check token budget
            assert provider.token_budget.daily_budget == 1000000
    
    def test_generate_text_auto_provider(self, mock_services):
        """Test generating text with auto provider selection."""
        provider = AIProvider(
            openai_api_key="openai_key",
            anthropic_api_key="anthropic_key"
        )
        
        # Mock time.time for elapsed time calculation
        with patch('app.services._provider.time.time', side_effect=[100, 101]):
            result = provider.generate_text(
                prompt="Test prompt",
                provider="auto"
            )
            
            # Check that OpenAI was used (first in preference order)
            assert result["provider"] == "openai"
            assert result["text"] == "OpenAI response"
            assert mock_services["openai"].generate_text.called
            assert not mock_services["anthropic"].generate_text.called
            
            # Check response metadata
            assert result["elapsed_time"] == 1
            assert result["original_provider"] == "auto"
            assert result["within_budget"] is True
    
    def test_generate_text_specific_provider(self, mock_services):
        """Test generating text with a specific provider."""
        provider = AIProvider(
            openai_api_key="openai_key",
            anthropic_api_key="anthropic_key"
        )
        
        result = provider.generate_text(
            prompt="Test prompt",
            provider="anthropic",
            model="claude-3-opus-20240229"
        )
        
        # Check that Anthropic was used
        assert result["provider"] == "anthropic"
        assert result["text"] == "Anthropic response"
        assert mock_services["anthropic"].generate_text.called
        assert not mock_services["openai"].generate_text.called
        
        # Check that correct model was used
        call_args = mock_services["anthropic"].generate_text.call_args[1]
        assert call_args["model"] == "claude-3-opus-20240229"
    
    def test_generate_text_budget_exceeded(self, mock_services):
        """Test generating text when budget is exceeded."""
        provider = AIProvider(
            openai_api_key="openai_key",
            daily_token_budget=500
        )
        
        # Mock token budget to report exceeded
        with patch.object(provider.token_budget, 'is_within_budget', return_value=False):
            result = provider.generate_text(
                prompt="Test prompt",
                provider="openai"
            )
            
            # Check that error was returned and no service was called
            assert "error" in result
            assert result["error"] == "Daily token budget exceeded"
            assert result["within_budget"] is False
            assert not mock_services["openai"].generate_text.called
    
    def test_generate_text_unavailable_provider(self, mock_services):
        """Test generating text with an unavailable provider."""
        provider = AIProvider(
            openai_api_key="openai_key"
        )
        
        # Set anthropic as unavailable
        provider.providers["anthropic"] = False
        
        result = provider.generate_text(
            prompt="Test prompt",
            provider="anthropic",
            fallback=False  # No fallback
        )
        
        # Check that error was returned
        assert "error" in result
        assert "not available" in result["error"]
        assert not mock_services["anthropic"].generate_text.called
        assert not mock_services["openai"].generate_text.called
    
    def test_generate_text_with_fallback(self, mock_services):
        """Test generating text with fallback to another provider."""
        provider = AIProvider(
            openai_api_key="openai_key",
            anthropic_api_key="anthropic_key"
        )
        
        # Make OpenAI service fail
        mock_services["openai"].generate_text.return_value = {
            "error": "OpenAI error"
        }
        
        result = provider.generate_text(
            prompt="Test prompt",
            provider="openai",
            fallback=True
        )
        
        # Check that Anthropic was used as fallback
        assert result["provider"] == "anthropic"
        assert result["text"] == "Anthropic response"
        assert mock_services["openai"].generate_text.called
        assert mock_services["anthropic"].generate_text.called
    
    def test_generate_text_circuit_breaker_open(self, mock_services):
        """Test generating text when circuit breaker is open."""
        provider = AIProvider(
            openai_api_key="openai_key",
            anthropic_api_key="anthropic_key"
        )
        
        # Set OpenAI circuit breaker as open
        provider.circuit_breakers["openai"].state = "OPEN"
        provider.circuit_breakers["openai"].last_failure_time = datetime.now()
        
        result = provider.generate_text(
            prompt="Test prompt",
            provider="openai",
            fallback=True
        )
        
        # Check that Anthropic was used due to circuit breaker
        assert result["provider"] == "anthropic"
        assert result["text"] == "Anthropic response"
        assert not mock_services["openai"].generate_text.called
        assert mock_services["anthropic"].generate_text.called
    
    def test_structured_generate(self, mock_services):
        """Test generating structured output."""
        provider = AIProvider(
            openai_api_key="openai_key"
        )
        
        # Mock generate_text to return JSON
        with patch.object(provider, 'generate_text') as mock_generate_text:
            mock_generate_text.return_value = {
                "text": '{"key": "value"}',
                "provider": "openai",
                "model": "gpt-4"
            }
            
            schema = {"type": "object", "properties": {"key": {"type": "string"}}}
            result = provider.structured_generate(
                prompt="Test prompt",
                output_schema=schema,
                provider="openai"
            )
            
            # Check that JSON was parsed
            assert "parsed_result" in result
            assert result["parsed_result"] == {"key": "value"}
            
            # Check that output_schema was included in prompt
            call_args = mock_generate_text.call_args[1]
            assert "prompt" in call_args
            assert json.dumps(schema) in call_args["prompt"]
    
    def test_analyze_code(self, mock_services):
        """Test analyzing code."""
        provider = AIProvider(
            openai_api_key="openai_key"
        )
        
        # Mock structured_generate
        with patch.object(provider, 'structured_generate') as mock_structured_generate:
            mock_structured_generate.return_value = {
                "text": '{"summary": "Good code"}',
                "parsed_result": {"summary": "Good code"},
                "provider": "openai"
            }
            
            result = provider.analyze_code(
                code="def test(): pass",
                language="python",
                analysis_type="security"
            )
            
            # Check that structured_generate was called correctly
            assert mock_structured_generate.called
            call_args = mock_structured_generate.call_args[1]
            assert "security" in call_args["prompt"]
            assert "python" in call_args["prompt"]
            assert "def test(): pass" in call_args["prompt"]
            
            # Check result
            assert result["parsed_result"]["summary"] == "Good code"
    
    def test_get_provider_status(self, mock_services):
        """Test getting provider status."""
        provider = AIProvider(
            openai_api_key="openai_key",
            anthropic_api_key="anthropic_key"
        )
        
        # Add some performance metrics
        provider.performance_metrics["openai"] = [
            {"elapsed_time": 1.0, "success": True},
            {"elapsed_time": 2.0, "success": True}
        ]
        
        # Set a circuit breaker state
        provider.circuit_breakers["anthropic"].state = "OPEN"
        provider.circuit_breakers["anthropic"].failures = 5
        provider.circuit_breakers["anthropic"].last_failure_time = datetime.now()
        
        # Mock token budget
        with patch.object(provider.token_budget, 'get_total_usage_today', return_value=500), \
             patch.object(provider.token_budget, 'get_remaining_budget', return_value=500), \
             patch.object(provider.token_budget, 'get_usage_today', return_value={"openai": 500}):
            
            status = provider.get_provider_status()
            
            # Check status for providers
            assert status["openai"]["available"] is True
            assert status["openai"]["circuit_state"] == "CLOSED"
            assert status["openai"]["message"] == "Ready"
            assert status["openai"]["avg_response_time"] == 1.5
            assert status["openai"]["success_rate"] == 100.0
            
            assert status["anthropic"]["available"] is True
            assert status["anthropic"]["circuit_state"] == "OPEN"
            assert status["anthropic"]["message"] == "Unavailable due to failures"
            assert status["anthropic"]["failures"] == 5
            
            # Check token budget info
            assert status["token_budget"]["total_budget"] == 1000000
            assert status["token_budget"]["used_today"] == 500
            assert status["token_budget"]["remaining"] == 500
            assert status["token_budget"]["usage_by_provider"] == {"openai": 500}
    
    def test_reset_circuit_breaker(self, mock_services):
        """Test resetting a circuit breaker."""
        provider = AIProvider(
            openai_api_key="openai_key"
        )
        
        # Set up circuit breaker state
        provider.circuit_breakers["openai"].state = "OPEN"
        provider.circuit_breakers["openai"].failures = 3
        provider.circuit_breakers["openai"].last_failure_time = datetime.now()
        
        # Reset circuit breaker
        result = provider.reset_circuit_breaker("openai")
        
        # Check result
        assert result["success"] is True
        assert result["provider"] == "openai"
        assert result["new_state"] == "CLOSED"
        
        # Check that circuit breaker was reset
        assert provider.circuit_breakers["openai"].state == "CLOSED"
        assert provider.circuit_breakers["openai"].failures == 0
        assert provider.circuit_breakers["openai"].last_failure_time is None
    
    def test_reset_nonexistent_circuit_breaker(self, mock_services):
        """Test resetting a nonexistent circuit breaker."""
        provider = AIProvider(
            openai_api_key="openai_key"
        )
        
        result = provider.reset_circuit_breaker("nonexistent")
        
        assert result["success"] is False
        assert "not found" in result["message"]
    
    def test_sanitize_prompt(self, mock_services):
        """Test sanitizing prompts for different providers."""
        provider = AIProvider(
            openai_api_key="openai_key",
            anthropic_api_key="anthropic_key"
        )
        
        # Test Anthropic sanitization
        anthropic_prompt = "Use <tags> for emphasis"
        sanitized = provider.sanitize_prompt(anthropic_prompt, "anthropic")
        assert "&lt;tags&gt;" in sanitized
        
        # Test OpenAI sanitization
        openai_prompt = "Test prompt with \u200b zero-width space"
        sanitized = provider.sanitize_prompt(openai_prompt, "openai")
        assert "\u200b" not in sanitized
    
    def test_sanitize_response(self, mock_services):
        """Test sanitizing responses from different providers."""
        provider = AIProvider(
            openai_api_key="openai_key",
            anthropic_api_key="anthropic_key"
        )
        
        # Test Anthropic response sanitization
        anthropic_response = "Here's the JSON you requested: {\"key\": \"value\"}"
        sanitized = provider.sanitize_response(anthropic_response, "anthropic")
        assert sanitized == "{\"key\": \"value\"}"
        
        # Test OpenAI response sanitization
        openai_response = "```json\n{\"key\": \"value\"}\n```"
        sanitized = provider.sanitize_response(openai_response, "openai")
        assert sanitized == "{\"key\": \"value\"}"
    
    def test_extract_json(self, mock_services):
        """Test extracting JSON from text."""
        provider = AIProvider(
            openai_api_key="openai_key"
        )
        
        # Test extracting from code block
        text = "Here is the output:\n```json\n{\"key\": \"value\"}\n```\nMore text"
        json_str = provider._extract_json(text)
        assert json_str == "{\"key\": \"value\"}"
        
        # Test extracting from braces
        text = "Here is the output: {\"key\": \"value\"} More text"
        json_str = provider._extract_json(text)
        assert json_str == "{\"key\": \"value\"}"
        
        # Test with no JSON
        text = "Here is some text with no JSON"
        json_str = provider._extract_json(text)
        assert json_str == "Here is some text with no JSON"
    
    def test_estimate_prompt_complexity(self, mock_services):
        """Test estimating prompt complexity."""
        provider = AIProvider(
            openai_api_key="openai_key"
        )
        
        # Test simple prompt
        simple_prompt = "What is 2+2?"
        simple_complexity = provider._estimate_prompt_complexity(simple_prompt)
        assert 1 <= simple_complexity <= 3  # Should be low complexity
        
        # Test complex prompt with code, JSON, and structure
        complex_prompt = """
        # Analysis Request
        
        Please analyze the following code:
        
        ```python
        def complex_function(a, b):
            return a + b
        ```
        
        And respond with a JSON structure like:
        
        {
            "analysis": "detailed analysis",
            "issues": ["issue1", "issue2"]
        }
        
        * Include performance considerations
        * Include security considerations
        * Include readability considerations
        """
        complex_complexity = provider._estimate_prompt_complexity(complex_prompt)
        assert 5 <= complex_complexity <= 10  # Should be high complexity
        
        # The complex prompt should be rated as more complex
        assert complex_complexity > simple_complexity

if __name__ == "__main__":
    pytest.main(["-v", __file__])