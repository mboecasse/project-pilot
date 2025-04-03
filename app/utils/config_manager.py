"""
ProjectPilot - AI-powered project management system
Configuration management system for settings and preferences.
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, Optional, Union, List, Set
import threading
from datetime import datetime
from pathlib import Path
from flask import current_app

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages application configuration settings and user preferences
    with support for hierarchical configs, defaults, and validation.
    """
    
    def __init__(self, 
                config_dir: Optional[str] = None,
                default_config_path: Optional[str] = None,
                cache_ttl: int = 300):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory for configuration files
            default_config_path: Path to default configuration file
            cache_ttl: Cache time-to-live in seconds
        """
        self.config_dir = config_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
        self.default_config_path = default_config_path or os.path.join(self.config_dir, 'defaults.yaml')
        self.cache_ttl = cache_ttl
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Configuration storage
        self.system_config = {}
        self.project_configs = {}
        self.user_configs = {}
        self.bot_configs = {}
        self.default_config = {}
        
        # Cache management
        self.cache_timestamps = {}
        self.config_locks = {
            'system': threading.RLock(),
            'project': threading.RLock(),
            'user': threading.RLock(),
            'bot': threading.RLock()
        }
        
        # Configuration schemas for validation
        self.schemas = {}
        
        # Initialize configurations
        self._load_default_config()
        self._load_system_config()
        
        logger.info("Configuration manager initialized")
    
    def get_system_config(self, key: Optional[str] = None) -> Any:
        """
        Get system configuration.
        
        Args:
            key: Optional config key to retrieve a specific setting
            
        Returns:
            Configuration value or entire config if key is None
        """
        # Reload if cache is stale
        self._check_reload_system_config()
        
        with self.config_locks['system']:
            if key is None:
                return self.system_config.copy()
            
            # Support nested keys with dot notation
            if '.' in key:
                parts = key.split('.')
                value = self.system_config
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        # Key not found, fallback to default
                        return self._get_default_value(key)
                return value
            
            return self.system_config.get(key, self._get_default_value(key))
    
    def get_project_config(self, project_id: int, key: Optional[str] = None) -> Any:
        """
        Get project configuration.
        
        Args:
            project_id: Project ID
            key: Optional config key to retrieve a specific setting
            
        Returns:
            Configuration value or entire config if key is None
        """
        # Reload if cache is stale
        self._check_reload_project_config(project_id)
        
        with self.config_locks['project']:
            if project_id not in self.project_configs:
                # Load project config if not cached
                self._load_project_config(project_id)
            
            if key is None:
                return self.project_configs.get(project_id, {}).copy()
            
            # Support nested keys with dot notation
            if '.' in key:
                parts = key.split('.')
                value = self.project_configs.get(project_id, {})
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        # Key not found, fallback to system then default
                        return self.get_system_config(key)
                return value
            
            project_config = self.project_configs.get(project_id, {})
            return project_config.get(key, self.get_system_config(key))
    
    def get_user_config(self, user_id: int, key: Optional[str] = None) -> Any:
        """
        Get user configuration.
        
        Args:
            user_id: User ID
            key: Optional config key to retrieve a specific setting
            
        Returns:
            Configuration value or entire config if key is None
        """
        # Reload if cache is stale
        self._check_reload_user_config(user_id)
        
        with self.config_locks['user']:
            if user_id not in self.user_configs:
                # Load user config if not cached
                self._load_user_config(user_id)
            
            if key is None:
                return self.user_configs.get(user_id, {}).copy()
            
            # Support nested keys with dot notation
            if '.' in key:
                parts = key.split('.')
                value = self.user_configs.get(user_id, {})
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        # Key not found, fallback to system then default
                        return self.get_system_config(key)
                return value
            
            user_config = self.user_configs.get(user_id, {})
            return user_config.get(key, self.get_system_config(key))
    
    def get_bot_config(self, bot_id: int, key: Optional[str] = None) -> Any:
        """
        Get worker bot configuration.
        
        Args:
            bot_id: Bot ID
            key: Optional config key to retrieve a specific setting
            
        Returns:
            Configuration value or entire config if key is None
        """
        # Reload if cache is stale
        self._check_reload_bot_config(bot_id)
        
        with self.config_locks['bot']:
            if bot_id not in self.bot_configs:
                # Load bot config if not cached
                self._load_bot_config(bot_id)
            
            if key is None:
                return self.bot_configs.get(bot_id, {}).copy()
            
            # Support nested keys with dot notation
            if '.' in key:
                parts = key.split('.')
                value = self.bot_configs.get(bot_id, {})
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        # Key not found, fallback to system then default
                        return self.get_system_config(key)
                return value
            
            bot_config = self.bot_configs.get(bot_id, {})
            return bot_config.get(key, self.get_system_config(key))
    
    def update_system_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update system configuration.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if update was successful
        """
        with self.config_locks['system']:
            # Validate updates
            if not self._validate_config(updates, 'system'):
                return False
            
            # Update configuration
            self._deep_update(self.system_config, updates)
            
            # Save to file
            result = self._save_system_config()
            
            return result
    
    def update_project_config(self, project_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update project configuration.
        
        Args:
            project_id: Project ID
            updates: Dictionary of configuration updates
            
        Returns:
            True if update was successful
        """
        with self.config_locks['project']:
            # Load project config if not cached
            if project_id not in self.project_configs:
                self._load_project_config(project_id)
            
            # Initialize if not exists
            if project_id not in self.project_configs:
                self.project_configs[project_id] = {}
            
            # Validate updates
            if not self._validate_config(updates, 'project'):
                return False
            
            # Update configuration
            self._deep_update(self.project_configs[project_id], updates)
            
            # Save to file
            result = self._save_project_config(project_id)
            
            return result
    
    def update_user_config(self, user_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update user configuration.
        
        Args:
            user_id: User ID
            updates: Dictionary of configuration updates
            
        Returns:
            True if update was successful
        """
        with self.config_locks['user']:
            # Load user config if not cached
            if user_id not in self.user_configs:
                self._load_user_config(user_id)
            
            # Initialize if not exists
            if user_id not in self.user_configs:
                self.user_configs[user_id] = {}
            
            # Validate updates
            if not self._validate_config(updates, 'user'):
                return False
            
            # Update configuration
            self._deep_update(self.user_configs[user_id], updates)
            
            # Save to file
            result = self._save_user_config(user_id)
            
            return result
    
    def update_bot_config(self, bot_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update worker bot configuration.
        
        Args:
            bot_id: Bot ID
            updates: Dictionary of configuration updates
            
        Returns:
            True if update was successful
        """
        with self.config_locks['bot']:
            # Load bot config if not cached
            if bot_id not in self.bot_configs:
                self._load_bot_config(bot_id)
            
            # Initialize if not exists
            if bot_id not in self.bot_configs:
                self.bot_configs[bot_id] = {}
            
            # Validate updates
            if not self._validate_config(updates, 'bot'):
                return False
            
            # Update configuration
            self._deep_update(self.bot_configs[bot_id], updates)
            
            # Save to file
            result = self._save_bot_config(bot_id)
            
            return result
    
    def reset_to_defaults(self, config_type: str, id: Optional[int] = None) -> bool:
        """
        Reset configuration to defaults.
        
        Args:
            config_type: Type of configuration ('system', 'project', 'user', 'bot')
            id: ID for project, user, or bot (not needed for system)
            
        Returns:
            True if reset was successful
        """
        if config_type == 'system':
            with self.config_locks['system']:
                # Reset to defaults
                self.system_config = self._get_default_system_config()
                
                # Save to file
                return self._save_system_config()
                
        elif config_type == 'project' and id is not None:
            with self.config_locks['project']:
                # Reset to defaults
                self.project_configs[id] = self._get_default_project_config()
                
                # Save to file
                return self._save_project_config(id)
                
        elif config_type == 'user' and id is not None:
            with self.config_locks['user']:
                # Reset to defaults
                self.user_configs[id] = self._get_default_user_config()
                
                # Save to file
                return self._save_user_config(id)
                
        elif config_type == 'bot' and id is not None:
            with self.config_locks['bot']:
                # Reset to defaults
                self.bot_configs[id] = self._get_default_bot_config()
                
                # Save to file
                return self._save_bot_config(id)
        
        return False
    
    def register_schema(self, config_type: str, schema: Dict[str, Any]) -> None:
        """
        Register a JSON schema for configuration validation.
        
        Args:
            config_type: Type of configuration ('system', 'project', 'user', 'bot')
            schema: JSON schema for validation
        """
        self.schemas[config_type] = schema
    
    def get_available_settings(self, config_type: str) -> Dict[str, Any]:
        """
        Get available settings and their descriptions.
        
        Args:
            config_type: Type of configuration ('system', 'project', 'user', 'bot')
            
        Returns:
            Dictionary of settings and descriptions
        """
        if config_type not in self.schemas:
            return {}
        
        schema = self.schemas[config_type]
        
        # Extract properties and descriptions from schema
        settings = {}
        
        if 'properties' in schema:
            for key, value in schema['properties'].items():
                settings[key] = {
                    'description': value.get('description', ''),
                    'type': value.get('type', 'string'),
                    'default': self._get_default_value(key),
                    'enum': value.get('enum', None)
                }
        
        return settings
    
    def _load_default_config(self) -> None:
        """Load default configuration."""
        try:
            if os.path.exists(self.default_config_path):
                with open(self.default_config_path, 'r') as f:
                    self.default_config = yaml.safe_load(f) or {}
            else:
                # Create default config if it doesn't exist
                self.default_config = self._create_default_config()
                
                # Save to file
                with open(self.default_config_path, 'w') as f:
                    yaml.dump(self.default_config, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error loading default config: {str(e)}")
            self.default_config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'system': {
                'max_concurrent_tasks': 10,
                'polling_interval': 60,
                'logging': {
                    'level': 'INFO',
                    'file_rotation': '1 day',
                    'max_files': 30
                },
                'ai': {
                    'providers': {
                        'openai': {
                            'enabled': True,
                            'api_key_env': 'OPENAI_API_KEY',
                            'default_model': 'gpt-4',
                            'max_tokens': 2000,
                            'temperature': 0.7
                        },
                        'anthropic': {
                            'enabled': True,
                            'api_key_env': 'ANTHROPIC_API_KEY',
                            'default_model': 'claude-3-opus-20240229',
                            'max_tokens': 2000,
                            'temperature': 0.7
                        },
                        'bedrock': {
                            'enabled': False,
                            'region': 'us-west-2',
                            'default_model': 'anthropic.claude-3-opus-20240229-v1:0',
                            'max_tokens': 2000,
                            'temperature': 0.7
                        }
                    },
                    'token_budget': {
                        'daily_limit': 1000000,
                        'warning_threshold': 0.8
                    }
                },
                'performance': {
                    'cache_ttl': 300,
                    'max_cache_size': 1000,
                    'enable_metrics': True
                },
                'security': {
                    'enable_rate_limiting': True,
                    'max_requests_per_minute': 100,
                    'require_authentication': True
                }
            },
            'project': {
                'default_task_priority': 2,
                'default_task_weight': 1,
                'auto_assign_tasks': True,
                'auto_spawn_bots': True,
                'required_bots': {
                    'architect': 1,
                    'developer': 2,
                    'tester': 1,
                    'devops': 1
                },
                'task_completion_criteria': {
                    'require_review': True,
                    'require_tests': True
                }
            },
            'user': {
                'theme': 'light',
                'notifications': {
                    'email': True,
                    'webapp': True,
                    'frequency': 'immediate'
                },
                'dashboard': {
                    'default_view': 'projects',
                    'show_welcome': True
                }
            },
            'bot': {
                'max_concurrent_tasks': 3,
                'idle_timeout': 3600,
                'error_retry_count': 3,
                'error_retry_delay': 300,
                'health_check_interval': 300,
                'auto_improve': True
            }
        }
    
    def _load_system_config(self) -> None:
        """Load system configuration."""
        try:
            system_config_path = os.path.join(self.config_dir, 'system.yaml')
            
            if os.path.exists(system_config_path):
                with open(system_config_path, 'r') as f:
                    self.system_config = yaml.safe_load(f) or {}
            else:
                # Create system config if it doesn't exist
                self.system_config = self._get_default_system_config()
                
                # Save to file
                with open(system_config_path, 'w') as f:
                    yaml.dump(self.system_config, f, default_flow_style=False)
            
            # Update cache timestamp
            self.cache_timestamps['system'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error loading system config: {str(e)}")
            self.system_config = self._get_default_system_config()
    
    def _load_project_config(self, project_id: int) -> None:
        """
        Load project configuration.
        
        Args:
            project_id: Project ID
        """
        try:
            project_config_path = os.path.join(self.config_dir, f'project_{project_id}.yaml')
            
            if os.path.exists(project_config_path):
                with open(project_config_path, 'r') as f:
                    self.project_configs[project_id] = yaml.safe_load(f) or {}
            else:
                # Create project config if it doesn't exist
                self.project_configs[project_id] = self._get_default_project_config()
                
                # Save to file
                with open(project_config_path, 'w') as f:
                    yaml.dump(self.project_configs[project_id], f, default_flow_style=False)
            
            # Update cache timestamp
            self.cache_timestamps[f'project_{project_id}'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error loading project config for project {project_id}: {str(e)}")
            self.project_configs[project_id] = self._get_default_project_config()
    
    def _load_user_config(self, user_id: int) -> None:
        """
        Load user configuration.
        
        Args:
            user_id: User ID
        """
        try:
            user_config_path = os.path.join(self.config_dir, f'user_{user_id}.yaml')
            
            if os.path.exists(user_config_path):
                with open(user_config_path, 'r') as f:
                    self.user_configs[user_id] = yaml.safe_load(f) or {}
            else:
                # Create user config if it doesn't exist
                self.user_configs[user_id] = self._get_default_user_config()
                
                # Save to file
                with open(user_config_path, 'w') as f:
                    yaml.dump(self.user_configs[user_id], f, default_flow_style=False)
            
            # Update cache timestamp
            self.cache_timestamps[f'user_{user_id}'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error loading user config for user {user_id}: {str(e)}")
            self.user_configs[user_id] = self._get_default_user_config()
    
    def _load_bot_config(self, bot_id: int) -> None:
        """
        Load worker bot configuration.
        
        Args:
            bot_id: Bot ID
        """
        try:
            bot_config_path = os.path.join(self.config_dir, f'bot_{bot_id}.yaml')
            
            if os.path.exists(bot_config_path):
                with open(bot_config_path, 'r') as f:
                    self.bot_configs[bot_id] = yaml.safe_load(f) or {}
            else:
                # Create bot config if it doesn't exist
                self.bot_configs[bot_id] = self._get_default_bot_config()
                
                # Save to file
                with open(bot_config_path, 'w') as f:
                    yaml.dump(self.bot_configs[bot_id], f, default_flow_style=False)
            
            # Update cache timestamp
            self.cache_timestamps[f'bot_{bot_id}'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error loading bot config for bot {bot_id}: {str(e)}")
            self.bot_configs[bot_id] = self._get_default_bot_config()
    
    def _save_system_config(self) -> bool:
        """
        Save system configuration to file.
        
        Returns:
            True if save was successful
        """
        try:
            system_config_path = os.path.join(self.config_dir, 'system.yaml')
            
            with open(system_config_path, 'w') as f:
                yaml.dump(self.system_config, f, default_flow_style=False)
            
            # Update cache timestamp
            self.cache_timestamps['system'] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving system config: {str(e)}")
            return False
    
    def _save_project_config(self, project_id: int) -> bool:
        """
        Save project configuration to file.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if save was successful
        """
        try:
            project_config_path = os.path.join(self.config_dir, f'project_{project_id}.yaml')
            
            with open(project_config_path, 'w') as f:
                yaml.dump(self.project_configs[project_id], f, default_flow_style=False)
            
            # Update cache timestamp
            self.cache_timestamps[f'project_{project_id}'] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving project config for project {project_id}: {str(e)}")
            return False
    
    def _save_user_config(self, user_id: int) -> bool:
        """
        Save user configuration to file.
        
        Args:
            user_id: User ID
            
        Returns:
            True if save was successful
        """
        try:
            user_config_path = os.path.join(self.config_dir, f'user_{user_id}.yaml')
            
            with open(user_config_path, 'w') as f:
                yaml.dump(self.user_configs[user_id], f, default_flow_style=False)
            
            # Update cache timestamp
            self.cache_timestamps[f'user_{user_id}'] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving user config for user {user_id}: {str(e)}")
            return False
    
    def _save_bot_config(self, bot_id: int) -> bool:
        """
        Save worker bot configuration to file.
        
        Args:
            bot_id: Bot ID
            
        Returns:
            True if save was successful
        """
        try:
            bot_config_path = os.path.join(self.config_dir, f'bot_{bot_id}.yaml')
            
            with open(bot_config_path, 'w') as f:
                yaml.dump(self.bot_configs[bot_id], f, default_flow_style=False)
            
            # Update cache timestamp
            self.cache_timestamps[f'bot_{bot_id}'] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving bot config for bot {bot_id}: {str(e)}")
            return False
    
    def _check_reload_system_config(self) -> None:
        """Check if system config should be reloaded."""
        if 'system' not in self.cache_timestamps:
            self._load_system_config()
            return
        
        time_since_load = (datetime.now() - self.cache_timestamps['system']).total_seconds()
        
        if time_since_load > self.cache_ttl:
            self._load_system_config()
    
    def _check_reload_project_config(self, project_id: int) -> None:
        """
        Check if project config should be reloaded.
        
        Args:
            project_id: Project ID
        """
        cache_key = f'project_{project_id}'
        
        if cache_key not in self.cache_timestamps:
            self._load_project_config(project_id)
            return
        
        time_since_load = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        
        if time_since_load > self.cache_ttl:
            self._load_project_config(project_id)
    
    def _check_reload_user_config(self, user_id: int) -> None:
        """
        Check if user config should be reloaded.
        
        Args:
            user_id: User ID
        """
        cache_key = f'user_{user_id}'
        
        if cache_key not in self.cache_timestamps:
            self._load_user_config(user_id)
            return
        
        time_since_load = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        
        if time_since_load > self.cache_ttl:
            self._load_user_config(user_id)
    
    def _check_reload_bot_config(self, bot_id: int) -> None:
        """
        Check if bot config should be reloaded.
        
        Args:
            bot_id: Bot ID
        """
        cache_key = f'bot_{bot_id}'
        
        if cache_key not in self.cache_timestamps:
            self._load_bot_config(bot_id)
            return
        
        time_since_load = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        
        if time_since_load > self.cache_ttl:
            self._load_bot_config(bot_id)
    
    def _get_default_system_config(self) -> Dict[str, Any]:
        """
        Get default system configuration.
        
        Returns:
            Default system configuration
        """
        return self.default_config.get('system', {}).copy()
    
    def _get_default_project_config(self) -> Dict[str, Any]:
        """
        Get default project configuration.
        
        Returns:
            Default project configuration
        """
        return self.default_config.get('project', {}).copy()
    
    def _get_default_user_config(self) -> Dict[str, Any]:
        """
        Get default user configuration.
        
        Returns:
            Default user configuration
        """
        return self.default_config.get('user', {}).copy()
    
    def _get_default_bot_config(self) -> Dict[str, Any]:
        """
        Get default worker bot configuration.
        
        Returns:
            Default bot configuration
        """
        return self.default_config.get('bot', {}).copy()
    
    def _get_default_value(self, key: str) -> Any:
        """
        Get default value for a configuration key.
        
        Args:
            key: Configuration key
            
        Returns:
            Default value or None if not found
        """
        # Support nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            config_type = parts[0]
            
            if config_type in self.default_config:
                value = self.default_config[config_type]
                
                for part in parts[1:]:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return None
                
                return value
            
            return None
        
        # Check if the key corresponds to a top-level config type
        if key in self.default_config:
            return self.default_config[key].copy()
        
        # Otherwise, look for the key in each config type
        for config_type, config in self.default_config.items():
            if isinstance(config, dict) and key in config:
                return config[key]
        
        return None
    
    def _validate_config(self, config: Dict[str, Any], config_type: str) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            config_type: Type of configuration
            
        Returns:
            True if configuration is valid
        """
        if config_type not in self.schemas:
            # No schema registered, assume valid
            return True
        
        try:
            import jsonschema
            
            # Validate against schema
            jsonschema.validate(instance=config, schema=self.schemas[config_type])
            return True
            
        except ImportError:
            # jsonschema not available, skip validation
            logger.warning("jsonschema not available, skipping validation")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep update a dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._deep_update(target[key], value)
            else:
                # Update or add key-value pair
                target[key] = value

# Initialize the global configuration manager
config_manager = ConfigManager()