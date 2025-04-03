"""
ProjectPilot - AI-powered project management system
Automated test case generator for Python modules.
"""

import os
import sys
import inspect
import ast
import re
import importlib
import importlib.util
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from datetime import datetime
import pytest

# Import AI services for test generation
from app.services._provider import AIProvider

logger = logging.getLogger(__name__)

class TestGenerator:
    """
    Automated test case generator that uses AI to create unit tests
    for Python modules, classes, and functions.
    """
    
    def __init__(self, 
                 project_root: str,
                 test_output_dir: Optional[str] = None,
                 ai_provider: Optional[AIProvider] = None,
                 max_retries: int = 2):
        """
        Initialize the test generator.
        
        Args:
            project_root: Project root directory
            test_output_dir: Directory to write test files (defaults to project_root/tests)
            ai_provider: AI provider for generating tests
            max_retries: Maximum number of retries for test generation
        """
        self.project_root = project_root
        self.test_output_dir = test_output_dir or os.path.join(project_root, 'tests')
        self.ai_provider = ai_provider or AIProvider()
        self.max_retries = max_retries
        
        # Statistics
        self.stats = {
            'modules_processed': 0,
            'modules_skipped': 0,
            'tests_generated': 0,
            'failures': 0,
            'total_functions_tested': 0,
            'total_classes_tested': 0
        }
        
        # Create test output directory if it doesn't exist
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Add project root to sys.path to enable imports
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        logger.info(f"Test generator initialized with root: {project_root}")
    
    def generate_tests(self,
                      target_modules: Optional[List[str]] = None,
                      exclude_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate test cases for all modules or specified modules.
        
        Args:
            target_modules: List of module paths to generate tests for (relative to project root)
                           If None, all modules will be processed
            exclude_patterns: List of patterns to exclude modules (e.g., ['test_', '_test'])
            
        Returns:
            Dictionary with generation statistics
        """
        start_time = datetime.now()
        
        # Set default exclude patterns if not provided
        if exclude_patterns is None:
            exclude_patterns = ['test_', '_test', '__pycache__', '.git', 'venv', 'env']
        
        # Collect all Python modules if no specific targets
        if target_modules is None:
            target_modules = self._collect_python_modules(self.project_root, exclude_patterns)
        
        # Generate tests for each module
        for module_path in target_modules:
            absolute_path = os.path.join(self.project_root, module_path)
            
            if not os.path.exists(absolute_path):
                logger.warning(f"Module not found: {module_path}")
                self.stats['modules_skipped'] += 1
                continue
            
            try:
                self._generate_tests_for_module(module_path)
                self.stats['modules_processed'] += 1
            except Exception as e:
                logger.error(f"Error generating tests for module {module_path}: {str(e)}")
                self.stats['failures'] += 1
                self.stats['modules_skipped'] += 1
        
        # Add timing information
        duration = (datetime.now() - start_time).total_seconds()
        self.stats['duration_seconds'] = duration
        self.stats['start_time'] = start_time.isoformat()
        self.stats['end_time'] = datetime.now().isoformat()
        
        # Generate summary report
        self._generate_summary_report()
        
        return self.stats
    
    def _collect_python_modules(self, 
                              directory: str, 
                              exclude_patterns: List[str]) -> List[str]:
        """
        Collect all Python modules in a directory.
        
        Args:
            directory: Directory to search
            exclude_patterns: Patterns to exclude
            
        Returns:
            List of module paths relative to project root
        """
        modules = []
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    # Skip excluded files
                    if any(pattern in file for pattern in exclude_patterns):
                        continue
                    
                    # Get path relative to project root
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, self.project_root)
                    
                    modules.append(rel_path)
        
        return modules
    
    def _generate_tests_for_module(self, module_path: str) -> None:
        """
        Generate tests for a single module.
        
        Args:
            module_path: Path to module (relative to project root)
        """
        # Extract module info
        module_info = self._extract_module_info(module_path)
        
        if not module_info:
            logger.warning(f"Failed to extract module info: {module_path}")
            self.stats['modules_skipped'] += 1
            return
        
        # Determine test file path
        test_file_path = self._get_test_file_path(module_path)
        
        # Check if test file already exists
        if os.path.exists(test_file_path):
            # Check if we should overwrite based on file contents
            with open(test_file_path, 'r') as f:
                content = f.read()
                if 'AUTO-GENERATED TEST FILE' not in content:
                    logger.info(f"Test file exists and was manually created, skipping: {test_file_path}")
                    self.stats['modules_skipped'] += 1
                    return
        
        # Generate test content
        test_content = self._generate_test_content(module_info)
        
        if not test_content:
            logger.warning(f"Failed to generate test content for: {module_path}")
            self.stats['modules_skipped'] += 1
            return
        
        # Write test file
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        logger.info(f"Generated test file: {test_file_path}")
        self.stats['tests_generated'] += 1
    
    def _extract_module_info(self, module_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract information about a module.
        
        Args:
            module_path: Path to module (relative to project root)
            
        Returns:
            Dictionary with module information or None if extraction fails
        """
        try:
            abs_path = os.path.join(self.project_root, module_path)
            
            # Parse module AST
            with open(abs_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # Extract module information
            module_info = {
                'name': os.path.basename(module_path).replace('.py', ''),
                'path': module_path,
                'abs_path': abs_path,
                'source': source,
                'docstring': ast.get_docstring(tree) or '',
                'classes': [],
                'functions': []
            }
            
            # Extract classes and functions
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_info(node, source)
                    module_info['classes'].append(class_info)
                    self.stats['total_classes_tested'] += 1
                    
                elif isinstance(node, ast.FunctionDef):
                    func_info = self._extract_function_info(node, source)
                    module_info['functions'].append(func_info)
                    self.stats['total_functions_tested'] += 1
            
            # Try to dynamically import the module to get additional information
            try:
                # Import module
                spec = importlib.util.spec_from_file_location(module_info['name'], abs_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Store the module object
                    module_info['module'] = module
                    
                    # Update information for classes and functions with real objects
                    for class_info in module_info['classes']:
                        if hasattr(module, class_info['name']):
                            class_info['obj'] = getattr(module, class_info['name'])
                    
                    for func_info in module_info['functions']:
                        if hasattr(module, func_info['name']):
                            func_info['obj'] = getattr(module, func_info['name'])
            except Exception as e:
                logger.warning(f"Could not import module {module_path}: {str(e)}")
                # Continue without dynamic information
            
            return module_info
            
        except Exception as e:
            logger.error(f"Error extracting module info for {module_path}: {str(e)}")
            return None
    
    def _extract_class_info(self, node: ast.ClassDef, source: str) -> Dict[str, Any]:
        """
        Extract information about a class.
        
        Args:
            node: AST node for the class
            source: Source code
            
        Returns:
            Dictionary with class information
        """
        class_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node) or '',
            'methods': [],
            'bases': [self._get_name_from_expr(base) for base in node.bases],
            'source': self._get_source_segment(source, node)
        }
        
        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item, source, is_method=True)
                class_info['methods'].append(method_info)
                self.stats['total_functions_tested'] += 1
        
        return class_info
    
    def _extract_function_info(self, 
                             node: ast.FunctionDef, 
                             source: str,
                             is_method: bool = False) -> Dict[str, Any]:
        """
        Extract information about a function or method.
        
        Args:
            node: AST node for the function
            source: Source code
            is_method: Whether this is a method or a standalone function
            
        Returns:
            Dictionary with function information
        """
        func_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node) or '',
            'is_method': is_method,
            'args': self._extract_function_args(node),
            'source': self._get_source_segment(source, node),
            'is_private': node.name.startswith('_') and not node.name.startswith('__') and not node.name.endswith('__')
        }
        
        # Extract return annotation if any
        if node.returns:
            func_info['returns'] = self._get_name_from_expr(node.returns)
        else:
            func_info['returns'] = None
        
        return func_info
    
    def _extract_function_args(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """
        Extract function arguments.
        
        Args:
            node: AST node for the function
            
        Returns:
            List of argument information dictionaries
        """
        args = []
        
        # Process positional arguments
        for i, arg in enumerate(node.args.args):
            # Skip 'self' for methods
            if i == 0 and arg.arg == 'self':
                continue
                
            arg_info = {
                'name': arg.arg,
                'annotation': self._get_name_from_expr(arg.annotation) if arg.annotation else None,
                'has_default': i >= len(node.args.args) - len(node.args.defaults)
            }
            
            # Add default value if available
            if arg_info['has_default']:
                default_index = i - (len(node.args.args) - len(node.args.defaults))
                default_node = node.args.defaults[default_index]
                arg_info['default'] = self._get_value_from_expr(default_node)
                
            args.append(arg_info)
        
        # Process *args
        if node.args.vararg:
            args.append({
                'name': f"*{node.args.vararg.arg}",
                'annotation': self._get_name_from_expr(node.args.vararg.annotation) if node.args.vararg.annotation else None,
                'has_default': False
            })
        
        # Process keyword-only arguments
        for i, arg in enumerate(node.args.kwonlyargs):
            arg_info = {
                'name': arg.arg,
                'annotation': self._get_name_from_expr(arg.annotation) if arg.annotation else None,
                'has_default': i < len(node.args.kw_defaults) and node.args.kw_defaults[i] is not None
            }
            
            # Add default value if available
            if arg_info['has_default']:
                default_node = node.args.kw_defaults[i]
                arg_info['default'] = self._get_value_from_expr(default_node)
                
            args.append(arg_info)
        
        # Process **kwargs
        if node.args.kwarg:
            args.append({
                'name': f"**{node.args.kwarg.arg}",
                'annotation': self._get_name_from_expr(node.args.kwarg.annotation) if node.args.kwarg.annotation else None,
                'has_default': False
            })
        
        return args
    
    def _get_name_from_expr(self, node: Optional[ast.AST]) -> Optional[str]:
        """
        Get a string representation of an AST expression.
        
        Args:
            node: AST node
            
        Returns:
            String representation or None
        """
        if node is None:
            return None
            
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_name_from_expr(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        elif isinstance(node, ast.Subscript):
            value = self._get_name_from_expr(node.value)
            if hasattr(node, 'slice') and isinstance(node.slice, ast.Index):
                # Python 3.8 compatibility
                slice_value = self._get_name_from_expr(node.slice.value)
                return f"{value}[{slice_value}]"
            elif hasattr(node, 'slice'):
                # Python 3.9+
                if isinstance(node.slice, ast.Name):
                    return f"{value}[{node.slice.id}]"
                elif isinstance(node.slice, ast.Subscript):
                    slice_value = self._get_name_from_expr(node.slice)
                    return f"{value}[{slice_value}]"
                elif isinstance(node.slice, ast.Tuple):
                    elements = []
                    for elt in node.slice.elts:
                        elements.append(self._get_name_from_expr(elt))
                    return f"{value}[{', '.join(elements)}]"
                else:
                    return f"{value}[...]"
            return f"{value}[...]"
        elif isinstance(node, ast.Call):
            func_name = self._get_name_from_expr(node.func)
            return f"{func_name}(...)"
        else:
            return "..."
    
    def _get_value_from_expr(self, node: Optional[ast.AST]) -> Any:
        """
        Try to get the Python value from an AST expression.
        
        Args:
            node: AST node
            
        Returns:
            Python value or string representation
        """
        if node is None:
            return None
            
        # Handle constants
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            # Python 3.7 compatibility
            return node.s
        elif isinstance(node, ast.Num):
            # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.NameConstant):
            # Python 3.7 compatibility
            return node.value
        elif isinstance(node, ast.Name):
            # Handle common built-ins
            if node.id == 'None':
                return None
            elif node.id == 'True':
                return True
            elif node.id == 'False':
                return False
            else:
                return node.id  # Return name as string
        elif isinstance(node, ast.List):
            return [self._get_value_from_expr(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(self._get_value_from_expr(elt) for elt in node.elts)
        elif isinstance(node, ast.Dict):
            keys = [self._get_value_from_expr(k) for k in node.keys]
            values = [self._get_value_from_expr(v) for v in node.values]
            return dict(zip(keys, values))
        else:
            # Return string representation for complex expressions
            return self._get_name_from_expr(node)
    
    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """
        Get source code segment for an AST node.
        
        Args:
            source: Complete source code
            node: AST node
            
        Returns:
            Source code segment
        """
        # Get line numbers (1-indexed in AST)
        start_line = node.lineno - 1  # Convert to 0-indexed
        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
        
        # Split source into lines and extract segment
        lines = source.splitlines()
        return '\n'.join(lines[start_line:end_line+1])
    
    def _get_test_file_path(self, module_path: str) -> str:
        """
        Get the path for the test file.
        
        Args:
            module_path: Module path (relative to project root)
            
        Returns:
            Path for the test file
        """
        # Extract filename
        module_name = os.path.basename(module_path).replace('.py', '')
        
        # Extract directory path (if any)
        dir_path = os.path.dirname(module_path)
        
        # Create test directory structure
        if dir_path:
            test_dir = os.path.join(self.test_output_dir, dir_path)
            os.makedirs(test_dir, exist_ok=True)
            return os.path.join(test_dir, f"test_{module_name}.py")
        else:
            return os.path.join(self.test_output_dir, f"test_{module_name}.py")
    
    def _generate_test_content(self, module_info: Dict[str, Any]) -> Optional[str]:
        """
        Generate test content for a module.
        
        Args:
            module_info: Module information
            
        Returns:
            Test content or None if generation fails
        """
        # Use AI to generate tests if available
        if self.ai_provider:
            for attempt in range(self.max_retries + 1):
                try:
                    # Generate with AI
                    content = self._generate_test_content_with_ai(module_info)
                    if content:
                        return content
                except Exception as e:
                    logger.warning(f"AI test generation failed (attempt {attempt+1}): {str(e)}")
            
        # Fall back to template-based generation
        return self._generate_test_content_from_template(module_info)
    
    def _generate_test_content_with_ai(self, module_info: Dict[str, Any]) -> Optional[str]:
        """
        Generate test content using AI.
        
        Args:
            module_info: Module information
            
        Returns:
            Test content or None if generation fails
        """
        # Prepare context for the AI
        module_source = module_info['source']
        
        # Skip if no source
        if not module_source:
            return None
        
        # Extract specific info for functions and classes for better test generation
        test_targets = []
        
        # Add functions to test
        for func in module_info['functions']:
            # Skip private functions unless they look like they need testing
            if func['is_private'] and not self._should_test_private_function(func):
                continue
                
            test_targets.append({
                'type': 'function',
                'name': func['name'],
                'docstring': func['docstring'],
                'args': func['args'],
                'returns': func['returns']
            })
        
        # Add classes to test
        for cls in module_info['classes']:
            class_target = {
                'type': 'class',
                'name': cls['name'],
                'docstring': cls['docstring'],
                'bases': cls['bases'],
                'methods': []
            }
            
            # Add methods to test
            for method in cls['methods']:
                # Skip private methods unless they look like they need testing
                if method['is_private'] and not self._should_test_private_function(method):
                    continue
                    
                class_target['methods'].append({
                    'name': method['name'],
                    'docstring': method['docstring'],
                    'args': method['args'],
                    'returns': method['returns']
                })
            
            test_targets.append(class_target)
        
        # Don't bother with AI if there's nothing to test
        if not test_targets:
            return None
        
        # Construct prompt for AI
        module_name = module_info['name']
        package_path = os.path.dirname(module_info['path']).replace('/', '.').replace('\\', '.')
        import_path = f"{package_path}.{module_name}" if package_path else module_name
        
        prompt = f"""
        Write pytest test cases for the following Python module:

        ```python
        {module_source}
        ```

        The module name is `{module_name}` and it would be imported as `{import_path}`.

        Please focus on testing these components:
        {self._format_test_targets(test_targets)}

        Generate a comprehensive pytest test file that includes:
        1. Proper imports (including pytest and the module under test)
        2. Test fixtures where appropriate
        3. Separate test functions for each function/method
        4. Tests for normal usage scenarios
        5. Tests for edge cases and error conditions
        6. Appropriate assertions
        7. Docstrings explaining the purpose of each test

        Format your response as a complete Python test file that can be executed with pytest.
        Start the file with a comment that this is an AUTO-GENERATED TEST FILE.
        """
        
        # Request test generation from AI
        response = self.ai_provider.generate_text(
            prompt=prompt,
            provider='anthropic',  # Anthropic tends to be good at code generation
            temperature=0.2,
            max_tokens=2000
        )
        
        if 'error' in response:
            logger.warning(f"Error generating tests with AI: {response['error']}")
            return None
        
        # Extract Python code
        text = response.get('text', '')
        code_match = re.search(r'```python\s+(.*?)```', text, re.DOTALL)
        
        if code_match:
            test_code = code_match.group(1).strip()
            
            # Add warning header if missing
            if not test_code.startswith('#'):
                test_code = f"# AUTO-GENERATED TEST FILE - DO NOT EDIT UNLESS YOU KNOW WHAT YOU'RE DOING\n# Generated by ProjectPilot TestGenerator\n\n{test_code}"
                
            return test_code
        else:
            # Fall back to the whole response if no code block is found
            return text.strip()
    
    def _generate_test_content_from_template(self, module_info: Dict[str, Any]) -> str:
        """
        Generate test content using a template.
        
        Args:
            module_info: Module information
            
        Returns:
            Test content
        """
        module_name = module_info['name']
        package_path = os.path.dirname(module_info['path']).replace('/', '.').replace('\\', '.')
        import_path = f"{package_path}.{module_name}" if package_path else module_name
        
        # Start with header
        content = f"""# AUTO-GENERATED TEST FILE - DO NOT EDIT UNLESS YOU KNOW WHAT YOU'RE DOING
# Generated by ProjectPilot TestGenerator

import pytest
from {import_path} import *

"""
        
        # Add function tests
        for func in module_info['functions']:
            # Skip private functions unless they look like they need testing
            if func['is_private'] and not self._should_test_private_function(func):
                continue
                
            content += self._generate_function_test(func, module_name)
        
        # Add class tests
        for cls in module_info['classes']:
            content += self._generate_class_test(cls, module_name)
        
        return content
    
    def _generate_function_test(self, func_info: Dict[str, Any], module_name: str) -> str:
        """
        Generate test for a function.
        
        Args:
            func_info: Function information
            module_name: Module name
            
        Returns:
            Test code for function
        """
        func_name = func_info['name']
        
        # Create a test function
        result = f"""
def test_{func_name}():
    \"\"\"
    Test for {func_name} function.
    
    This test verifies that the {func_name} function works as expected.
    \"\"\"
    # TODO: Implement test
    pass

"""
        return result
    
    def _generate_class_test(self, class_info: Dict[str, Any], module_name: str) -> str:
        """
        Generate test for a class.
        
        Args:
            class_info: Class information
            module_name: Module name
            
        Returns:
            Test code for class
        """
        class_name = class_info['name']
        
        # Start with class test
        result = f"""
class Test{class_name}:
    \"\"\"
    Tests for {class_name} class.
    \"\"\"
    
    @pytest.fixture
    def {class_name.lower()}_instance(self):
        \"\"\"
        Create a {class_name} instance for testing.
        \"\"\"
        # TODO: Replace with appropriate initialization
        return {class_name}()
        
"""
        
        # Add method tests
        for method in class_info['methods']:
            # Skip private methods unless they look like they need testing
            if method['is_private'] and not self._should_test_private_function(method):
                continue
                
            # Skip special methods
            if method['name'].startswith('__') and method['name'].endswith('__'):
                continue
                
            method_name = method['name']
            
            # Generate test method
            result += f"""    def test_{method_name}(self, {class_name.lower()}_instance):
        \"\"\"
        Test for {method_name} method.
        \"\"\"
        # TODO: Implement test
        pass
        
"""
        
        return result
    
    def _should_test_private_function(self, func_info: Dict[str, Any]) -> bool:
        """
        Determine if a private function should be tested.
        
        Args:
            func_info: Function information
            
        Returns:
            True if the function should be tested
        """
        # Always skip _private functions unless they have a substantial docstring
        if func_info['name'].startswith('_') and not func_info['name'].startswith('__'):
            # Test if it has a substantial docstring
            return len(func_info['docstring']) > 50
        
        # Test all other functions
        return True
    
    def _format_test_targets(self, test_targets: List[Dict[str, Any]]) -> str:
        """
        Format test targets for AI prompt.
        
        Args:
            test_targets: List of test targets
            
        Returns:
            Formatted string
        """
        result = []
        
        for target in test_targets:
            if target['type'] == 'function':
                args = ', '.join(arg['name'] for arg in target['args'])
                result.append(f"Function: {target['name']}({args})")
                if target['docstring']:
                    result.append(f"  Docstring: {target['docstring'].splitlines()[0]}")
                    
            elif target['type'] == 'class':
                result.append(f"Class: {target['name']}")
                if target['docstring']:
                    result.append(f"  Docstring: {target['docstring'].splitlines()[0]}")
                
                for method in target['methods']:
                    args = ', '.join(arg['name'] for arg in method['args'])
                    result.append(f"  Method: {method['name']}({args})")
                    if method['docstring']:
                        result.append(f"    Docstring: {method['docstring'].splitlines()[0]}")
        
        return '\n'.join(result)
    
    def _generate_summary_report(self) -> None:
        """Generate summary report for test generation."""
        report = f"""# Test Generation Summary

Generated: {datetime.now().isoformat()}

## Statistics

- Modules processed: {self.stats['modules_processed']}
- Modules skipped: {self.stats['modules_skipped']}
- Tests generated: {self.stats['tests_generated']}
- Failures: {self.stats['failures']}
- Total functions tested: {self.stats['total_functions_tested']}
- Total classes tested: {self.stats['total_classes_tested']}
- Duration: {self.stats.get('duration_seconds', 0):.2f} seconds

## Output Directory

Tests were written to: `{self.test_output_dir}`

## Next Steps

1. Review the generated tests
2. Implement TODOs and missing test cases
3. Run tests with pytest: `pytest {self.test_output_dir}`
"""
        
        # Write report
        with open(os.path.join(self.test_output_dir, 'test_generation_report.md'), 'w') as f:
            f.write(report)

def run_tests(test_directory: str, module_pattern: Optional[str] = None) -> Dict[str, Any]:
    """
    Run tests in the test directory and return results.
    
    Args:
        test_directory: Directory containing tests
        module_pattern: Pattern to match test modules (e.g., 'test_*.py')
        
    Returns:
        Dictionary with test results
    """
    results = {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'skipped': 0,
        'duration': 0,
        'failing_tests': []
    }
    
    try:
        # Set up command-line arguments
        args = [test_directory]
        if module_pattern:
            args.append(module_pattern)
        
        # Create a temporary file to capture output
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as output_file:
            output_path = output_file.name
        
        start_time = datetime.now()
        
        # Run pytest
        exit_code = pytest.main(args + ['-v', f'--junitxml={output_path}'])
        
        results['duration'] = (datetime.now() - start_time).total_seconds()
        
        # Parse results
        if os.path.exists(output_path):
            # Read XML results
            import xml.etree.ElementTree as ET
            tree = ET.parse(output_path)
            root = tree.getroot()
            
            # Get test summary
            for testsuite in root.findall('.//testsuite'):
                results['total'] += int(testsuite.get('tests', 0))
                results['errors'] += int(testsuite.get('errors', 0))
                results['failures'] += int(testsuite.get('failures', 0))
                results['skipped'] += int(testsuite.get('skipped', 0))
            
            # Calculate passed tests
            results['passed'] = results['total'] - results['errors'] - results['failures'] - results['skipped']
            
            # Get failing tests
            for testcase in root.findall('.//testcase'):
                failure = testcase.find('failure')
                error = testcase.find('error')
                
                if failure is not None or error is not None:
                    results['failing_tests'].append({
                        'name': f"{testcase.get('classname')}.{testcase.get('name')}",
                        'message': (failure.get('message') if failure is not None else error.get('message')),
                        'type': 'failure' if failure is not None else 'error'
                    })
            
            # Clean up
            os.unlink(output_path)
            
        return results
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        results['error'] = str(e)
        return results

if __name__ == '__main__':
    # Example usage
    generator = TestGenerator(project_root='.')
    results = generator.generate_tests()
    print(f"Generated {results['tests_generated']} tests.")
    
    # Run the tests
    test_results = run_tests('./tests')
    print(f"Test results: {test_results['passed']} passed, {test_results['failed']} failed, {test_results['errors']} errors, {test_results['skipped']} skipped")