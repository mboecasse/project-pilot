"""
ProjectPilot - AI-powered project management system
Auto-documentation generator for code and project structure.
"""

import os
import sys
import inspect
import ast
import re
import json
import importlib
import pkgutil
import logging
import time
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from datetime import datetime
import markdown
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from flask import current_app

# Import AI services for enhanced documentation
from app.services._provider import AIProvider

logger = logging.getLogger(__name__)

class DocNode:
    """Represents a node in the documentation structure."""
    
    def __init__(self, name: str, path: str, node_type: str):
        """
        Initialize a documentation node.
        
        Args:
            name: Node name
            path: File system path
            node_type: Type of node ('package', 'module', 'class', 'function', 'method')
        """
        self.name = name
        self.path = path
        self.type = node_type
        self.children = []
        self.docstring = None
        self.code = None
        self.signature = None
        self.parent = None
        self.attributes = {}
    
    def add_child(self, child: 'DocNode') -> None:
        """Add a child node."""
        self.children.append(child)
        child.parent = self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            'name': self.name,
            'type': self.type,
            'path': self.path,
            'docstring': self.docstring,
            'signature': self.signature,
            'attributes': self.attributes,
            'children': [child.to_dict() for child in self.children]
        }

class CodeStructureAnalyzer:
    """Analyzes Python code structure for documentation."""
    
    def __init__(self, root_path: str):
        """
        Initialize code structure analyzer.
        
        Args:
            root_path: Root directory path to analyze
        """
        self.root_path = root_path
        self.doc_tree = DocNode("root", root_path, "package")
    
    def analyze(self) -> DocNode:
        """
        Analyze code structure starting from root path.
        
        Returns:
            DocNode representing the root of the documentation tree
        """
        self._process_directory(self.root_path, self.doc_tree)
        return self.doc_tree
    
    def _process_directory(self, dir_path: str, parent_node: DocNode) -> None:
        """
        Process a directory and its contents.
        
        Args:
            dir_path: Directory path
            parent_node: Parent doc node
        """
        for item in os.listdir(dir_path):
            full_path = os.path.join(dir_path, item)
            
            # Skip hidden files and directories
            if item.startswith('.'):
                continue
                
            # Skip __pycache__ and similar
            if item == '__pycache__' or item.endswith('.pyc'):
                continue
                
            if os.path.isdir(full_path):
                # Check if it's a Python package
                if os.path.exists(os.path.join(full_path, '__init__.py')):
                    node = DocNode(item, full_path, "package")
                    parent_node.add_child(node)
                    self._process_directory(full_path, node)
            elif item.endswith('.py'):
                self._process_python_file(full_path, parent_node)
    
    def _process_python_file(self, file_path: str, parent_node: DocNode) -> None:
        """
        Process a Python file.
        
        Args:
            file_path: Python file path
            parent_node: Parent doc node
        """
        file_name = os.path.basename(file_path)
        
        # Skip __init__.py for brevity in documentation
        if file_name == '__init__.py':
            # Still process it but don't create a separate node
            self._extract_module_info(file_path, parent_node)
            return
            
        module_name = file_name[:-3]  # Remove .py extension
        module_node = DocNode(module_name, file_path, "module")
        parent_node.add_child(module_node)
        
        # Extract module info
        self._extract_module_info(file_path, module_node)
    
    def _extract_module_info(self, file_path: str, module_node: DocNode) -> None:
        """
        Extract information from a Python module.
        
        Args:
            file_path: Python file path
            module_node: Module doc node
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
            tree = ast.parse(file_content)
            
            # Extract module docstring
            module_node.docstring = ast.get_docstring(tree)
            module_node.code = file_content
            
            # Process classes and functions
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    self._process_class(node, file_content, module_node)
                elif isinstance(node, ast.FunctionDef):
                    self._process_function(node, file_content, module_node)
                    
        except Exception as e:
            logger.error(f"Error processing module {file_path}: {str(e)}")
            module_node.attributes['error'] = str(e)
    
    def _process_class(self, node: ast.ClassDef, source: str, parent_node: DocNode) -> None:
        """
        Process a class definition.
        
        Args:
            node: AST class node
            source: Source code
            parent_node: Parent doc node
        """
        class_node = DocNode(node.name, parent_node.path, "class")
        parent_node.add_child(class_node)
        
        # Extract class docstring
        class_node.docstring = ast.get_docstring(node)
        
        # Extract class code
        class_node.code = self._extract_source_segment(source, node)
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._extract_attribute_name(base)}")
        
        class_node.attributes['bases'] = bases
        
        # Process methods and nested classes
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._process_method(item, source, class_node)
            elif isinstance(item, ast.ClassDef):
                self._process_class(item, source, class_node)
    
    def _process_function(self, node: ast.FunctionDef, source: str, parent_node: DocNode) -> None:
        """
        Process a function definition.
        
        Args:
            node: AST function node
            source: Source code
            parent_node: Parent doc node
        """
        func_node = DocNode(node.name, parent_node.path, "function")
        parent_node.add_child(func_node)
        
        # Extract function docstring
        func_node.docstring = ast.get_docstring(node)
        
        # Extract function code
        func_node.code = self._extract_source_segment(source, node)
        
        # Extract function signature
        func_node.signature = self._extract_function_signature(node)
        
        # Extract return annotation if available
        if node.returns:
            if isinstance(node.returns, ast.Name):
                func_node.attributes['returns'] = node.returns.id
            elif isinstance(node.returns, ast.Subscript):
                func_node.attributes['returns'] = self._extract_subscript_name(node.returns)
    
    def _process_method(self, node: ast.FunctionDef, source: str, parent_node: DocNode) -> None:
        """
        Process a method definition.
        
        Args:
            node: AST function node
            source: Source code
            parent_node: Parent doc node
        """
        method_node = DocNode(node.name, parent_node.path, "method")
        parent_node.add_child(method_node)
        
        # Extract method docstring
        method_node.docstring = ast.get_docstring(node)
        
        # Extract method code
        method_node.code = self._extract_source_segment(source, node)
        
        # Extract method signature
        method_node.signature = self._extract_function_signature(node)
        
        # Extract return annotation if available
        if node.returns:
            if isinstance(node.returns, ast.Name):
                method_node.attributes['returns'] = node.returns.id
            elif isinstance(node.returns, ast.Subscript):
                method_node.attributes['returns'] = self._extract_subscript_name(node.returns)
    
    def _extract_source_segment(self, source: str, node: ast.AST) -> str:
        """
        Extract source code segment for an AST node.
        
        Args:
            source: Complete source code
            node: AST node
            
        Returns:
            Source code segment
        """
        start_line = node.lineno - 1
        end_line = node.end_lineno
        
        lines = source.splitlines()
        return '\n'.join(lines[start_line:end_line])
    
    def _extract_function_signature(self, node: ast.FunctionDef) -> str:
        """
        Extract function signature.
        
        Args:
            node: AST function node
            
        Returns:
            Function signature as string
        """
        args_list = []
        
        # Add positional args
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_str += ': ' + arg.annotation.id
                elif isinstance(arg.annotation, ast.Subscript):
                    arg_str += ': ' + self._extract_subscript_name(arg.annotation)
            args_list.append(arg_str)
        
        # Add *args if present
        if node.args.vararg:
            vararg_str = '*' + node.args.vararg.arg
            if node.args.vararg.annotation:
                if isinstance(node.args.vararg.annotation, ast.Name):
                    vararg_str += ': ' + node.args.vararg.annotation.id
                elif isinstance(node.args.vararg.annotation, ast.Subscript):
                    vararg_str += ': ' + self._extract_subscript_name(node.args.vararg.annotation)
            args_list.append(vararg_str)
        
        # Add keyword args
        for arg in node.args.kwonlyargs:
            arg_str = arg.arg
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_str += ': ' + arg.annotation.id
                elif isinstance(arg.annotation, ast.Subscript):
                    arg_str += ': ' + self._extract_subscript_name(arg.annotation)
            args_list.append(arg_str)
        
        # Add **kwargs if present
        if node.args.kwarg:
            kwarg_str = '**' + node.args.kwarg.arg
            if node.args.kwarg.annotation:
                if isinstance(node.args.kwarg.annotation, ast.Name):
                    kwarg_str += ': ' + node.args.kwarg.annotation.id
                elif isinstance(node.args.kwarg.annotation, ast.Subscript):
                    kwarg_str += ': ' + self._extract_subscript_name(node.args.kwarg.annotation)
            args_list.append(kwarg_str)
        
        # Combine everything
        signature = f"def {node.name}({', '.join(args_list)})"
        
        # Add return type if specified
        if node.returns:
            if isinstance(node.returns, ast.Name):
                signature += f" -> {node.returns.id}"
            elif isinstance(node.returns, ast.Subscript):
                signature += f" -> {self._extract_subscript_name(node.returns)}"
        
        return signature
    
    def _extract_subscript_name(self, node: ast.Subscript) -> str:
        """
        Extract name from a subscript node (e.g., List[str]).
        
        Args:
            node: AST subscript node
            
        Returns:
            Subscript name as string
        """
        if isinstance(node.value, ast.Name):
            value_name = node.value.id
        else:
            value_name = "Complex"
        
        if isinstance(node.slice, ast.Name):
            slice_name = node.slice.id
            return f"{value_name}[{slice_name}]"
        elif isinstance(node.slice, ast.Subscript):
            slice_name = self._extract_subscript_name(node.slice)
            return f"{value_name}[{slice_name}]"
        elif hasattr(node.slice, 'value') and isinstance(node.slice.value, ast.Name):
            # For Python 3.8 compatibility
            slice_name = node.slice.value.id
            return f"{value_name}[{slice_name}]"
        elif isinstance(node.slice, ast.Tuple):
            elts = []
            for elt in node.slice.elts:
                if isinstance(elt, ast.Name):
                    elts.append(elt.id)
                elif isinstance(elt, ast.Subscript):
                    elts.append(self._extract_subscript_name(elt))
            return f"{value_name}[{', '.join(elts)}]"
        else:
            return f"{value_name}[...]"
    
    def _extract_attribute_name(self, node: ast.Attribute) -> str:
        """
        Extract name from an attribute node (e.g., module.Class).
        
        Args:
            node: AST attribute node
            
        Returns:
            Attribute name as string
        """
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._extract_attribute_name(node.value)}.{node.attr}"
        else:
            return f"?.{node.attr}"

class DocumentationGenerator:
    """
    Generates comprehensive documentation for the project.
    Uses AI to enhance documentation where needed.
    """
    
    def __init__(self, 
                 project_root: str,
                 output_dir: str = 'docs',
                 ai_provider: Optional[AIProvider] = None):
        """
        Initialize documentation generator.
        
        Args:
            project_root: Project root directory
            output_dir: Output directory for documentation
            ai_provider: AI provider for enhancing documentation
        """
        self.project_root = project_root
        self.output_dir = os.path.join(project_root, output_dir)
        self.ai_provider = ai_provider or AIProvider()
        self.analyzer = CodeStructureAnalyzer(project_root)
        self.doc_tree = None
        self.auto_generated_docs = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Documentation generator initialized with root: {project_root}")
    
    def generate(self, 
                 enhance_docstrings: bool = True,
                 include_tests: bool = True,
                 diagram_types: List[str] = ['module', 'class', 'workflow']) -> Dict[str, Any]:
        """
        Generate project documentation.
        
        Args:
            enhance_docstrings: Whether to enhance docstrings with AI
            include_tests: Whether to include test generation
            diagram_types: Types of diagrams to generate
            
        Returns:
            Dictionary with generation statistics
        """
        start_time = time.time()
        
        # Analyze code structure
        self.doc_tree = self.analyzer.analyze()
        
        # Create index page
        self._create_index_page()
        
        # Create documentation pages for each component
        file_count = self._create_component_pages(self.doc_tree)
        
        # Create API reference
        self._create_api_reference()
        
        # Generate diagrams
        diagram_count = self._generate_diagrams(diagram_types)
        
        # Enhance docstrings with AI if requested
        enhanced_count = 0
        if enhance_docstrings:
            enhanced_count = self._enhance_docstrings()
        
        # Generate tests if requested
        test_count = 0
        if include_tests:
            test_count = self._generate_tests()
        
        # Create search index
        self._create_search_index()
        
        # Generate completion summary
        elapsed_time = time.time() - start_time
        summary = {
            'timestamp': datetime.now().isoformat(),
            'file_count': file_count,
            'diagram_count': diagram_count,
            'enhanced_docstrings': enhanced_count,
            'test_count': test_count,
            'elapsed_time': elapsed_time,
            'output_directory': self.output_dir
        }
        
        # Save summary
        with open(os.path.join(self.output_dir, 'generation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Documentation generated: {file_count} files, {diagram_count} diagrams, {enhanced_count} enhanced docstrings")
        
        return summary
    
    def _create_index_page(self) -> None:
        """Create index page for documentation."""
        # Create title and header
        project_name = os.path.basename(self.project_root)
        index_content = f"""# {project_name} Documentation

## Overview

This documentation provides a comprehensive reference to the {project_name} codebase.

### Project Structure

```
{self._generate_project_tree()}
```

### Key Components

{self._generate_key_components_list()}

### Getting Started

- [API Reference](api_reference.md)
- [Module Index](module_index.md)
- [Class Index](class_index.md)
- [Workflow Diagrams](diagrams/workflows.md)
"""
        
        # Write index file
        with open(os.path.join(self.output_dir, 'index.md'), 'w') as f:
            f.write(index_content)
        
        # Generate HTML version
        html_content = markdown.markdown(index_content)
        with open(os.path.join(self.output_dir, 'index.html'), 'w') as f:
            f.write(self._wrap_html(html_content, f"{project_name} Documentation"))
    
    def _create_component_pages(self, node: DocNode, parent_path: str = '') -> int:
        """
        Create documentation pages for each component.
        
        Args:
            node: Current doc node
            parent_path: Parent path for creating directory structure
            
        Returns:
            Number of files created
        """
        file_count = 0
        
        if node.type in ['package', 'module']:
            # Create directory for package
            current_path = os.path.join(parent_path, node.name) if node.name != 'root' else ''
            component_dir = os.path.join(self.output_dir, current_path)
            os.makedirs(component_dir, exist_ok=True)
            
            # Create module page
            if node.type == 'module':
                file_path = os.path.join(self.output_dir, f"{current_path}.md")
                file_count += self._create_module_page(node, file_path)
        
        # Process children
        for child in node.children:
            file_count += self._create_component_pages(child, 
                                                    os.path.join(parent_path, node.name) if node.name != 'root' else '')
        
        return file_count
    
    def _create_module_page(self, node: DocNode, file_path: str) -> int:
        """
        Create documentation page for a module.
        
        Args:
            node: Module doc node
            file_path: Output file path
            
        Returns:
            1 if file created, 0 otherwise
        """
        try:
            module_name = node.name
            
            # Start with module docstring
            content = f"# Module: {module_name}\n\n"
            
            if node.docstring:
                content += f"{node.docstring}\n\n"
            else:
                content += "*No module docstring provided.*\n\n"
            
            # Add file path
            rel_path = os.path.relpath(node.path, self.project_root)
            content += f"**File:** `{rel_path}`\n\n"
            
            # Add classes
            classes = [child for child in node.children if child.type == 'class']
            if classes:
                content += "## Classes\n\n"
                for cls in classes:
                    content += f"### {cls.name}\n\n"
                    
                    if cls.attributes.get('bases'):
                        bases = cls.attributes['bases']
                        content += f"*Inherits from: {', '.join(bases)}*\n\n"
                    
                    if cls.docstring:
                        content += f"{cls.docstring}\n\n"
                    else:
                        content += "*No class docstring provided.*\n\n"
                    
                    # Add methods
                    methods = [child for child in cls.children if child.type == 'method']
                    if methods:
                        content += "#### Methods\n\n"
                        for method in methods:
                            content += f"##### `{method.signature}`\n\n"
                            
                            if method.docstring:
                                content += f"{method.docstring}\n\n"
                            else:
                                content += "*No method docstring provided.*\n\n"
            
            # Add functions
            functions = [child for child in node.children if child.type == 'function']
            if functions:
                content += "## Functions\n\n"
                for func in functions:
                    content += f"### `{func.signature}`\n\n"
                    
                    if func.docstring:
                        content += f"{func.docstring}\n\n"
                    else:
                        content += "*No function docstring provided.*\n\n"
            
            # Write to file
            with open(file_path, 'w') as f:
                f.write(content)
            
            # Generate HTML version
            html_content = markdown.markdown(content)
            html_path = file_path.replace('.md', '.html')
            with open(html_path, 'w') as f:
                f.write(self._wrap_html(html_content, f"Module: {module_name}"))
            
            return 1
            
        except Exception as e:
            logger.error(f"Error creating module page for {node.name}: {str(e)}")
            return 0
    
    def _create_api_reference(self) -> None:
        """Create API reference documentation."""
        content = "# API Reference\n\n"
        
        # Create module index
        content += "## Module Index\n\n"
        modules = self._find_all_nodes_by_type('module')
        
        for module in sorted(modules, key=lambda x: x.name):
            rel_path = os.path.relpath(module.path, self.project_root)
            content += f"- [{module.name}]({rel_path.replace('.py', '.md')})\n"
        
        content += "\n## Class Index\n\n"
        classes = self._find_all_nodes_by_type('class')
        
        for cls in sorted(classes, key=lambda x: x.name):
            # Find parent module
            parent = cls.parent
            while parent and parent.type != 'module':
                parent = parent.parent
            
            if parent:
                module_rel_path = os.path.relpath(parent.path, self.project_root)
                content += f"- [{cls.name}]({module_rel_path.replace('.py', '.md')}#class-{cls.name.lower()})\n"
        
        # Write to file
        api_ref_path = os.path.join(self.output_dir, 'api_reference.md')
        with open(api_ref_path, 'w') as f:
            f.write(content)
        
        # Generate HTML version
        html_content = markdown.markdown(content)
        with open(api_ref_path.replace('.md', '.html'), 'w') as f:
            f.write(self._wrap_html(html_content, "API Reference"))
    
    def _generate_diagrams(self, diagram_types: List[str]) -> int:
        """
        Generate diagrams for documentation.
        
        Args:
            diagram_types: Types of diagrams to generate
            
        Returns:
            Number of diagrams generated
        """
        diagram_count = 0
        diagrams_dir = os.path.join(self.output_dir, 'diagrams')
        os.makedirs(diagrams_dir, exist_ok=True)
        
        if 'module' in diagram_types:
            # Generate module dependency diagram
            diagram_count += self._generate_module_diagram(diagrams_dir)
        
        if 'class' in diagram_types:
            # Generate class hierarchy diagram
            diagram_count += self._generate_class_diagram(diagrams_dir)
        
        if 'workflow' in diagram_types:
            # Generate workflow diagrams
            diagram_count += self._generate_workflow_diagrams(diagrams_dir)
        
        return diagram_count
    
    def _generate_module_diagram(self, diagrams_dir: str) -> int:
        """
        Generate module dependency diagram.
        
        Args:
            diagrams_dir: Diagrams directory
            
        Returns:
            1 if diagram generated, 0 otherwise
        """
        try:
            # Create Mermaid diagram
            modules = self._find_all_nodes_by_type('module')
            
            diagram = "```mermaid\ngraph TD\n"
            
            # Add nodes
            for module in modules:
                module_id = module.name.replace('.', '_')
                diagram += f"    {module_id}[{module.name}]\n"
            
            # Add edges by analyzing imports
            for module in modules:
                if not module.code:
                    continue
                    
                module_id = module.name.replace('.', '_')
                
                # Parse imports from the code
                imports = self._extract_imports(module.code)
                
                for imp in imports:
                    # Check if it's an internal module
                    for target in modules:
                        if imp == target.name:
                            target_id = target.name.replace('.', '_')
                            diagram += f"    {module_id} --> {target_id}\n"
                            break
            
            diagram += "```\n"
            
            # Write diagram to file
            with open(os.path.join(diagrams_dir, 'module_dependencies.md'), 'w') as f:
                f.write("# Module Dependencies\n\n")
                f.write(diagram)
            
            return 1
            
        except Exception as e:
            logger.error(f"Error generating module diagram: {str(e)}")
            return 0
    
    def _generate_class_diagram(self, diagrams_dir: str) -> int:
        """
        Generate class hierarchy diagram.
        
        Args:
            diagrams_dir: Diagrams directory
            
        Returns:
            1 if diagram generated, 0 otherwise
        """
        try:
            # Create Mermaid diagram
            classes = self._find_all_nodes_by_type('class')
            
            diagram = "```mermaid\nclassDiagram\n"
            
            # Create class hierarchy
            for cls in classes:
                # Add class name
                diagram += f"    class {cls.name}\n"
                
                # Add inheritance relationships
                if cls.attributes.get('bases'):
                    for base in cls.attributes['bases']:
                        # Skip external bases
                        if any(c.name == base for c in classes):
                            diagram += f"    {base} <|-- {cls.name}\n"
                
                # Add methods
                methods = [child for child in cls.children if child.type == 'method']
                for method in methods:
                    # Clean signature for mermaid format
                    clean_sig = method.name + "()"
                    diagram += f"    {cls.name} : {clean_sig}\n"
            
            diagram += "```\n"
            
            # Write diagram to file
            with open(os.path.join(diagrams_dir, 'class_hierarchy.md'), 'w') as f:
                f.write("# Class Hierarchy\n\n")
                f.write(diagram)
            
            return 1
            
        except Exception as e:
            logger.error(f"Error generating class diagram: {str(e)}")
            return 0
    
    def _generate_workflow_diagrams(self, diagrams_dir: str) -> int:
        """
        Generate workflow diagrams.
        
        Args:
            diagrams_dir: Diagrams directory
            
        Returns:
            Number of diagrams generated
        """
        try:
            # Find key workflow classes/modules
            workflow_keywords = ['workflow', 'process', 'pipeline', 'service', 'manager', 'bot']
            
            potential_workflows = []
            modules = self._find_all_nodes_by_type('module')
            classes = self._find_all_nodes_by_type('class')
            
            # Find by name matching
            for module in modules:
                if any(keyword in module.name.lower() for keyword in workflow_keywords):
                    potential_workflows.append(module)
                    
            for cls in classes:
                if any(keyword in cls.name.lower() for keyword in workflow_keywords):
                    potential_workflows.append(cls)
            
            if not potential_workflows:
                logger.warning("No potential workflows found")
                return 0
            
            # For each potential workflow, create a diagram
            diagram_count = 0
            
            for workflow in potential_workflows:
                # Generate workflow diagram using AI
                if self.ai_provider:
                    diagram = self._generate_workflow_with_ai(workflow)
                    
                    if diagram:
                        # Write diagram to file
                        filename = f"{workflow.name.lower()}_workflow.md"
                        with open(os.path.join(diagrams_dir, filename), 'w') as f:
                            f.write(f"# {workflow.name} Workflow\n\n")
                            f.write(diagram)
                        
                        diagram_count += 1
            
            # Create workflows index
            if diagram_count > 0:
                with open(os.path.join(diagrams_dir, 'workflows.md'), 'w') as f:
                    f.write("# Workflow Diagrams\n\n")
                    
                    for workflow in potential_workflows:
                        filename = f"{workflow.name.lower()}_workflow.md"
                        if os.path.exists(os.path.join(diagrams_dir, filename)):
                            f.write(f"- [{workflow.name} Workflow]({filename})\n")
            
            return diagram_count
            
        except Exception as e:
            logger.error(f"Error generating workflow diagrams: {str(e)}")
            return 0
    
    def _generate_workflow_with_ai(self, node: DocNode) -> Optional[str]:
        """
        Generate workflow diagram using AI.
        
        Args:
            node: DocNode representing a workflow component
            
        Returns:
            Mermaid diagram string or None if generation failed
        """
        try:
            # Get code and docstring for context
            code = node.code
            docstring = node.docstring or ""
            
            if not code:
                # If we don't have code, we can't generate a workflow
                return None
            
            prompt = f"""
            Analyze the following Python code and create a workflow diagram using Mermaid syntax:

            ```python
            {code}
            ```

            {docstring}

            Create a flowchart or sequence diagram in Mermaid syntax that visualizes the workflow or process implemented in this code.
            Focus on the key steps, data flow, and decision points.
            Use appropriate shapes and connectors to represent the process flow.
            
            Format your response as a Mermaid diagram code block like this:
            ```mermaid
            graph TD
                A[Start] --> B[Process]
                B --> C[End]
            ```
            
            OR
            
            ```mermaid
            sequenceDiagram
                participant A as Component A
                participant B as Component B
                A->>B: Action
            ```
            
            Return only the Mermaid code block and nothing else.
            """
            
            response = self.ai_provider.generate_text(
                prompt=prompt,
                provider='anthropic',  # Anthropic tends to be good at diagrams
                temperature=0.2,
                max_tokens=1000
            )
            
            if 'error' in response:
                logger.warning(f"Error generating workflow diagram for {node.name}: {response['error']}")
                return None
            
            # Extract mermaid code block
            text = response.get('text', '')
            mermaid_match = re.search(r'```mermaid\s+(.*?)```', text, re.DOTALL)
            
            if mermaid_match:
                mermaid_code = mermaid_match.group(1).strip()
                return f"```mermaid\n{mermaid_code}\n```"
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating workflow with AI for {node.name}: {str(e)}")
            return None
    
    def _enhance_docstrings(self) -> int:
        """
        Enhance docstrings with AI where missing or incomplete.
        
        Returns:
            Number of docstrings enhanced
        """
        if not self.ai_provider:
            logger.warning("AI provider not available, skipping docstring enhancement")
            return 0
        
        try:
            enhanced_count = 0
            
            # Find nodes with missing or minimal docstrings
            all_nodes = []
            all_nodes.extend(self._find_all_nodes_by_type('module'))
            all_nodes.extend(self._find_all_nodes_by_type('class'))
            all_nodes.extend(self._find_all_nodes_by_type('function'))
            all_nodes.extend(self._find_all_nodes_by_type('method'))
            
            for node in all_nodes:
                if not node.docstring or len(node.docstring.strip()) < 30:
                    if node.code:
                        # Generate docstring with AI
                        improved_docstring = self._generate_docstring_with_ai(node)
                        
                        if improved_docstring:
                            # Store the auto-generated docstring
                            self.auto_generated_docs[f"{node.type}:{node.name}"] = improved_docstring
                            node.docstring = improved_docstring
                            enhanced_count += 1
            
            # Save auto-generated docstrings for reference
            if enhanced_count > 0:
                with open(os.path.join(self.output_dir, 'auto_generated_docs.json'), 'w') as f:
                    json.dump(self.auto_generated_docs, f, indent=2)
            
            return enhanced_count
            
        except Exception as e:
            logger.error(f"Error enhancing docstrings: {str(e)}")
            return 0
    
    def _generate_docstring_with_ai(self, node: DocNode) -> Optional[str]:
        """
        Generate docstring for a node using AI.
        
        Args:
            node: DocNode to generate docstring for
            
        Returns:
            Generated docstring or None if generation failed
        """
        try:
            # Get code for context
            code = node.code
            
            if not code:
                return None
            
            prompt = f"""
            Write a comprehensive docstring for the following Python {node.type}:

            ```python
            {code}
            ```

            Write the docstring in Google-style format with the following sections:
            - Brief description
            - Args (for functions/methods)
            - Returns (for functions/methods)
            - Raises (if applicable)
            - Examples (if helpful)

            For classes, include:
            - Brief description
            - Attributes
            - Methods overview

            Be concise but thorough. Focus on explaining the purpose, inputs, outputs, and behavior.
            Do not include any markdown formatting, just the plain text docstring.
            """
            
            response = self.ai_provider.generate_text(
                prompt=prompt,
                provider='auto',
                temperature=0.1,
                max_tokens=500
            )
            
            if 'error' in response:
                logger.warning(f"Error generating docstring for {node.name}: {response['error']}")
                return None
            
            # Clean up the response
            docstring = response.get('text', '').strip()
            
            # Remove any markdown formatting
            docstring = re.sub(r'^```.*, '', docstring, flags=re.MULTILINE)
            docstring = re.sub(r'^```, '', docstring, flags=re.MULTILINE)
            
            return docstring.strip()
                
        except Exception as e:
            logger.error(f"Error generating docstring with AI for {node.name}: {str(e)}")
            return None
    
    def _generate_tests(self) -> int:
        """
        Generate test cases for functions and methods.
        
        Returns:
            Number of test files generated
        """
        if not self.ai_provider:
            logger.warning("AI provider not available, skipping test generation")
            return 0
        
        try:
            test_count = 0
            tests_dir = os.path.join(self.output_dir, 'tests')
            os.makedirs(tests_dir, exist_ok=True)
            
            # Find all modules
            modules = self._find_all_nodes_by_type('module')
            
            for module in modules:
                rel_path = os.path.relpath(module.path, self.project_root)
                
                # Skip test files themselves
                if 'test_' in module.name or module.name.endswith('_test'):
                    continue
                
                # Generate test file
                test_file_path = os.path.join(tests_dir, f"test_{module.name}.py")
                
                # Generate test content
                test_content = self._generate_test_file_with_ai(module)
                
                if test_content:
                    with open(test_file_path, 'w') as f:
                        f.write(test_content)
                    test_count += 1
            
            # Generate test index
            if test_count > 0:
                with open(os.path.join(tests_dir, 'index.md'), 'w') as f:
                    f.write("# Generated Test Cases\n\n")
                    f.write("This directory contains automatically generated test cases for the project.\n\n")
                    f.write("## Test Files\n\n")
                    
                    for test_file in sorted(os.listdir(tests_dir)):
                        if test_file.startswith('test_') and test_file.endswith('.py'):
                            f.write(f"- [{test_file}]({test_file})\n")
            
            return test_count
            
        except Exception as e:
            logger.error(f"Error generating tests: {str(e)}")
            return 0
    
    def _generate_test_file_with_ai(self, module: DocNode) -> Optional[str]:
        """
        Generate test file for a module using AI.
        
        Args:
            module: Module DocNode
            
        Returns:
            Test file content or None if generation failed
        """
        try:
            # Get code and docstring for context
            code = module.code
            
            if not code:
                return None
            
            # Get all functions and classes that need testing
            test_targets = []
            
            # Add functions
            functions = [child for child in module.children if child.type == 'function']
            for func in functions:
                if not func.name.startswith('_'):  # Skip private functions
                    test_targets.append((func.name, 'function', func.signature, func.docstring))
            
            # Add classes and methods
            classes = [child for child in module.children if child.type == 'class']
            for cls in classes:
                if not cls.name.startswith('_'):  # Skip private classes
                    methods = [child for child in cls.children if child.type == 'method']
                    for method in methods:
                        if not method.name.startswith('_'):  # Skip private methods
                            test_targets.append((f"{cls.name}.{method.name}", 'method', method.signature, method.docstring))
            
            if not test_targets:
                return None
            
            prompt = f"""
            Write a comprehensive pytest test file for the following Python module:

            ```python
            {code}
            ```

            The test file should include test cases for the following functions/methods:
            {json.dumps(test_targets, indent=2)}

            For each function/method:
            1. Create test cases for normal usage
            2. Create test cases for edge cases
            3. Create test cases for error conditions
            4. Include appropriate assertions
            5. Use pytest fixtures where appropriate
            6. Include test docstrings explaining each test case

            Format the test file as valid Python code with proper imports and structure.
            Include a test_* function for each function/method.
            """
            
            response = self.ai_provider.generate_text(
                prompt=prompt,
                provider='anthropic',  # Anthropic tends to be good at code generation
                temperature=0.2,
                max_tokens=2000
            )
            
            if 'error' in response:
                logger.warning(f"Error generating tests for {module.name}: {response['error']}")
                return None
            
            # Extract Python code block
            text = response.get('text', '')
            code_match = re.search(r'```python\s+(.*?)```', text, re.DOTALL)
            
            if code_match:
                test_code = code_match.group(1).strip()
                return test_code
            else:
                # Fall back to the whole response if no code block is found
                return text.strip()
                
        except Exception as e:
            logger.error(f"Error generating tests with AI for {module.name}: {str(e)}")
            return None
    
    def _create_search_index(self) -> None:
        """Create search index for documentation."""
        search_index = []
        
        # Index modules
        for module in self._find_all_nodes_by_type('module'):
            search_index.append({
                'title': f"Module: {module.name}",
                'path': os.path.relpath(module.path, self.project_root).replace('.py', '.html'),
                'summary': module.docstring[:100] + '...' if module.docstring and len(module.docstring) > 100 else module.docstring or 'No description',
                'type': 'module'
            })
        
        # Index classes
        for cls in self._find_all_nodes_by_type('class'):
            # Find parent module
            parent = cls.parent
            while parent and parent.type != 'module':
                parent = parent.parent
            
            if parent:
                search_index.append({
                    'title': f"Class: {cls.name}",
                    'path': os.path.relpath(parent.path, self.project_root).replace('.py', '.html') + f"#class-{cls.name.lower()}",
                    'summary': cls.docstring[:100] + '...' if cls.docstring and len(cls.docstring) > 100 else cls.docstring or 'No description',
                    'type': 'class'
                })
        
        # Write search index
        with open(os.path.join(self.output_dir, 'search_index.json'), 'w') as f:
            json.dump(search_index, f, indent=2)
    
    def _find_all_nodes_by_type(self, node_type: str) -> List[DocNode]:
        """
        Find all nodes of a specific type in the doc tree.
        
        Args:
            node_type: Type of node to find
            
        Returns:
            List of matching nodes
        """
        result = []
        
        def _traverse(node):
            if node.type == node_type:
                result.append(node)
            for child in node.children:
                _traverse(child)
        
        _traverse(self.doc_tree)
        return result
    
    def _generate_project_tree(self, max_depth: int = 3) -> str:
        """
        Generate ASCII tree representation of the project structure.
        
        Args:
            max_depth: Maximum depth to display
            
        Returns:
            ASCII tree as string
        """
        tree_str = []
        
        def _traverse(dir_path, prefix='', depth=0):
            if depth > max_depth:
                return
                
            items = sorted(os.listdir(dir_path))
            
            # Filter out non-Python files and directories
            python_items = []
            for item in items:
                if item.startswith('.'):
                    continue
                if item == '__pycache__' or item.endswith('.pyc'):
                    continue
                
                full_path = os.path.join(dir_path, item)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, '__init__.py')):
                    python_items.append(item)
                elif item.endswith('.py'):
                    python_items.append(item)
            
            for i, item in enumerate(python_items):
                is_last = i == len(python_items) - 1
                tree_str.append(prefix + ('└── ' if is_last else '├── ') + item)
                
                full_path = os.path.join(dir_path, item)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, '__init__.py')):
                    _traverse(
                        full_path, 
                        prefix + ('    ' if is_last else '│   '), 
                        depth + 1
                    )
        
        _traverse(self.project_root)
        return '\n'.join(tree_str)
    
    def _generate_key_components_list(self) -> str:
        """
        Generate list of key components in the project.
        
        Returns:
            Markdown list of key components
        """
        result = []
        
        # Find modules with interesting names
        key_modules = []
        for module in self._find_all_nodes_by_type('module'):
            module_name = module.name.lower()
            if any(keyword in module_name for keyword in ['service', 'manager', 'utils', 'model', 'config']):
                key_modules.append(module)
        
        # Sort by name
        key_modules.sort(key=lambda x: x.name)
        
        # Generate list items
        for module in key_modules:
            rel_path = os.path.relpath(module.path, self.project_root)
            summary = module.docstring.split('\n')[0] if module.docstring else 'No description'
            result.append(f"- [{module.name}]({rel_path.replace('.py', '.md')}) - {summary}")
        
        return '\n'.join(result)
    
    def _extract_imports(self, code: str) -> List[str]:
        """
        Extract import names from Python code.
        
        Args:
            code: Python code
            
        Returns:
            List of imported module names
        """
        imports = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            # If parsing fails, use regex as fallback
            import_regex = r'(?:from|import)\s+([a-zA-Z0-9_.]+)(?:\s+import|\s*,|\s*$)'
            matches = re.findall(import_regex, code)
            imports.extend(matches)
        
        return imports
    
    def _wrap_html(self, content: str, title: str) -> str:
        """
        Wrap markdown content in HTML template.
        
        Args:
            content: HTML content
            title: Page title
            
        Returns:
            Full HTML document
        """
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
        }}
        pre {{
            background-color: #f6f8fa;
            border-radius: 3px;
            padding: 16px;
            overflow: auto;
        }}
        code {{
            background-color: rgba(27, 31, 35, 0.05);
            border-radius: 3px;
            padding: 0.2em 0.4em;
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        }}
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}
        h1 {{ font-size: 2em; padding-bottom: 0.3em; border-bottom: 1px solid #eaecef; }}
        h2 {{ font-size: 1.5em; padding-bottom: 0.3em; border-bottom: 1px solid #eaecef; }}
        h3 {{ font-size: 1.25em; }}
        a {{ color: #0366d6; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 8px; border: 1px solid #dfe2e5; }}
        tr:nth-child(even) {{ background-color: #f6f8fa; }}
        .sidebar {{ 
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            width: 250px;
            overflow-y: auto;
            padding: 20px;
            background-color: #f6f8fa;
            border-right: 1px solid #dfe2e5;
        }}
        .content {{
            margin-left: 250px;
            padding: 20px;
        }}
        .search {{
            margin-bottom: 20px;
        }}
        .search input {{
            width: 100%;
            padding: 8px;
            border: 1px solid #dfe2e5;
            border-radius: 3px;
        }}
        @media (max-width: 768px) {{
            .sidebar {{
                display: none;
            }}
            .content {{
                margin-left: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="content">
        {content}
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Add table of contents
            const headings = document.querySelectorAll('h2, h3');
            if (headings.length > 0) {{
                const toc = document.createElement('div');
                toc.innerHTML = '<h2>Table of Contents</h2><ul id="toc-list"></ul>';
                const tocList = toc.querySelector('#toc-list');
                
                headings.forEach(function(heading) {{
                    const id = heading.textContent.toLowerCase().replace(/[^a-z0-9]+/g, '-');
                    heading.id = id;
                    
                    const li = document.createElement('li');
                    const a = document.createElement('a');
                    a.href = '#' + id;
                    a.textContent = heading.textContent;
                    
                    if (heading.tagName === 'H3') {{
                        li.style.marginLeft = '20px';
                    }}
                    
                    li.appendChild(a);
                    tocList.appendChild(li);
                }});
                
                const content = document.querySelector('.content');
                content.insertBefore(toc, content.firstChild.nextSibling);
            }}
        }});
    </script>
</body>
</html>"""