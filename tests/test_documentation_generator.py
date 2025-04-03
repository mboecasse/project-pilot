"""
Test file for the DocumentationGenerator utility.
"""

import os
import sys
import pytest
import tempfile
import json
from unittest.mock import patch, MagicMock
import shutil

# Import the module to test
from app.utils.documentation_generator import DocumentationGenerator, CodeStructureAnalyzer, DocNode

class TestDocNode:
    """Test the DocNode class."""
    
    def test_init(self):
        """Test DocNode initialization."""
        node = DocNode("test_node", "/path/to/file", "module")
        assert node.name == "test_node"
        assert node.path == "/path/to/file"
        assert node.type == "module"
        assert node.children == []
        assert node.docstring is None
        assert node.code is None
        assert node.signature is None
        assert node.parent is None
        assert node.attributes == {}
    
    def test_add_child(self):
        """Test adding a child node."""
        parent = DocNode("parent", "/path", "package")
        child = DocNode("child", "/path/child", "module")
        
        parent.add_child(child)
        
        assert child in parent.children
        assert child.parent == parent
    
    def test_to_dict(self):
        """Test converting node to dictionary."""
        parent = DocNode("parent", "/path", "package")
        parent.docstring = "Parent docstring"
        parent.signature = "parent()"
        parent.attributes = {"attr": "value"}
        
        child = DocNode("child", "/path/child", "module")
        child.docstring = "Child docstring"
        parent.add_child(child)
        
        result = parent.to_dict()
        
        assert result["name"] == "parent"
        assert result["type"] == "package"
        assert result["path"] == "/path"
        assert result["docstring"] == "Parent docstring"
        assert result["signature"] == "parent()"
        assert result["attributes"] == {"attr": "value"}
        assert len(result["children"]) == 1
        assert result["children"][0]["name"] == "child"

class TestCodeStructureAnalyzer:
    """Test the CodeStructureAnalyzer class."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as tempdir:
            # Create a simple project structure
            os.makedirs(os.path.join(tempdir, "package"))
            with open(os.path.join(tempdir, "package", "__init__.py"), "w") as f:
                f.write("# Package init")
                
            with open(os.path.join(tempdir, "package", "module.py"), "w") as f:
                f.write("""\"\"\"Module docstring.\"\"\"

class TestClass:
    \"\"\"Test class docstring.\"\"\"
    
    def __init__(self, arg1, arg2=None):
        \"\"\"Initialize method.\"\"\"
        self.arg1 = arg1
        self.arg2 = arg2
    
    def test_method(self, param: str) -> bool:
        \"\"\"Test method docstring.\"\"\"
        return True

def test_function(param1, param2=None):
    \"\"\"Test function docstring.\"\"\"
    return param1
""")
            
            yield tempdir
    
    def test_analyze(self, temp_project):
        """Test analyzing a project structure."""
        analyzer = CodeStructureAnalyzer(temp_project)
        result = analyzer.analyze()
        
        # Check root node
        assert result.name == "root"
        assert result.type == "package"
        assert len(result.children) == 1
        
        # Check package node
        package_node = result.children[0]
        assert package_node.name == "package"
        assert package_node.type == "package"
        assert len(package_node.children) == 1
        
        # Check module node
        module_node = package_node.children[0]
        assert module_node.name == "module"
        assert module_node.type == "module"
        assert module_node.docstring == "Module docstring."
        assert len(module_node.children) == 2
        
        # Check class and function
        class_node = next((n for n in module_node.children if n.type == "class"), None)
        func_node = next((n for n in module_node.children if n.type == "function"), None)
        
        assert class_node is not None
        assert class_node.name == "TestClass"
        assert class_node.docstring == "Test class docstring."
        
        assert func_node is not None
        assert func_node.name == "test_function"
        assert func_node.docstring == "Test function docstring."
        
        # Check methods
        method_node = next((n for n in class_node.children if n.type == "method"), None)
        assert method_node is not None
        assert method_node.name == "test_method"
        assert method_node.docstring == "Test method docstring."
        assert method_node.signature == "def test_method(self, param: str) -> bool"

class TestDocumentationGenerator:
    """Test the DocumentationGenerator class."""
    
    @pytest.fixture
    def mock_ai_provider(self):
        """Create a mock AI provider."""
        mock_provider = MagicMock()
        mock_provider.generate_text.return_value = {
            "text": "```mermaid\ngraph TD\n    A[Start] --> B[Process]\n    B --> C[End]\n```"
        }
        return mock_provider
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project structure for testing."""
        with tempfile.TemporaryDirectory() as tempdir:
            # Create a simple project structure
            os.makedirs(os.path.join(tempdir, "sample_pkg"))
            with open(os.path.join(tempdir, "sample_pkg", "__init__.py"), "w") as f:
                f.write("# Package init")
                
            with open(os.path.join(tempdir, "sample_pkg", "module.py"), "w") as f:
                f.write("""\"\"\"
                Sample module for documentation testing.
                \"\"\"
                
                import os
                import sys
                
                class SampleClass:
                    \"\"\"Sample class for testing.\"\"\"
                    
                    def __init__(self, name):
                        \"\"\"Initialize with name.\"\"\"
                        self.name = name
                    
                    def get_name(self) -> str:
                        \"\"\"Return the name.\"\"\"
                        return self.name
                
                def sample_function(param1: str, param2: int = 0) -> bool:
                    \"\"\"Sample function that returns a boolean.\"\"\"
                    return bool(param1) and param2 >= 0
                """)
            
            yield tempdir
    
    def test_init(self, temp_project, mock_ai_provider):
        """Test initialization of DocumentationGenerator."""
        output_dir = "custom_docs"
        generator = DocumentationGenerator(
            project_root=temp_project,
            output_dir=output_dir,
            ai_provider=mock_ai_provider
        )
        
        assert generator.project_root == temp_project
        assert generator.output_dir == os.path.join(temp_project, output_dir)
        assert generator.ai_provider == mock_ai_provider
        assert generator.analyzer is not None
        assert generator.doc_tree is None
        assert generator.auto_generated_docs == {}
        
        # Check that output directory was created
        assert os.path.exists(generator.output_dir)
    
    @patch("app.utils.documentation_generator.markdown.markdown")
    def test_generate(self, mock_markdown, temp_project, mock_ai_provider):
        """Test documentation generation."""
        mock_markdown.return_value = "<p>Markdown content</p>"
        
        generator = DocumentationGenerator(
            project_root=temp_project,
            output_dir="docs",
            ai_provider=mock_ai_provider
        )
        
        # Generate documentation
        result = generator.generate(
            enhance_docstrings=True,
            include_tests=True,
            diagram_types=["module"]
        )
        
        # Check that main files were created
        assert os.path.exists(os.path.join(generator.output_dir, "index.md"))
        assert os.path.exists(os.path.join(generator.output_dir, "index.html"))
        assert os.path.exists(os.path.join(generator.output_dir, "api_reference.md"))
        assert os.path.exists(os.path.join(generator.output_dir, "diagrams", "module_dependencies.md"))
        
        # Check statistics in result
        assert "file_count" in result
        assert "diagram_count" in result
        assert "timestamp" in result
        assert "elapsed_time" in result
        
        # Check that generation_summary.json was created
        summary_path = os.path.join(generator.output_dir, "generation_summary.json")
        assert os.path.exists(summary_path)
        
        with open(summary_path, "r") as f:
            summary = json.load(f)
            assert summary["file_count"] > 0
            assert summary["diagram_count"] > 0
    
    def test_find_all_nodes_by_type(self, temp_project, mock_ai_provider):
        """Test finding nodes by type."""
        generator = DocumentationGenerator(
            project_root=temp_project,
            output_dir="docs",
            ai_provider=mock_ai_provider
        )
        
        # First analyze the project to create the doc_tree
        generator.doc_tree = generator.analyzer.analyze()
        
        # Find modules
        modules = generator._find_all_nodes_by_type("module")
        assert len(modules) > 0
        assert all(node.type == "module" for node in modules)
        
        # Find classes
        classes = generator._find_all_nodes_by_type("class")
        assert len(classes) > 0
        assert all(node.type == "class" for node in classes)
        
        # Find methods
        methods = generator._find_all_nodes_by_type("method")
        assert len(methods) > 0
        assert all(node.type == "method" for node in methods)
    
    def test_generate_project_tree(self, temp_project, mock_ai_provider):
        """Test generating project tree."""
        generator = DocumentationGenerator(
            project_root=temp_project,
            output_dir="docs",
            ai_provider=mock_ai_provider
        )
        
        tree = generator._generate_project_tree(max_depth=3)
        
        assert isinstance(tree, str)
        assert "sample_pkg" in tree
        assert "__init__.py" in tree
        assert "module.py" in tree
    
    @patch("app.utils.documentation_generator.os.makedirs")
    def test_create_component_pages(self, mock_makedirs, temp_project, mock_ai_provider):
        """Test creating component pages."""
        generator = DocumentationGenerator(
            project_root=temp_project,
            output_dir="docs",
            ai_provider=mock_ai_provider
        )
        
        # Create a sample doc tree
        root = DocNode("root", temp_project, "package")
        package = DocNode("sample_pkg", os.path.join(temp_project, "sample_pkg"), "package")
        module = DocNode("module", os.path.join(temp_project, "sample_pkg", "module.py"), "module")
        module.docstring = "Sample module"
        module.code = "# Sample code"
        
        func = DocNode("sample_function", module.path, "function")
        func.docstring = "Sample function"
        func.signature = "def sample_function(param): ..."
        
        cls = DocNode("SampleClass", module.path, "class")
        cls.docstring = "Sample class"
        
        method = DocNode("method", cls.path, "method")
        method.docstring = "Sample method"
        method.signature = "def method(self): ..."
        
        # Build the tree
        root.add_child(package)
        package.add_child(module)
        module.add_child(func)
        module.add_child(cls)
        cls.add_child(method)
        
        generator.doc_tree = root
        
        # Mock file operations
        open_mock = MagicMock()
        with patch("builtins.open", open_mock):
            file_count = generator._create_component_pages(root)
        
        assert file_count > 0
        assert mock_makedirs.called
    
    @patch("app.utils.documentation_generator.os.path.exists")
    @patch("app.utils.documentation_generator.open", new_callable=MagicMock)
    def test_generate_workflow_with_ai(self, mock_open, mock_exists, mock_ai_provider):
        """Test generating workflow diagram with AI."""
        mock_exists.return_value = True
        
        generator = DocumentationGenerator(
            project_root="/fake/path",
            output_dir="docs",
            ai_provider=mock_ai_provider
        )
        
        node = DocNode("workflow", "/path/to/workflow.py", "module")
        node.code = "def process(): pass"
        node.docstring = "A workflow module"
        
        result = generator._generate_workflow_with_ai(node)
        
        assert result is not None
        assert "```mermaid" in result
        assert mock_ai_provider.generate_text.called
        
        # Check that the prompt includes the code
        prompt = mock_ai_provider.generate_text.call_args[1]["prompt"]
        assert "def process(): pass" in prompt
    
    def test_wrap_html(self, mock_ai_provider):
        """Test HTML wrapping function."""
        generator = DocumentationGenerator(
            project_root="/fake/path",
            output_dir="docs",
            ai_provider=mock_ai_provider
        )
        
        html = generator._wrap_html("<p>Test content</p>", "Test Title")
        
        assert "<!DOCTYPE html>" in html
        assert "<title>Test Title</title>" in html
        assert "<p>Test content</p>" in html
        assert "font-family" in html  # Check for CSS styles

if __name__ == "__main__":
    pytest.main(["-v", __file__])