import os
import re
import json
import ast
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Core libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import subprocess
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgrammingLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    TYPESCRIPT = "typescript"
    PHP = "php"
    RUBY = "ruby"

class CodeComplexity(Enum):
    """Code complexity levels"""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"

@dataclass
class CodeGenerationRequest:
    """Request structure for code generation"""
    task_description: str
    language: ProgrammingLanguage
    complexity: CodeComplexity
    framework: Optional[str] = None
    additional_requirements: List[str] = None
    include_tests: bool = True
    include_documentation: bool = True
    code_style: str = "clean_code"

@dataclass
class GeneratedCode:
    """Structure for generated code output"""
    id: str
    request: CodeGenerationRequest
    main_code: str
    test_code: str
    documentation: str
    dependencies: List[str]
    execution_instructions: str
    code_quality_score: float
    generation_timestamp: str
    estimated_complexity: str
    file_structure: Dict[str, str]

@dataclass
class TestCase:
    """Structure for individual test cases"""
    name: str
    description: str
    test_type: str  # unit, integration, functional
    test_code: str
    expected_outcome: str
    test_data: Dict[str, Any]
    priority: str

class GraniteCodeEngine:
    """Enhanced AI engine for code generation using IBM Granite models"""
    
    def __init__(self, 
                 base_model: str = "ibm-granite/granite-3.3-2b-instruct",
                 code_model: str = "ibm-granite/granite-20b-code-instruct"):
        self.base_model_name = base_model
        self.code_model_name = code_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model instances
        self.base_tokenizer = None
        self.base_model = None
        self.code_tokenizer = None  
        self.code_model = None
        
        self.load_models()
    
    def load_models(self):
        """Load both base and code generation models"""
        try:
            logger.info("Loading Granite models...")
            
            # Load base model for general tasks
            self.base_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Set pad tokens
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
            # Try to load code model (fallback to base model if not available)
            try:
                self.code_tokenizer = AutoTokenizer.from_pretrained(
                    self.code_model_name,
                    trust_remote_code=True
                )
                self.code_model = AutoModelForCausalLM.from_pretrained(
                    self.code_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.code_tokenizer.pad_token is None:
                    self.code_tokenizer.pad_token = self.code_tokenizer.eos_token
                logger.info("Code-specific model loaded successfully")
            except Exception as e:
                logger.warning(f"Code model not available, using base model: {str(e)}")
                self.code_tokenizer = self.base_tokenizer
                self.code_model = self.base_model
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, use_code_model: bool = False, 
                         max_length: int = 1024, temperature: float = 0.7) -> str:
        """Generate response using appropriate model"""
        try:
            # Select model and tokenizer
            tokenizer = self.code_tokenizer if use_code_model else self.base_tokenizer
            model = self.code_model if use_code_model else self.base_model
            
            inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    no_repeat_ngram_size=3
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from response
            response = response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return ""

class CodeTemplateManager:
    """Manages code templates and boilerplate for different languages"""
    
    @staticmethod
    def get_language_config(language: ProgrammingLanguage) -> Dict[str, str]:
        """Get language-specific configuration"""
        configs = {
            ProgrammingLanguage.PYTHON: {
                "extension": ".py",
                "comment": "#",
                "test_framework": "pytest",
                "package_manager": "pip",
                "entry_point": "if __name__ == '__main__':"
            },
            ProgrammingLanguage.JAVASCRIPT: {
                "extension": ".js",
                "comment": "//",
                "test_framework": "jest",
                "package_manager": "npm",
                "entry_point": "// Entry point"
            },
            ProgrammingLanguage.JAVA: {
                "extension": ".java",
                "comment": "//",
                "test_framework": "junit",
                "package_manager": "maven",
                "entry_point": "public static void main(String[] args)"
            },
            ProgrammingLanguage.TYPESCRIPT: {
                "extension": ".ts",
                "comment": "//",
                "test_framework": "jest",
                "package_manager": "npm",
                "entry_point": "// Entry point"
            }
        }
        return configs.get(language, configs[ProgrammingLanguage.PYTHON])
    
    @staticmethod
    def get_test_template(language: ProgrammingLanguage) -> str:
        """Get test template for specific language"""
        templates = {
            ProgrammingLanguage.PYTHON: """
import pytest
import unittest
from unittest.mock import Mock, patch
# Import the module to test
# from your_module import YourClass

class Test{ClassName}(unittest.TestCase):
    def setUp(self):
        # Setup test fixtures
        pass
    
    def test_{function_name}_basic(self):
        # Basic functionality test
        pass
    
    def test_{function_name}_edge_cases(self):
        # Edge cases test
        pass
    
    def test_{function_name}_error_handling(self):
        # Error handling test
        pass

if __name__ == '__main__':
    unittest.main()
""",
            
            ProgrammingLanguage.JAVASCRIPT: """
const {{ {ClassName} }} = require('./{module_name}');

describe('{ClassName}', () => {{
    beforeEach(() => {{
        // Setup test fixtures
    }});
    
    test('should handle basic functionality', () => {{
        // Basic functionality test
    }});
    
    test('should handle edge cases', () => {{
        // Edge cases test
    }});
    
    test('should handle errors gracefully', () => {{
        // Error handling test
    }});
}});
""",
            
            ProgrammingLanguage.JAVA: """
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

public class {ClassName}Test {{
    private {ClassName} instance;
    
    @BeforeEach
    void setUp() {{
        // Setup test fixtures
        instance = new {ClassName}();
    }}
    
    @Test
    void test{FunctionName}Basic() {{
        // Basic functionality test
    }}
    
    @Test
    void test{FunctionName}EdgeCases() {{
        // Edge cases test
    }}
    
    @Test
    void test{FunctionName}ErrorHandling() {{
        // Error handling test
    }}
}}
"""
        }
        return templates.get(language, templates[ProgrammingLanguage.PYTHON])

class CodeGenerator:
    """Main code generation engine"""
    
    def __init__(self, ai_engine: GraniteCodeEngine):
        self.ai_engine = ai_engine
        self.template_manager = CodeTemplateManager()
    
    def create_code_generation_prompt(self, request: CodeGenerationRequest) -> str:
        """Create optimized prompt for code generation"""
        language_config = self.template_manager.get_language_config(request.language)
        
        complexity_guidelines = {
            CodeComplexity.SIMPLE: "Write simple, beginner-friendly code with clear comments",
            CodeComplexity.INTERMEDIATE: "Write well-structured code following best practices",
            CodeComplexity.ADVANCED: "Write sophisticated code with advanced patterns and optimizations",
            CodeComplexity.ENTERPRISE: "Write enterprise-grade code with full error handling, logging, and scalability"
        }
        
        framework_section = f"\nFramework: {request.framework}" if request.framework else ""
        requirements_section = f"\nAdditional Requirements:\n" + "\n".join([f"- {req}" for req in request.additional_requirements]) if request.additional_requirements else ""
        
        prompt = f"""You are an expert {request.language.value} developer. Generate production-ready code based on the following requirements:

Task Description: {request.task_description}
Programming Language: {request.language.value}
Complexity Level: {request.complexity.value}
Code Style: {request.code_style}{framework_section}{requirements_section}

Guidelines:
- {complexity_guidelines[request.complexity]}
- Follow {request.language.value} best practices and conventions
- Include proper error handling and input validation
- Add comprehensive docstrings/comments
- Make code modular and reusable
- Include type hints where applicable
- Ensure code is secure and follows SOLID principles

Generate the complete, functional code:

```{request.language.value}"""
        
        return prompt
    
    def generate_code(self, request: CodeGenerationRequest) -> str:
        """Generate main code based on request"""
        try:
            prompt = self.create_code_generation_prompt(request)
            response = self.ai_engine.generate_response(
                prompt, 
                use_code_model=True, 
                max_length=2048, 
                temperature=0.6
            )
            
            # Extract code from response
            code = self.extract_code_from_response(response, request.language)
            return code
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return f"# Error generating code: {str(e)}"
    
    def extract_code_from_response(self, response: str, language: ProgrammingLanguage) -> str:
        """Extract clean code from AI response"""
        try:
            # Look for code blocks
            code_pattern = rf"```{language.value}(.*?)```"
            match = re.search(code_pattern, response, re.DOTALL | re.IGNORECASE)
            
            if match:
                return match.group(1).strip()
            
            # If no code blocks found, try to extract code after the prompt
            lines = response.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if any(keyword in line.lower() for keyword in ['def ', 'class ', 'function ', 'import ', 'const ', 'var ']):
                    in_code = True
                
                if in_code:
                    code_lines.append(line)
            
            return '\n'.join(code_lines).strip() if code_lines else response.strip()
            
        except Exception as e:
            logger.warning(f"Error extracting code: {str(e)}")
            return response.strip()

class TestCaseGenerator:
    """Generates comprehensive test cases for generated code"""
    
    def __init__(self, ai_engine: GraniteCodeEngine):
        self.ai_engine = ai_engine
        self.template_manager = CodeTemplateManager()
    
    def create_test_generation_prompt(self, code: str, language: ProgrammingLanguage, 
                                    request: CodeGenerationRequest) -> str:
        """Create prompt for test case generation"""
        language_config = self.template_manager.get_language_config(language)
        test_framework = language_config["test_framework"]
        
        prompt = f"""You are an expert QA engineer. Generate comprehensive test cases for the following {language.value} code:

Original Task: {request.task_description}
Programming Language: {language.value}
Test Framework: {test_framework}

Code to test:
```{language.value}
{code}
```

Generate comprehensive test cases that include:
1. Unit tests for individual functions/methods
2. Integration tests for component interactions
3. Edge case tests (boundary conditions, empty inputs, etc.)
4. Error handling tests (invalid inputs, exceptions)
5. Performance tests where applicable

Requirements:
- Use {test_framework} testing framework
- Include setup and teardown methods
- Add descriptive test names and documentation
- Cover positive and negative test scenarios
- Include mock objects where needed
- Aim for high code coverage

Generate the complete test code:

```{language.value}"""
        
        return prompt
    
    def generate_test_cases(self, code: str, language: ProgrammingLanguage, 
                          request: CodeGenerationRequest) -> str:
        """Generate comprehensive test cases"""
        try:
            prompt = self.create_test_generation_prompt(code, language, request)
            response = self.ai_engine.generate_response(
                prompt, 
                use_code_model=True, 
                max_length=2048, 
                temperature=0.5
            )
            
            # Extract test code from response
            test_code = self.extract_code_from_response(response, language)
            return test_code
            
        except Exception as e:
            logger.error(f"Error generating test cases: {str(e)}")
            return f"# Error generating test cases: {str(e)}"
    
    def extract_code_from_response(self, response: str, language: ProgrammingLanguage) -> str:
        """Extract test code from AI response"""
        try:
            # Look for code blocks
            code_pattern = rf"```{language.value}(.*?)```"
            match = re.search(code_pattern, response, re.DOTALL | re.IGNORECASE)
            
            if match:
                return match.group(1).strip()
            
            return response.strip()
            
        except Exception as e:
            logger.warning(f"Error extracting test code: {str(e)}")
            return response.strip()

class DocumentationGenerator:
    """Generates code documentation and README files"""
    
    def __init__(self, ai_engine: GraniteCodeEngine):
        self.ai_engine = ai_engine
    
    def generate_documentation(self, code: str, language: ProgrammingLanguage, 
                                request: CodeGenerationRequest) -> str:
        """Generate comprehensive documentation"""
        try:
            prompt = f"""Generate comprehensive documentation for the following {language.value} code:

Original Task: {request.task_description}
Programming Language: {language.value}

Code:
```{language.value}
{code}
```

Generate documentation that includes:
1. Overview and purpose
2. Installation and setup instructions
3. Usage examples
4. API documentation (functions, classes, parameters)
5. Dependencies and requirements
6. Configuration options
7. Troubleshooting guide
8. Contributing guidelines

Format as a professional README.md file:"""
            
            response = self.ai_engine.generate_response(
                prompt,
                use_code_model=False,
                max_length=1536,
                temperature=0.6
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating documentation: {str(e)}")
            return f"# Error generating documentation: {str(e)}"

class CodeQualityAnalyzer:
    """Analyzes code quality and provides scoring"""
    
    def __init__(self, ai_engine: GraniteCodeEngine):
        self.ai_engine = ai_engine
    
    def analyze_code_quality(self, code: str, language: ProgrammingLanguage) -> Dict[str, Any]:
        """Analyze code quality and return metrics"""
        try:
            prompt = f"""Analyze the following {language.value} code for quality metrics:

Code:
```{language.value}
{code}
```

Analyze the code based on:
1. Code structure and organization
2. Readability and maintainability
3. Error handling
4. Security considerations
5. Performance implications
6. Best practices adherence
7. Documentation quality
8. Testability

Respond in JSON format:
{{
    "overall_score": score_0_to_100,
    "metrics": {{
        "readability": score_0_to_100,
        "maintainability": score_0_to_100,
        "security": score_0_to_100,
        "performance": score_0_to_100,
        "best_practices": score_0_to_100
    }},
    "strengths": ["strength1", "strength2"],
    "improvements": ["improvement1", "improvement2"],
    "complexity_estimate": "simple|intermediate|advanced|enterprise"
}}

Analysis:"""
            
            response = self.ai_engine.generate_response(
                prompt,
                use_code_model=False,
                max_length=512,
                temperature=0.3
            )
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
            
            # Fallback analysis
            return {
                "overall_score": 75,
                "metrics": {
                    "readability": 75,
                    "maintainability": 75,
                    "security": 70,
                    "performance": 75,
                    "best_practices": 75
                },
                "strengths": ["Generated code structure"],
                "improvements": ["Add more error handling"],
                "complexity_estimate": "intermediate"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing code quality: {str(e)}")
            return {"overall_score": 50, "error": str(e)}

class MultilingualCodeGenerator:
    """Main orchestrator class for multilingual code generation"""
    
    def __init__(self):
        self.ai_engine = GraniteCodeEngine()
        self.code_generator = CodeGenerator(self.ai_engine)
        self.test_generator = TestCaseGenerator(self.ai_engine)
        self.doc_generator = DocumentationGenerator(self.ai_engine)
        self.quality_analyzer = CodeQualityAnalyzer(self.ai_engine)
        self.template_manager = CodeTemplateManager()
    
    def generate_complete_solution(self, request: CodeGenerationRequest) -> GeneratedCode:
        """Generate complete code solution with tests and documentation"""
        try:
            logger.info(f"Generating {request.language.value} code for: {request.task_description}")
            
            # Generate main code
            logger.info("Generating main code...")
            main_code = self.code_generator.generate_code(request)
            
            # Generate test cases
            test_code = ""
            if request.include_tests:
                logger.info("Generating test cases...")
                test_code = self.test_generator.generate_test_cases(main_code, request.language, request)
            
            # Generate documentation
            documentation = ""
            if request.include_documentation:
                logger.info("Generating documentation...")
                documentation = self.doc_generator.generate_documentation(main_code, request.language, request)
            
            # Analyze code quality
            logger.info("Analyzing code quality...")
            quality_analysis = self.quality_analyzer.analyze_code_quality(main_code, request.language)
            
            # Create file structure
            language_config = self.template_manager.get_language_config(request.language)
            file_structure = self._create_file_structure(request, main_code, test_code, documentation, language_config)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(main_code, request.language)
            
            # Create execution instructions
            execution_instructions = self._create_execution_instructions(request, dependencies)
            
            # Create result object
            result = GeneratedCode(
                id=f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                request=request,
                main_code=main_code,
                test_code=test_code,
                documentation=documentation,
                dependencies=dependencies,
                execution_instructions=execution_instructions,
                code_quality_score=quality_analysis.get("overall_score", 0),
                generation_timestamp=datetime.now().isoformat(),
                estimated_complexity=quality_analysis.get("complexity_estimate", "intermediate"),
                file_structure=file_structure
            )
            
            logger.info(f"Code generation complete. Quality score: {result.code_quality_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating complete solution: {str(e)}")
            raise
    
    def _create_file_structure(self, request: CodeGenerationRequest, main_code: str, 
                                test_code: str, documentation: str, language_config: Dict) -> Dict[str, str]:
        """Create appropriate file structure for the generated code"""
        extension = language_config["extension"]
        
        # Extract class/function names for better file naming
        file_name = self._extract_main_identifier(main_code, request.language)
        if not file_name:
            file_name = "main"
        
        structure = {
            f"{file_name}{extension}": main_code,
            "README.md": documentation
        }
        
        if test_code:
            structure[f"test_{file_name}{extension}"] = test_code
        
        return structure
    
    def _extract_main_identifier(self, code: str, language: ProgrammingLanguage) -> str:
        """Extract main class or function name from code"""
        try:
            if language == ProgrammingLanguage.PYTHON:
                # Look for class or main function
                class_match = re.search(r'class\s+(\w+)', code)
                if class_match:
                    return class_match.group(1).lower()
                
                func_match = re.search(r'def\s+(\w+)', code)
                if func_match:
                    return func_match.group(1)
            
            elif language == ProgrammingLanguage.JAVA:
                class_match = re.search(r'public\s+class\s+(\w+)', code)
                if class_match:
                    return class_match.group(1)
            
            elif language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
                # Look for function declarations or class definitions
                class_match = re.search(r'class\s+(\w+)', code)
                if class_match:
                    return class_match.group(1).lower()
                
                func_match = re.search(r'function\s+(\w+)', code)
                if func_match:
                    return func_match.group(1)
            
            return "main"
            
        except Exception:
            return "main"
    
    def _extract_dependencies(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Extract dependencies from generated code"""
        dependencies = []
        
        try:
            if language == ProgrammingLanguage.PYTHON:
                # Find import statements
                import_matches = re.findall(r'import\s+([^\s\n]+)', code)
                from_matches = re.findall(r'from\s+([^\s\n]+)\s+import', code)
                dependencies.extend(import_matches + from_matches)
            
            elif language == ProgrammingLanguage.JAVASCRIPT or language == ProgrammingLanguage.TYPESCRIPT:
                # Find require/import statements
                require_matches = re.findall(r'require\([\'"]([^\'"]+)[\'"]\)', code)
                import_matches = re.findall(r'import.*from\s+[\'"]([^\'"]+)[\'"]', code)
                dependencies.extend(require_matches + import_matches)
            
            elif language == ProgrammingLanguage.JAVA:
                # Find import statements
                import_matches = re.findall(r'import\s+([^\s;]+)', code)
                dependencies.extend(import_matches)
            
            # Filter out standard library imports
            filtered_deps = [dep for dep in dependencies if not dep.startswith(('.', '/'))]
            return list(set(filtered_deps))
            
        except Exception as e:
            logger.warning(f"Error extracting dependencies: {str(e)}")
            return []
    
    def _create_execution_instructions(self, request: CodeGenerationRequest, dependencies: List[str]) -> str:
        """Create execution instructions for the generated code"""
        language_config = self.template_manager.get_language_config(request.language)
        
        instructions = f"""# Execution Instructions for {request.language.value} Code

## Prerequisites
- {request.language.value} runtime environment installed

## Dependencies
"""
        
        if dependencies:
            if request.language == ProgrammingLanguage.PYTHON:
                instructions += "Install dependencies using pip:\n```bash\n"
                for dep in dependencies:
                    instructions += f"pip install {dep}\n"
                instructions += "```\n"
            elif request.language in [ProgrammingLanguage.JAVASCRIPT, ProgrammingLanguage.TYPESCRIPT]:
                instructions += "Install dependencies using npm:\n```bash\n"
                for dep in dependencies:
                    instructions += f"npm install {dep}\n"
                instructions += "```\n"
        else:
            instructions += "No external dependencies required.\n"
        
        instructions += f"""
## Running the Code
1. Save the code to a file with {language_config['extension']} extension
2. Execute using the appropriate command for {request.language.value}

## Running Tests
1. Save test code to a separate test file
2. Run tests using {language_config['test_framework']}

## Notes
- Ensure all dependencies are installed before running
- Check console output for any runtime errors
- Modify configuration as needed for your environment
"""
        
        return instructions
    
    def save_generated_code(self, result: GeneratedCode, output_dir: str):
        """Save all generated files to specified directory"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each file in the structure
            for filename, content in result.file_structure.items():
                file_path = os.path.join(output_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Save execution instructions
            instructions_path = os.path.join(output_dir, "INSTRUCTIONS.md")
            with open(instructions_path, 'w', encoding='utf-8') as f:
                f.write(result.execution_instructions)
            
            # Save metadata
            metadata = {
                "generation_id": result.id,
                "timestamp": result.generation_timestamp,
                "request": asdict(result.request),
                "code_quality_score": result.code_quality_score,
                "estimated_complexity": result.estimated_complexity,
                "dependencies": result.dependencies
            }
            
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Generated code saved to: {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving generated code: {str(e)}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # Initialize the code generator
    generator = MultilingualCodeGenerator()
    
    # Example request
    request = CodeGenerationRequest(
        task_description="Create a REST API for user management with CRUD operations, authentication, and data validation",
        language=ProgrammingLanguage.PYTHON,
        complexity=CodeComplexity.INTERMEDIATE,
        framework="FastAPI",
        additional_requirements=[
            "Use SQLAlchemy for database operations",
            "Implement JWT authentication",
            "Add input validation with Pydantic",
            "Include error handling and logging"
        ],
        include_tests=True,
        include_documentation=True,
        code_style="clean_code"
    )
    
    try:
        # Generate complete solution
        result = generator.generate_complete_solution(request)
        
        # Save to directory
        output_dir = f"generated_code_{result.id}"
        generator.save_generated_code(result, output_dir)
        
        print(f"Code generation complete!")
        print(f"Quality Score: {result.code_quality_score}/100")
        print(f"Estimated Complexity: {result.estimated_complexity}")
        print(f"Files generated: {len(result.file_structure)}")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")