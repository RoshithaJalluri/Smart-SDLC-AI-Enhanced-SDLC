import ast
import re
import json
import difflib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import traceback
import subprocess
import sys
from io import StringIO
import contextlib

class BugSeverity(Enum):
    """Bug severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class BugCategory(Enum):
    """Bug categories for classification"""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    INDENTATION_ERROR = "indentation_error"
    NAME_ERROR = "name_error"
    ATTRIBUTE_ERROR = "attribute_error"
    INDEX_ERROR = "index_error"
    KEY_ERROR = "key_error"
    VALUE_ERROR = "value_error"
    PERFORMANCE_ISSUE = "performance_issue"
    STYLE_ISSUE = "style_issue"
    SECURITY_ISSUE = "security_issue"

@dataclass
class BugReport:
    """Data class for bug analysis results"""
    line_number: int
    column: int
    error_type: str
    error_message: str
    severity: BugSeverity
    category: BugCategory
    code_snippet: str
    suggested_fix: str
    explanation: str
    confidence: float
    
@dataclass
class FixResult:
    """Data class for bug fix results"""
    original_code: str
    fixed_code: str
    bugs_found: List[BugReport]
    fixes_applied: List[Dict]
    diff: str
    success: bool
    error_message: Optional[str] = None
    
class StaticAnalyzer:
    """Static code analysis for bug detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_syntax(self, code: str) -> List[BugReport]:
        """Analyze code for syntax errors"""
        bugs = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            bug = BugReport(
                line_number=e.lineno or 0,
                column=e.offset or 0,
                error_type="SyntaxError",
                error_message=str(e),
                severity=BugSeverity.CRITICAL,
                category=BugCategory.SYNTAX_ERROR,
                code_snippet=self._get_code_snippet(code, e.lineno or 0),
                suggested_fix=self._suggest_syntax_fix(code, e),
                explanation=self._explain_syntax_error(e),
                confidence=0.9
            )
            bugs.append(bug)
            
        return bugs
    
    def analyze_imports(self, code: str) -> List[BugReport]:
        """Analyze import statements for issues"""
        bugs = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                try:
                    # Try to compile the import statement
                    compile(line, '<string>', 'exec')
                except SyntaxError as e:
                    bug = BugReport(
                        line_number=i + 1,
                        column=0,
                        error_type="ImportError",
                        error_message=f"Invalid import syntax: {str(e)}",
                        severity=BugSeverity.HIGH,
                        category=BugCategory.IMPORT_ERROR,
                        code_snippet=line,
                        suggested_fix=self._suggest_import_fix(line),
                        explanation="Import statement has incorrect syntax",
                        confidence=0.8
                    )
                    bugs.append(bug)
                    
        return bugs
    
    def analyze_indentation(self, code: str) -> List[BugReport]:
        """Analyze indentation issues"""
        bugs = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
                
            # Check for mixed tabs and spaces
            if '\t' in line and ' ' in line[:len(line) - len(line.lstrip())]:
                bug = BugReport(
                    line_number=i + 1,
                    column=0,
                    error_type="IndentationError",
                    error_message="Mixed tabs and spaces in indentation",
                    severity=BugSeverity.MEDIUM,
                    category=BugCategory.INDENTATION_ERROR,
                    code_snippet=line,
                    suggested_fix=line.expandtabs(4),
                    explanation="Mixed tabs and spaces can cause IndentationError",
                    confidence=0.95
                )
                bugs.append(bug)
                
        return bugs
    
    def analyze_common_patterns(self, code: str) -> List[BugReport]:
        """Analyze for common bug patterns"""
        bugs = []
        lines = code.split('\n')
        
        patterns = [
            (r'=\s*=(?!=)', "Assignment instead of comparison", "Use '==' for comparison"),
            (r'if\s+.*=(?!=)', "Assignment in if condition", "Use '==' for comparison in conditions"),
            (r'for\s+\w+\s+in\s+range\(\s*len\(.*\)\s*\):', "Inefficient range(len()) loop", "Iterate directly over the collection"),
            (r'except\s*:', "Bare except clause", "Specify exception types to catch"),
            (r'==\s*True', "Comparison with True", "Use 'if condition:' instead of 'if condition == True:'"),
            (r'==\s*False', "Comparison with False", "Use 'if not condition:' instead of 'if condition == False:'"),
        ]
        
        for i, line in enumerate(lines):
            for pattern, message, fix_suggestion in patterns:
                if re.search(pattern, line):
                    bug = BugReport(
                        line_number=i + 1,
                        column=0,
                        error_type="StyleWarning",
                        error_message=message,
                        severity=BugSeverity.LOW,
                        category=BugCategory.STYLE_ISSUE,
                        code_snippet=line.strip(),
                        suggested_fix=fix_suggestion,
                        explanation=f"Pattern detected: {message}",
                        confidence=0.7
                    )
                    bugs.append(bug)
                    
        return bugs
    
    def _get_code_snippet(self, code: str, line_num: int, context: int = 2) -> str:
        """Get code snippet around the error line"""
        lines = code.split('\n')
        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)
        
        snippet_lines = []
        for i in range(start, end):
            marker = ">>> " if i == line_num - 1 else "    "
            snippet_lines.append(f"{marker}{i+1:3d}: {lines[i]}")
            
        return '\n'.join(snippet_lines)
    
    def _suggest_syntax_fix(self, code: str, error: SyntaxError) -> str:
        """Suggest fixes for syntax errors"""
        error_msg = str(error).lower()
        
        if "invalid syntax" in error_msg:
            if "(" in error_msg or ")" in error_msg:
                return "Check for missing or extra parentheses"
            elif ":" in error_msg:
                return "Check for missing colon (:) at end of if/for/while/def statements"
            elif "indent" in error_msg:
                return "Check indentation - Python uses consistent spacing"
                
        return "Review syntax around the highlighted line"
    
    def _suggest_import_fix(self, line: str) -> str:
        """Suggest fixes for import errors"""
        if line.startswith('from') and 'import' not in line:
            return f"{line} import <module_name>"
        elif line.count('import') > 1:
            return "Split multiple imports into separate lines"
        return "Check import syntax: 'import module' or 'from module import item'"
    
    def _explain_syntax_error(self, error: SyntaxError) -> str:
        """Generate explanation for syntax errors"""
        error_msg = str(error).lower()
        
        explanations = {
            "invalid syntax": "The Python interpreter couldn't understand this line. Common causes include missing colons, unmatched parentheses, or incorrect operators.",
            "unexpected eof": "The file ended unexpectedly. Check for unclosed parentheses, brackets, or quotes.",
            "unindent": "Indentation doesn't match any outer indentation level. Check your spacing/tabs.",
            "expected": "Python expected a specific character or keyword that wasn't found."
        }
        
        for key, explanation in explanations.items():
            if key in error_msg:
                return explanation
                
        return "Syntax error detected. Review the code structure and Python syntax rules."

class RuntimeAnalyzer:
    """Runtime code analysis and testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def test_execution(self, code: str, test_inputs: List[Any] = None) -> List[BugReport]:
        """Test code execution to find runtime errors"""
        bugs = []
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            # Compile the code
            compiled_code = compile(code, '<string>', 'exec')
            
            # Execute in a safe namespace
            namespace = {
                '__builtins__': __builtins__,
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
            }
            
            exec(compiled_code, namespace)
            
        except Exception as e:
            # Extract error information
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                filename, line_num, func_name, text = tb[-1]
            else:
                line_num, text = 0, ""
            
            bug = BugReport(
                line_number=line_num,
                column=0,
                error_type=type(e).__name__,
                error_message=str(e),
                severity=self._classify_runtime_severity(e),
                category=self._classify_runtime_category(e),
                code_snippet=text or "",
                suggested_fix=self._suggest_runtime_fix(e),
                explanation=self._explain_runtime_error(e),
                confidence=0.8
            )
            bugs.append(bug)
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
        return bugs
    
    def _classify_runtime_severity(self, error: Exception) -> BugSeverity:
        """Classify runtime error severity"""
        critical_errors = (SyntaxError, IndentationError, SystemExit)
        high_errors = (NameError, AttributeError, TypeError, ImportError)
        medium_errors = (ValueError, KeyError, IndexError)
        
        if isinstance(error, critical_errors):
            return BugSeverity.CRITICAL
        elif isinstance(error, high_errors):
            return BugSeverity.HIGH
        elif isinstance(error, medium_errors):
            return BugSeverity.MEDIUM
        else:
            return BugSeverity.LOW
    
    def _classify_runtime_category(self, error: Exception) -> BugCategory:
        """Classify runtime error category"""
        category_map = {
            NameError: BugCategory.NAME_ERROR,
            AttributeError: BugCategory.ATTRIBUTE_ERROR,
            TypeError: BugCategory.TYPE_ERROR,
            ValueError: BugCategory.VALUE_ERROR,
            KeyError: BugCategory.KEY_ERROR,
            IndexError: BugCategory.INDEX_ERROR,
            ImportError: BugCategory.IMPORT_ERROR,
            IndentationError: BugCategory.INDENTATION_ERROR,
        }
        
        return category_map.get(type(error), BugCategory.RUNTIME_ERROR)
    
    def _suggest_runtime_fix(self, error: Exception) -> str:
        """Suggest fixes for runtime errors"""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        suggestions = {
            'NameError': "Variable is not defined. Check variable names and ensure they're assigned before use.",
            'AttributeError': "Object doesn't have the specified attribute. Check object type and available methods.",
            'TypeError': "Wrong data type for operation. Check variable types and function arguments.",
            'ValueError': "Correct type but inappropriate value. Validate input values.",
            'KeyError': "Dictionary key doesn't exist. Use .get() method or check key existence.",
            'IndexError': "List index out of range. Check list length and index bounds.",
            'ImportError': "Module not found. Check module name and installation.",
            'IndentationError': "Incorrect indentation. Use consistent spaces (4 spaces recommended).",
        }
        
        return suggestions.get(error_type, "Review the error message and check the problematic line.")
    
    def _explain_runtime_error(self, error: Exception) -> str:
        """Generate detailed explanation for runtime errors"""
        error_type = type(error).__name__
        
        explanations = {
            'NameError': "This error occurs when Python tries to use a variable that hasn't been defined yet.",
            'AttributeError': "This happens when you try to access a method or property that doesn't exist on an object.",
            'TypeError': "This error occurs when an operation is performed on an inappropriate type.",
            'ValueError': "This happens when a function receives an argument of correct type but inappropriate value.",
            'KeyError': "This occurs when trying to access a dictionary key that doesn't exist.",
            'IndexError': "This happens when trying to access a list index that's out of range.",
            'ImportError': "This occurs when Python can't find or import a specified module.",
        }
        
        return explanations.get(error_type, f"{error_type} occurred during code execution.")

class BugResolver:
    """Main bug resolver class that combines analysis and fixing"""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = self._setup_logger(log_level)
        self.static_analyzer = StaticAnalyzer()
        self.runtime_analyzer = RuntimeAnalyzer()
        
    def _setup_logger(self, level: str) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def analyze_code(self, code: str, include_runtime: bool = True) -> List[BugReport]:
        """Comprehensive code analysis"""
        self.logger.info("Starting code analysis...")
        
        all_bugs = []
        
        # Static analysis
        all_bugs.extend(self.static_analyzer.analyze_syntax(code))
        all_bugs.extend(self.static_analyzer.analyze_imports(code))
        all_bugs.extend(self.static_analyzer.analyze_indentation(code))
        all_bugs.extend(self.static_analyzer.analyze_common_patterns(code))
        
        # Runtime analysis (only if no syntax errors)
        if include_runtime and not any(bug.category == BugCategory.SYNTAX_ERROR for bug in all_bugs):
            all_bugs.extend(self.runtime_analyzer.test_execution(code))
        
        # Sort by severity and line number
        all_bugs.sort(key=lambda x: (x.severity.value, x.line_number))
        
        self.logger.info(f"Analysis complete: {len(all_bugs)} issues found")
        return all_bugs
    
    def generate_fixes(self, code: str, bugs: List[BugReport]) -> str:
        """Generate fixed version of the code"""
        fixed_code = code
        lines = fixed_code.split('\n')
        
        # Apply fixes in reverse line order to preserve line numbers
        bugs_by_line = sorted(bugs, key=lambda x: x.line_number, reverse=True)
        
        for bug in bugs_by_line:
            if bug.line_number > 0 and bug.line_number <= len(lines):
                line_idx = bug.line_number - 1
                original_line = lines[line_idx]
                
                fixed_line = self._apply_fix(original_line, bug)
                if fixed_line != original_line:
                    lines[line_idx] = fixed_line
                    self.logger.info(f"Applied fix at line {bug.line_number}")
        
        return '\n'.join(lines)
    
    def _apply_fix(self, line: str, bug: BugReport) -> str:
        """Apply specific fix to a line of code"""
        if bug.category == BugCategory.INDENTATION_ERROR:
            return line.expandtabs(4)
        elif bug.category == BugCategory.STYLE_ISSUE:
            return self._apply_style_fix(line, bug)
        
        # For other errors, return original line (would need AI model for complex fixes)
        return line
    
    def _apply_style_fix(self, line: str, bug: BugReport) -> str:
        """Apply style-related fixes"""
        # Fix assignment vs comparison
        if "Assignment instead of comparison" in bug.error_message:
            line = re.sub(r'=(?!=)', '==', line)
        
        # Fix comparison with True/False
        elif "Comparison with True" in bug.error_message:
            line = re.sub(r'==\s*True', '', line)
        elif "Comparison with False" in bug.error_message:
            line = re.sub(r'==\s*False', '', line)
            if 'if' in line:
                line = line.replace('if ', 'if not ')
        
        return line
    
    def generate_diff(self, original_code: str, fixed_code: str) -> str:
        """Generate diff between original and fixed code"""
        original_lines = original_code.splitlines(keepends=True)
        fixed_lines = fixed_code.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            fixed_lines,
            fromfile='original.py',
            tofile='fixed.py',
            lineterm=''
        )
        
        return ''.join(diff)
    
    def resolve_bugs(self, code: str, auto_fix: bool = True) -> FixResult:
        """Main method to analyze and fix bugs"""
        self.logger.info("Starting bug resolution...")
        
        try:
            # Analyze code for bugs
            bugs = self.analyze_code(code)
            
            if not bugs:
                self.logger.info("No bugs found!")
                return FixResult(
                    original_code=code,
                    fixed_code=code,
                    bugs_found=[],
                    fixes_applied=[],
                    diff="",
                    success=True
                )
            
            # Generate fixes
            fixed_code = code
            fixes_applied = []
            
            if auto_fix:
                fixed_code = self.generate_fixes(code, bugs)
                
                # Track what fixes were applied
                for bug in bugs:
                    if bug.category in [BugCategory.INDENTATION_ERROR, BugCategory.STYLE_ISSUE]:
                        fixes_applied.append({
                            'line': bug.line_number,
                            'type': bug.category.value,
                            'description': bug.suggested_fix
                        })
            
            # Generate diff
            diff = self.generate_diff(code, fixed_code)
            
            self.logger.info(f"Bug resolution complete: {len(fixes_applied)} fixes applied")
            
            return FixResult(
                original_code=code,
                fixed_code=fixed_code,
                bugs_found=bugs,
                fixes_applied=fixes_applied,
                diff=diff,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Bug resolution failed: {str(e)}")
            return FixResult(
                original_code=code,
                fixed_code=code,
                bugs_found=[],
                fixes_applied=[],
                diff="",
                success=False,
                error_message=str(e)
            )
    
    def generate_report(self, result: FixResult) -> Dict:
        """Generate comprehensive bug report"""
        report = {
            'summary': {
                'total_bugs': len(result.bugs_found),
                'fixes_applied': len(result.fixes_applied),
                'success': result.success
            },
            'bugs_by_severity': {},
            'bugs_by_category': {},
            'detailed_bugs': [],
            'applied_fixes': result.fixes_applied,
            'diff': result.diff
        }
        
        # Group bugs by severity and category
        for bug in result.bugs_found:
            severity = bug.severity.value
            category = bug.category.value
            
            report['bugs_by_severity'][severity] = report['bugs_by_severity'].get(severity, 0) + 1
            report['bugs_by_category'][category] = report['bugs_by_category'].get(category, 0) + 1
            
            report['detailed_bugs'].append({
                'line': bug.line_number,
                'column': bug.column,
                'type': bug.error_type,
                'message': bug.error_message,
                'severity': bug.severity.value,
                'category': bug.category.value,
                'snippet': bug.code_snippet,
                'fix_suggestion': bug.suggested_fix,
                'explanation': bug.explanation,
                'confidence': bug.confidence
            })
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Example buggy code for testing
    buggy_code = '''
import os
import sys

def calculate_average(numbers):
    if len(numbers) = 0:  # Bug: assignment instead of comparison
        return 0
    total = 0
    for i in range(len(numbers)):  # Bug: inefficient loop
        total += numbers[i]
    return total / len(numbers)

def process_data(data_list):
	# Bug: mixed tabs and spaces
    results = []
    for item in data_list:
        if item == True:  # Bug: comparison with True
            results.append(item * 2)
        elif item == False:  # Bug: comparison with False
            results.append(0)
    return results

# Test the functions
numbers = [1, 2, 3, 4, 5]
average = calculate_average(numbers)
print(f"Average: {average}")

data = [True, False, True, False]
processed = process_data(data)
print(f"Processed: {processed}")
'''
    
    # Initialize bug resolver
    resolver = BugResolver()
    
    # Analyze and fix bugs
    result = resolver.resolve_bugs(buggy_code, auto_fix=True)
    
    # Generate report
    report = resolver.generate_report(result)
    
    print("=== BUG RESOLUTION REPORT ===")
    print(f"Total bugs found: {report['summary']['total_bugs']}")
    print(f"Fixes applied: {report['summary']['fixes_applied']}")
    print(f"Success: {report['summary']['success']}")
    
    print("\n=== BUGS BY SEVERITY ===")
    for severity, count in report['bugs_by_severity'].items():
        print(f"{severity.upper()}: {count}")
    
    print("\n=== DETAILED BUGS ===")
    for bug in report['detailed_bugs']:
        print(f"Line {bug['line']}: {bug['type']} - {bug['message']}")
        print(f"  Fix: {bug['fix_suggestion']}")
        print(f"  Confidence: {bug['confidence']:.1%}")
        print()
    
    if result.diff:
        print("=== CODE DIFF ===")
        print(result.diff)
        
    print("\n=== FIXED CODE ===")
    print(result.fixed_code)