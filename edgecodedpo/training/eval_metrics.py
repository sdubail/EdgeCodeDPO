import ast
import subprocess
from typing import Dict, Any
import torch

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from radon.complexity import cc_visit_ast
from pylint.lint import Run
from pylint.reporters.text import TextReporter

from edgecodedpo.utils.generated_code_parsing import safe_parse_code

def evaluate_code_quality(code: str) -> Dict[str, float|None]:
    """
    Evaluate the quality of the generated code.
    """
    try:
        # Parse the code into an AST
        tree = safe_parse_code(code)
            
        # Calculate type annotation coverage
        type_annotation_coverage = calculate_type_annotation_coverage(tree)
        
        # Calculate docstring coverage
        docstring_coverage = calculate_docstring_coverage(tree)

        # Calculate code complexity
        code_complexity = calculate_code_complexity(tree)

        # Check PEP 8 compliance
        pep8_compliance = check_pep8_compliance(code)

        # Calculate comment density
        comment_density = calculate_comment_density(code)
        
        return {
            "type_annotation_coverage": type_annotation_coverage,
            "docstring_coverage": docstring_coverage,
            "code_complexity": code_complexity,
            "pep8_compliance": pep8_compliance,
            "comment_density": comment_density,
        }
    except Exception as e:
        print(f"Error evaluating code quality: {e}")
        return {
            "type_annotation_coverage": None,
            "docstring_coverage": None,
            "code_complexity": None,
            "pep8_compliance": None,
            "comment_density": None,
        }


def calculate_type_annotation_coverage(tree: ast.AST, metric_level: int=2) -> float|None:
    """
    Calculate the percentage of functions/methods with type annotations.
    
    metric_level:
        0 - At least one annotation (return type OR any argument)
        1 - Return type AND at least one argument annotated
        2 - Return type AND all arguments annotated
    """
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if not functions:
        return None

    match metric_level:
        case 0:
            annotated_functions = [
                fn for fn in functions if fn.returns or any(arg.annotation for arg in fn.args.args if arg.arg not in {"self", "cls"})
            ]
        case 1:
            annotated_functions = [
                fn for fn in functions if fn.returns and any(arg.annotation for arg in fn.args.args if arg.arg not in {"self", "cls"})
            ]
        case 2:
            annotated_functions = [
                fn for fn in functions if fn.returns and all(arg.annotation for arg in fn.args.args if arg.arg not in {"self", "cls"})
            ]
        case _:
            raise NotImplementedError("calculate_type_annotation_coverage has unrecognized metric_level.")
    
    return len(annotated_functions) / len(functions)


def calculate_docstring_coverage(tree: ast.AST) -> float|None:
    """
    Calculate the percentage of functions/methods with docstrings.
    """
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if not functions:
        return None

    documented_functions = [fn for fn in functions if ast.get_docstring(fn)]
    return len(documented_functions) / len(functions)


def calculate_code_complexity(tree: ast.AST) -> float|None:
    """
    Calculate the average cyclomatic complexity of the code.
    """
    try:
        # Calculate cyclomatic complexity for each function/method in the AST
        complexities = [func.complexity for func in cc_visit_ast(tree)]
        return np.mean(complexities) if complexities else 0.0
    except Exception as e:
        print(f"Error calculating code complexity: {e}")
        return None


def check_pep8_compliance(code: str) -> float:
    """
    Check PEP 8 compliance using pylint.

    This function runs pylint on the given code and returns a normalized score 
    between 0.0 and 1.0, where 1.0 means full compliance.

    Criteria checked (PEP 8):
    - Indentation and whitespace usage
    - Naming conventions (variables, functions, classes)
    - Line length (max 79 characters)
    - Import ordering and structure
    - Proper spacing around operators and expressions
    - Docstrings and comments formatting
    - Avoidance of unused variables and imports
    - Function and class structure best practices

    Parameters:
    code (str): The Python code to analyze.

    Returns:
    float: A compliance score between 0.0 (poor) and 1.0 (perfect).
    """
    try:
        # Run pylint and capture the score
        reporter = TextReporter()
        Run([code], reporter=reporter, do_exit=False)
        score = reporter.linter.stats["global_note"]
        return max(0.0, score / 10.0)  # Normalize to [0, 1]
    except Exception:
        return None


def calculate_comment_density(code: str) -> float:
    """
    Calculate the density of comments in the code.

    Parameters:
    code (str): The Python code to analyze.

    Returns:
    float: The ratio of comment lines to total lines in the code.
    """
    lines = code.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return 0.0

    comment_lines = 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("#"):
            comment_lines += 1

    return comment_lines / total_lines


def execute_code(code: str) -> Dict[str, Any]:
    """
    Execute the generated code and check for errors.
    """
    try:
        # Execute the code and capture the output
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, timeout=10
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout.strip(),
            "error": result.stderr.strip(),
        }
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

def calculate_code_similarity(code1: str, code2: str) -> float:
    """
    Calculate the similarity between two code snippets using GPT-2 embeddings with padding.
    """
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained("gpt2")

    # Tokenize & pad to same length
    inputs = tokenizer(
        [code1, code2], 
        return_tensors="pt", 
        padding="max_length", 
        max_length=128, 
        truncation=True
    )

    # Get embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state

    # Take mean over sequence dimension to get fixed-size vector
    embeddings1 = outputs[0].mean(dim=0).numpy()
    embeddings2 = outputs[1].mean(dim=0).numpy()

    similarity = cosine_similarity([embeddings1], [embeddings2])[0][0]
    return similarity

