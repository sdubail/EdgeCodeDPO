import re
import ast

def is_valid_python(code):
    """
    Checks if the given code is valid Python by attempting to parse it.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def extract_code_blocks(text):
    """
    Extracts Python code blocks from a given text using regex.
    If no code blocks are found, it checks if the entire text is valid Python code.
    """
    code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
    
    if code_blocks:  # If explicitly formatted code blocks exist
        return code_blocks
    elif is_valid_python(text.strip()):  # Check if the whole text is Python code
        return [text.strip()]
    else:  # If it's not valid Python code, return an empty list
        return []

def preprocess_code_blocks(code_blocks):
    """
    Remove code blocks that are not of interest:
    - pip install lines/blocks
    """
    preprocess_code = []

    for code in code_blocks:
        lines = code.split("\n")
        cleaned_lines = []

        for line in lines:
            # Check if the line is an install and remove it
            if line.startswith("pip install "):
                continue

            cleaned_lines.append(line)

        if cleaned_lines and max([len(line) for line in cleaned_lines])>2:
            preprocess_code.append("\n".join(cleaned_lines))

    return preprocess_code


def assemble_code_blocks(code_blocks):
    """
    Combines multiple extracted code blocks into a single script.
    - Removes duplicate imports.
    - Ensures logical flow.
    """
    assembled_code = []
    seen_imports = set()

    for code in code_blocks:
        lines = code.split("\n")
        cleaned_lines = []

        for line in lines:
            # Check if the line is an install and remove it
            if line.startswith("pip install "):
                continue
            
            # Check if the line is an import statement and prevent duplicates
            if line.startswith(("import ", "from ")):
                if line not in seen_imports:
                    seen_imports.add(line)
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)

        assembled_code.extend(cleaned_lines)
        assembled_code.append("\n")  # Add a newline between blocks for readability

    return "\n".join(assembled_code)


def safe_parse_code(code):
    """
    Parses the extracted code safely using AST.
    Returns the AST tree if successful, None otherwise.
    """
    try:
        tree = ast.parse(code)
        return tree
    except SyntaxError as e:
        print(f"SyntaxError while parsing code: {e}")
        return None

