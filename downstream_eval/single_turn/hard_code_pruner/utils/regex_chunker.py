from typing import List
import re
from pathlib import Path
import time
def split_code_by_functions(code: str, language: str = "python", custom_separator: str = "# --CHUNK_SEPARATOR-- #") -> List[str]:
    """
    Split code into chunks based on function and class definitions for various languages.
    Also splits on custom separator if provided.
    
    Args:
        code: The code to split
        language: Programming language of the code (python, cpp, java, typescript, rust, go)
        custom_separator: Optional custom separator string to also split on
        
    Returns:
        List of code chunks, each containing a function, class, or class method
    """
    print(f"Splitting code by functions and classes for language: {language}")
    start_time = time.time()
    
    # Define regex patterns for different languages
    patterns = {
        # Python: Simplified to match 'def' or 'class' followed by content until the next def/class or end
        "python": r'(^|\n)(\s*)(def|class)\s+[^\n]+(\n(?!\s*(?:def|class)\s)[^\n]*)*',
        # C++: Improved to better handle multi-line declarations
        "cpp": r'(^|\n)(\s*)(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*:\s*[^{]*)?|(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*\s+)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
        # Java: Improved for multi-line method declarations
        "java": r'(^|\n)(\s*)(?:(?:public|private|protected|static|final|abstract|synchronized)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:<.*>)?(?:[a-zA-Z_][a-zA-Z0-9_<>:,\s]*)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*throws\s+[^{;]*)?)\s*(?:{[^}]*}|[^;]*;)?',
        # TypeScript: Enhanced to handle multi-line methods and arrow functions
        "typescript": r'(^|\n)(\s*)(?:(?:public|private|protected|static|abstract)\s+)*(?:class\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+implements\s+[^{]*)?|(?:(?:public|private|protected|static|async)\s+)*(?:function\s+)?(?:[a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<.*>)?\s*\([^{;]*\)\s*(?::\s*[^{;]*\s*)?(?:=>)?)\s*(?:{[^}]*}|[^;]*;)?',
        # Rust: Improved for multi-line function declarations
        "rust": r'(^|\n)(\s*)(?:pub\s+)?(?:struct\s+[a-zA-Z_][a-zA-Z0-9_]*|impl(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)?(?:\s+for\s+[a-zA-Z_][a-zA-Z0-9_]*)?|(?:async\s+)?fn\s+[a-zA-Z_][a-zA-Z0-9_]*\s*(?:<.*>)?\s*\([^{;]*\)(?:\s*->\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
        # Go: Improved for multi-line function declarations
        "go": r'(^|\n)(\s*)(?:type\s+[a-zA-Z_][a-zA-Z0-9_]*\s+struct|func\s+(?:\([^)]*\)\s*)?[a-zA-Z_][a-zA-Z0-9_]*\s*\([^{;]*\)(?:\s*[^{;]*\s*)?)\s*(?:{[^}]*}|[^;]*;)?',
    }
    
    # Use default Python pattern if language not supported
    if language.lower() not in patterns:
        language = "python"
    
    # First check if we need to split by custom separator
    separator_chunks = []
    if custom_separator and custom_separator in code:
        print(f"Custom separator '{custom_separator}' found, first splitting by separator")
        separator_chunks = [chunk for chunk in code.split(custom_separator) if chunk.strip()]
    else:
        separator_chunks = [code]  # Just one chunk - the entire code

    # Function to split a single chunk by functions/classes
    def split_chunk_by_pattern(chunk_code):
        function_pattern = re.compile(patterns[language.lower()], re.MULTILINE)
        matches = list(function_pattern.finditer(chunk_code))
        
        if not matches:
            return [chunk_code]  # No matches, return whole chunk
            
        result_chunks = []
        
        # Add code before first match
        if matches[0].start() > 0:
            result_chunks.append(chunk_code[:matches[0].start()])
        
        # Process each match
        for i, match in enumerate(matches):
            start = match.start()
            
            # End is either start of next match or end of code
            if i < len(matches) - 1:
                end = matches[i + 1].start()
            else:
                end = len(chunk_code)
            
            result_chunks.append(chunk_code[start:end])
        
        return result_chunks
    
    # Now apply function/class splitting to each separator chunk
    final_chunks = []
    for chunk in separator_chunks:
        function_chunks = split_chunk_by_pattern(chunk)
        final_chunks.extend(function_chunks)
    
    end_time = time.time()
    print(f"Code splitting completed in {end_time - start_time:.2f} seconds")
    print(f"Split code into {len(final_chunks)} chunks (using both separator and patterns)")
    
    return final_chunks

if __name__ == "__main__":
    file = Path(__file__).parent / "code_stmt_splitter.py"
    code = file.read_text()
    chunks = split_code_by_functions(code)
    for i, c in enumerate(chunks):
        print(f"--- Chunk {i} ---")
        print(c)
