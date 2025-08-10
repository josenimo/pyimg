import sys
from pathlib import Path
from typing import List

def generate_input_paths(input_path: str, file_ending: str) -> List[Path]:
    """
    Resolves an input path to a list of file paths with user-friendly checks.

    - Expands the tilde character ('~') to the user's home directory.
    - If input_path is a file, it checks if it has the correct ending.
    - If input_path is a directory, it finds all files with the correct
      ending, searching recursively through all subdirectories.

    Args:
        input_path: The path to a file or directory.
        file_ending: The file extension to look for (e.g., ".tif", ".txt").

    Returns:
        A list of Path objects for each file to be processed.
    
    Raises:
        FileNotFoundError: If the input_path does not exist after checks.
    """

    # Create the Path object and expand the tilde ('~')
    path = Path(input_path).expanduser()
    
    if not path.exists():
        raise FileNotFoundError(f"Error: The path '{path}' does not exist.")

    if path.is_file():
        if path.suffix.lower() == file_ending.lower():
            return [path]
        else:
            print(f"Warning: '{path.name}' is not a '{file_ending}' file. Skipping.")
            return []

    elif path.is_dir():
        return list(path.glob(f'**/*{file_ending}'))
        
    else:
        print(f"Warning: '{input_path}' is not a valid file or directory. Skipping.")
        return []