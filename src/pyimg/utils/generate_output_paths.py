import sys
from pathlib import Path
from typing import List

def generate_output_paths(
    input_paths: List[Path],
    original_input: str,
    suffix: str,
    output_dir_name: str = "output"
) -> List[Path]:
    """
    Generates a list of output paths corresponding to a list of input paths.

    - If the original input was a directory, it creates a new directory
      (e.g., 'output') and places all output files inside it.
    - If the original input was a single file, the output file is placed
      in the same directory.

    Args:
        input_paths: The list of files to be processed (from find_files).
        original_input: The original path string the user provided.
        suffix: A string to add to the output filenames (e.g., '_scaled').
        output_dir_name: The name for the new directory if one is created.

    Returns:
        A list of Path objects for each output file, matching the input order.
    """
    original_path = Path(original_input).expanduser()
    output_paths = []

    if original_path.is_file():
        for in_path in input_paths: # This loop will only run once
            new_filename = f"{in_path.stem}{suffix}{in_path.suffix}"
            output_paths.append(in_path.parent / new_filename)

    elif original_path.is_dir():
        output_dir = original_path.parent / f"{original_path.name}_{output_dir_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved in: {output_dir}")
        
        for in_path in input_paths:
            new_filename = f"{in_path.stem}{suffix}{in_path.suffix}"
            output_paths.append(output_dir / new_filename)

    return output_paths