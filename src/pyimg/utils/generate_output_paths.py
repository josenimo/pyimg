import sys
from pathlib import Path
from typing import List

def get_stem_and_suffixes(path: Path) -> tuple[str, str]:
    """
    Returns the stem and suffix for a path, handling double extensions like .ome.tif/.ome.tiff.
    """
    filename = path.name
    for ext in ['.ome.tif', '.ome.tiff']:
        if filename.lower().endswith(ext):
            stem = filename[:-len(ext)]
            return stem, ext
    return path.stem, path.suffix

def generate_output_paths(
    input_paths: List[Path],
    original_input: str,
    output_path: str,
    suffix: str,
) -> List[Path]:
    """
    Generates a list of output paths corresponding to a list of input paths.
    Applies the suffix before the extensions, preserving double extensions.
    Maps subdirectories under original_input to subdirectories under output_path.
    """
    original_path = Path(original_input).expanduser()
    output_path_obj = Path(output_path).expanduser()
    output_paths = []

    if original_path.is_file():
        # Heuristic to check if output_path is intended to be a directory
        is_output_dir = (
            output_path_obj.is_dir() 
            or original_input.endswith('/') 
            or original_input.endswith('\\') 
            or output_path.endswith('/') 
            or output_path.endswith('\\')
            or not output_path_obj.suffix
        )
        if is_output_dir:
            output_path_obj.mkdir(parents=True, exist_ok=True)
            for in_path in input_paths:
                stem, ext = get_stem_and_suffixes(in_path)
                new_filename = f"{stem}{suffix}{ext}"
                output_paths.append(output_path_obj / new_filename)
        else:
            # Output path is a specific file path
            output_paths.append(output_path_obj)

    elif original_path.is_dir():
        # Ensure output directory exists
        output_path_obj.mkdir(parents=True, exist_ok=True)
        for in_path in input_paths:
            try:
                relative_path = in_path.relative_to(original_path)
            except ValueError:
                # Fallback if path is not relative
                relative_path = Path(in_path.name)
            stem, ext = get_stem_and_suffixes(in_path)
            parent_dir = output_path_obj / relative_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            new_filename = f"{stem}{suffix}{ext}"
            output_paths.append(parent_dir / new_filename)

    return output_paths