from pathlib import Path
from typing import Callable, List, Dict, Any
from loguru import logger

from .generate_input_paths import generate_input_paths
from .generate_output_paths import generate_output_paths

def batch_process(
    process_func: Callable,
    input_path: str,
    output_path: str,
    func_kwargs: Dict[str, Any],
    force: bool = False,
    file_ending: str = ".tif",
    output_suffix: str = "_processed",
):
    """
    Applies a processing function to a batch of images, handling file I/O.

    Args:
        process_func: The function to apply to each image.
            (must accept input_path and output_path as first two arguments)
        input_path: Path to a single file or a directory of files.
        output_path: Path to the output file or directory.
        func_kwargs: A dictionary of keyword arguments to pass to the process_func.
        force: If True, overwrite existing output files.
        file_ending: The file extension to look for in the input directory.
        output_suffix: The suffix to add to output filenames.
    """
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)

    if input_path_obj.is_dir():
        input_files = generate_input_paths(input_path, file_ending=file_ending)
        if output_path_obj.is_file():
            logger.error("Input is a directory, but output is a file. Please specify an output directory.")
            return
        output_path_obj.mkdir(parents=True, exist_ok=True)
        output_files = generate_output_paths(input_files, input_path, suffix=output_suffix)

    elif input_path_obj.is_file():
        input_files = [input_path_obj]
        if output_path_obj.is_dir():
            output_path_obj.mkdir(parents=True, exist_ok=True)
            output_files = [output_path_obj / f"{input_path_obj.stem}{output_suffix}{input_path_obj.suffix}"]
        else:
            output_files = [output_path_obj]

    logger.info(f"Found {len(input_files)} image(s) to process.")

    for in_file, out_file in zip(input_files, output_files):
        if not force and out_file.exists():
            logger.info(f"Skipping {in_file.name}, output already exists.")
            continue
        
        logger.info(f"Processing {in_file.name} -> {out_file.name}")
        try:
            process_func(str(in_file), str(out_file), **func_kwargs)
        except Exception as e:
            logger.error(f"Failed to process {in_file.name}: {e}")
