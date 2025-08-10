# Gemini Code Assistant Context

## Project Overview

This project is a collection of Python scripts for scientific image analysis, with a focus on handling large TIFF images and associated GeoJSON annotations. It is currently a "frankenstein project" with various image processing functions, and the user's overall goal is to organize all these functions into a sensible package for private use. The scripts are designed to be run from the command line and leverage libraries like `dask` for out-of-core processing of large datasets, `tifffile` for reading and writing TIFF files, and `geopandas` for handling geospatial data.

The project is organized into three main modules:

*   **`ImageUtils`**: A collection of utility scripts for various image and data manipulation tasks.
*   **`pygeojson`**: A module for processing and converting GeoJSON files, particularly for extracting annotations.
*   **`pyimg`**: A module for core image processing tasks like downscaling, pyramidizing, and projection.

## User's Goals and Concerns

The user aims to refactor and organize this codebase into a well-structured Python package for private use. Key concerns and desired outcomes include:

*   **Addressing Redundancies**: Identify and eliminate duplicate or overlapping functionalities across the scripts.
*   **Handling Large Files**: Ensure efficient processing of larger-than-memory image files, building upon existing `dask` implementations.
*   **Simplifying Complexity**: Refactor and simplify functions that are currently too complex or difficult to maintain.
*   **Dual Interface**: All functions should be runnable both via command-line interface (CLI) and directly within Jupyter notebooks.
*   **Batch Processing**: Functions should accept input directories and be capable of batch processing images with a user-definable file extension (defaulting to `.tif`/`.tiff`).
*   **Google-style Docstrings**: Implement comprehensive Google-style docstrings for all functions to facilitate automated documentation generation.
*   **Documentation Site**: The docstrings should enable the creation of a browsable documentation website for the package.

## Key Scripts and Usage

The primary entry points for this project are the command-line scripts located in the `CLI` subdirectories of `pygeojson` and `pyimg`, and the scripts in the root of `ImageUtils`.

### Image Processing (`pyimg`)

*   **`downscale_image_dask.py`**: Scales down a large TIFF image to a lower bit depth (e.g., from 32-bit to 8-bit) to reduce file size.
    ```bash
    python src/pyimg/CLI/downscale_image_dask.py --input /path/to/large_image.tif --output /path/to/scaled_image.tif
    ```

*   **`pyramidize_ome_tif.py`**: Creates a multi-resolution image pyramid from an OME-TIFF file, which allows for efficient viewing and analysis of very large images.
    ```bash
    python src/pyimg/CLI/pyramidize_ome_tif.py --input /path/to/image.ome.tif --output /path/to/pyramid.ome.tif
    ```

### GeoJSON Processing (`pygeojson`)

*   **`annotated_geojson_to_csv.py`**: Converts a GeoJSON file with annotations into a CSV file, extracting relevant information about classified cells.
    ```bash
    python src/pygeojson/CLI/annotated_geojson_to_csv.py --input /path/to/annotations.geojson --output /path/to/annotations.csv
    ```

### Utilities (`ImageUtils`)

The `ImageUtils` directory contains a variety of scripts for more specific tasks. Refer to the individual scripts for their specific arguments and usage.

## Development Conventions

*   **Command-Line Interfaces**: All scripts are designed to be run from the command line and use the `argparse` library for argument parsing.
*   **Modularity**: The project is organized into modules (`pyimg`, `pygeojson`) with a separation between the command-line interface and the core Python logic.
*   **Logging**: The `loguru` library is used for logging in some of the scripts, providing clear and formatted output.
*   **Dependencies**: The project relies on a number of scientific Python libraries, including `dask`, `tifffile`, `geopandas`, `numpy`, and `loguru`. While a `requirements.txt` file is not present, these are the key dependencies.
