# pyimg

A Python package for scientific image analysis, with a focus on handling large TIFF images. This package provides a set of tools for common image processing tasks, designed to work efficiently with larger-than-memory datasets by leveraging Dask for out-of-core computation.

All functionalities are available as both command-line interfaces (CLI) and Python functions for use in scripts or Jupyter notebooks.

## Features

- **Data Type Scaling**: Scale images to a lower bit depth (e.g., 32-bit to 8-bit) to reduce file size.
- **Image Pyramids**: Generate multi-resolution OME-TIFF pyramids for fast, interactive viewing of large images.
- **Channel Projection**: Merge multi-channel images into a single or multi-channel image using sum, mean, or max projections.
- **Image Binning**: Downsample images by a given integer factor.
- **Pyramid Layer Extraction**: Extract a specific resolution level from an image pyramid.
- **Batch Processing**: Process entire directories of images with a single command.
- **OME-TIFF Metadata**: Preserves and updates OME-TIFF metadata where applicable.

## Installation

To install the package and its dependencies, navigate to the project root and run:

```bash
pip install .
```

## Usage

### Command-Line Interface

Each processing module can be run directly from the command line. The scripts support processing single files or batch processing entire directories.

#### Example: Scaling an Image to 8-bit

```bash
python src/pyimg/scale_dtype.py --input /path/to/large_image.tif --output /path/to/scaled_image.tif --dtype uint8
```

#### Example: Creating an Image Pyramid

```bash
python src/pyimg/pyramidize.py --input /path/to/image.ome.tif --output /path/to/pyramid.ome.tif
```

### Python Functions

The core logic of each module can be imported and used directly in your Python scripts or notebooks.

```python
from pyimg.scale_dtype import scale_dtype
from pyimg.pyramidize import pyramidize

# Scale an image
scale_dtype("path/to/input.tif", "path/to/output_scaled.tif", output_dtype="uint8")

# Pyramidize an image
pyramidize("path/to/input.tif", "path/to/output_pyramid.tif")
```

## Core Modules

- **`scale_dtype.py`**: Scales the dynamic range of an image to a new data type (e.g., `uint8`, `uint16`).
- **`pyramidize.py`**: Creates a multi-resolution OME-TIFF pyramid.
- **`binning.py`**: Reduces image dimensions by an integer factor using mean coarsening.
- **`get_pyramid_layer.py`**: Extracts a single level from a pyramidal TIFF.
- **`project.py`**: Provides channel projection (max, mean, sum) using a marker CSV configuration.
- **`utils/`**: Contains helper functions for batch processing, file path generation, and image metadata handling.

## Testing

The package includes a comprehensive, parameterized test suite that validates behaviors across various dimensions (`XY`, `CYX`, `CXY`), metadata structures, and data types.

To run the test suite:
```bash
uv run pytest
```

---

## Roadmap & Next Steps

Below is the integrated development roadmap, consolidating outstanding tasks and requirements for the next session.

### 1. Refinement & Bug Fixes
- **Batch Processing Output & Suffix Fixes (`batch_process.py`):**
  - Ensure the user-specified `--output` path is honored in directories.
  - Fix double extension suffix replacement (e.g., so that `image.ome.tif` outputs `image_scaled.ome.tif` instead of `image.ome_scaled.tif`).
- **Centralized & Improved Logging:**
  - Standardize logging levels and formats.
  - Restrict channel metadata printout to a concise summary (instead of printing one line per channel).
  - Add execution timer logs to track image processing durations.
- **Dask Performance & Lazy Optimization:**
  - Log Dask array chunk sizes before and after rechunking steps.
  - Refactor `get_pyramid_layer.py` using `aszarr()` to load pyramid layers lazily.
  - Optimize memory usage during multi-channel projection saving in `project.py`.
- **QuPath Compatibility:**
  - Verify and configure output options (e.g. `tile=(512, 512)`) so generated OME-TIFF files open correctly in QuPath.

### 2. New Features & Capabilities
- **`compress.py` (Image Compression Converter):**
  - Implement a tool to check the existing compression schema of a TIFF/OME-TIFF image.
  - Convert/re-compress the image to a desired compression algorithm (e.g. LZW, Zlib, JPEG, Uncompressed) using tiled, out-of-core writing.
- **`cli.py` (Centralized Command-Line Entrypoint):**
  - Consolidate all independent command-line scripts into a unified CLI (e.g. using `click` or `argparse` to expose subcommands like `pyimg scale`, `pyimg bin`, `pyimg pyramid`, `pyimg compress`).
- **Background Subtraction & Quantification Updates:**
  - Adapt background subtraction metadata helper functions.
  - Develop/update `quant 2.0` and channel overlay features.

### 3. Benchmarking
- **Computational Benchmarking Suite:**
  - Implement a script or test harness to profile resource usage:
    - **RAM consumption** (Peak memory overhead).
    - **CPU computation time**.
    - **Wall-clock execution time**.
  - Specifically profile the performance when reading/writing large, multiplexed immunofluorescence (mIF) pyramidal images.
