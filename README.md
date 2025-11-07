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
- **`project_v2.py`**: Provides the `merge_channels_dask` function to project channels in a Dask array.
- **`utils/`**: Contains helper functions for batch processing, file path generation, and image metadata handling.

## Future Work

- Centralized and improved logging.
- Add `tqdm` progress bars for batch processing.
- Implement channel overlay functionality.
- Ensure generated images are fully compatible with QuPath.
