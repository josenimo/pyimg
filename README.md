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

## Completed Roadmap Items

### 1. Refinement & Bug Fixes
- **Batch Processing Output & Suffix Fixes (`batch_process.py`):**
  - Honored user-specified output directory path.
  - Implemented `get_stem_and_suffixes` to correctly handle double extension suffixes (e.g., `image.ome.tif` -> `image_scaled.ome.tif`).
- **Centralized & Improved Logging:**
  - Standardized channel lists in `describe_image` to a concise single-line summary to prevent terminal pollution.
  - Added elapsed execution timer logs to track image processing durations for all batch processing runs.
- **Dask Performance & Lazy Optimization:**
  - Logged Dask array chunk sizes before and after every rechunking operation.
  - Refactored `get_pyramid_layer.py` to use `aszarr()` and `da.from_zarr()` for lazy loading of pyramid layers.
  - Squeezed leading singleton dimensions on projected image load and rechunked the Dask arrays to spatial tiles (`512x512`), removing memory-intensive `.compute()` operations during saving.
- **QuPath Compatibility:**
  - Set default writing options to `tile=(512, 512)` and `compression="zlib"` for all modules, enabling immediate viewing in QuPath.

### 2. New Features & Capabilities
- **`compress.py` (Image Compression Converter):**
  - Added `check_compression` to check/list compression schemas.
  - Added `compress_image` to convert TIFF compressions (Zlib, LZW, JPEG, or Uncompressed) out-of-core.
- **`cli.py` (Unified CLI Entrypoint):**
  - Consolidated all scripts into a unified Click entrypoint (`pyimg`). Exposed subcommands: `scale`, `bin`, `pyramid`, `layer`, `project`, `compress`, and `benchmark`.
- **Benchmarking Suite:**
  - Implemented `benchmark.py` mapping execution resource utilization (peak RAM footprint via process RSS polling thread, CPU process time, and Wall-clock time).

---

## Next Steps & Deferred Tasks

- **Background Subtraction & Quantification Updates:**
  - Adapt background subtraction metadata helper functions.
  - Develop/update `quant 2.0` and channel overlay features.

---

## Design Decisions & Conventions

To keep development consistent between sessions, please adhere to the following decisions:

1. **Lazy Loading and Out-of-Core Processing**:
   - Always load TIFF arrays lazily using `dask_image.imread.imread` or `tifffile`'s `aszarr()` wrapped in `dask.array.from_zarr`.
   - Never call `.compute()` on large Dask arrays before writing to disk. Instead, pass the Dask array directly to `tifffile.imwrite` along with a valid tile configuration (e.g. `tile=(512, 512)`) and compression type. This directs `tifffile` to evaluate and write the array chunk-by-chunk/tile-by-tile.
2. **Double Extensions Suffix Handling**:
   - Use the `get_stem_and_suffixes` helper function in `src/pyimg/utils/generate_output_paths.py` when appending suffixes to filenames. This ensures double extensions like `.ome.tif` or `.ome.tiff` are preserved.
3. **QuPath & Tiling Standard**:
   - All spatial image outputs (2D or higher) should be written with tiling (`tile=(512, 512)`) and zlib compression (`compression="zlib"`) to guarantee compatibility with QuPath.
4. **Dimension Squeezing**:
   - Squeeze leading singleton dimensions (Z, T, or C) from Dask arrays if they represent 4D structures (e.g., `(1, C, Y, X)`) before applying spatial functions or projections that expect `(C, Y, X)` layout.
5. **Monkeypatching Zarr v3 Stores**:
   - Zarr v3 abstract classes require implementations for `set_partial_values` and `supports_partial_writes`. Since `tifffile`'s `ZarrTiffStore` inherits from `Store` but does not define these, they are monkeypatched dynamically on package initialization in `src/pyimg/__init__.py`.
6. **Dependencies Management**:
   - Use `uv add <package>` instead of `uv pip install <package>` to manage dependency declarations in `pyproject.toml`.

