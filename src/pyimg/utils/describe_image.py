import argparse
from pathlib import Path
import dask.array as da
import dask_image.imread
import tifffile
import numpy as np
from loguru import logger

def describe_image(file_path: Path):
    """
    Calculates and logs basic features and metadata for a TIFF image
    using Dask for out-of-core computation.
    """
    logger.info(f"\n--- Describing: {file_path} ---")
    try:
        # Step 1: Use dask_image to open the pixel data as a Dask array.
        # This does NOT load the image into memory.
        dask_arr = dask_image.imread.imread(str(file_path))
        logger.info(f"Successfully loaded image with Dask.")

        # Step 2: Use tifffile to open the file for metadata access only.
        with tifffile.TiffFile(file_path) as tif:
            # Get metadata from the first page/series
            page = tif.pages[0]
            tags = {tag.name: tag.value for tag in page.tags.values()}
            
            logger.info("\n## File Metadata (from tifffile)")
            logger.info(f"  - Software: {tags.get('software', 'N/A')}")
            logger.info(f"  - Description: {str(tags.get('image_description', 'N/A'))[:100]}...")
            logger.info(f"  - Resolution: {tags.get('x_resolution', 'N/A')}")
            logger.info(f"  - Photometric: {page.photometric.name}")

        # Step 3: Calculate array features using Dask.
        min_val, max_val, mean_val = da.min(dask_arr), da.max(dask_arr), da.mean(dask_arr)
        
        logger.info("\n## Dask Array Features")
        logger.info(f"  - Shape: {dask_arr.shape}")
        logger.info(f"  - Data Type: {dask_arr.dtype}")
        logger.info(f"  - Dask Chunks: {dask_arr.chunksize}")
        logger.info(f"  - On-Disk Size: {dask_arr.nbytes / 1e9:.2f} GB")

        # Step 4: Trigger the computation.
        logger.info("\nComputing statistics...")
        min_res, max_res, mean_res = da.compute(min_val, max_val, mean_val)
        
        logger.info(f"  - Min Pixel Value: {min_res}")
        logger.info(f"  - Max Pixel Value: {max_res}")
        logger.info(f"  - Mean Pixel Value: {mean_res:.2f}")

    except Exception as e:
        logger.info(f"❌ Failed to process {file_path}: {e}")