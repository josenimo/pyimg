"""
This script creates a multi-resolution OME-TIFF pyramid from a single 2D or 3D image.
It uses dask for out-of-core computation to handle large images and enables parallel processing.
"""

import argparse
import dask.array as da
import dask_image.imread
import tifffile
from loguru import logger

from utils.batch_process import batch_process
from utils.generate_input_paths import generate_input_paths
from utils.generate_output_paths import generate_output_paths
from utils.read_metadata import read_metadata

def pyramidize_image(input_path: str, output_path: str, min_size: int = 256):
    """
    Creates a multi-resolution OME-TIFF pyramid from a single image.
    Handles both 2D and 3D (with channels) images.

    Args:
        input_path (str): Path to the input TIFF image.
        output_path (str): Path to save the output OME-TIFF pyramid.
        min_size (int): The minimum size of the smallest pyramid level's spatial dimensions.
    """
    logger.info(f"Reading input image: {input_path}")
    img = dask_image.imread.imread(input_path)

    if img.ndim == 2:
        img = img[da.newaxis, :, :]
        logger.info("Image is 2D, adding a channel dimension.")

    metadata = read_metadata(input_path)

    logger.info(f"Generating pyramid for {input_path} with shape {img.shape}")
    pyramid = [img]

    while min(pyramid[-1].shape[1:]) > min_size:
        downsample_factors = {0: 1, 1: 2, 2: 2}
        downsampled = da.coarsen(da.mean, pyramid[-1], downsample_factors, trim_excess=True).astype(img.dtype)
        pyramid.append(downsampled)

    logger.info(f"Generated {len(pyramid)} pyramid levels.")

    logger.info(f"Writing pyramid to {output_path} using TiffWriter...")
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        for i, level in enumerate(pyramid):
            logger.debug(f"Writing level {i} with shape {level.shape}")
            tif.write(
                level,
                subfiletype=1 if i > 0 else 0,
                photometric=metadata.get("photometric", "minisblack"),
                compression=metadata.get("compression", "zlib"),
                metadata=metadata if i == 0 else None,
                subifds=(len(pyramid) - 1) if i == 0 else None,
            )
    logger.success(f"Successfully created pyramid: {output_path}")

def main():
    """Main function to parse arguments and run the batch processing."""
    parser = argparse.ArgumentParser(description="Create a multi-resolution OME-TIFF pyramid.")
    parser.add_argument("--input", required=True, help="Input file or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--ext", default="tif", help="File extension to process.")
    parser.add_argument("--min_size", type=int, default=256, help="Minimum size of the smallest pyramid level.")
    args = parser.parse_args()

    input_paths = generate_input_paths(args.input, ext=args.ext)
    output_paths = generate_output_paths(input_paths, args.output)

    from functools import partial
    processing_func = partial(pyramidize_image, min_size=args.min_size)

    batch_process(processing_func, input_paths, output_paths)

if __name__ == "__main__":
    main()
