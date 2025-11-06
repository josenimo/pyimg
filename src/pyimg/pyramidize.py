import argparse
import dask.array as da
from dask_image import imread
import tifffile
from loguru import logger
import ome_types

from pyimg.utils import batch_process, describe_image, read_metadata

def pyramidize(image_path: str, output_path: str, min_size: int = 512):
    """
    Creates a multi-resolution OME-TIFF pyramid from a single image.
    Handles both 2D and 3D (with channels) images.

    Args:
        input_path (str): Path to the input TIFF image.
        output_path (str): Path to save the output OME-TIFF pyramid.
        min_size (int): The minimum size of the smallest pyramid level's spatial dimensions.
    """
    # parse inputs
    if not isinstance(min_size, int):
        raise ValueError("min_size must be an integer")

    # describe input
    describe_image(image_path)

    # load and change image metadata
    ome_metadata = read_metadata(image_path)
    # unsure how to log this into metadata
    ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8')

    # load image, ASSUME CYX
    image = imread.imread(image_path)
    image = image.rechunk(chunks={0:1, 1:512, 2:512})
    logger.info(f"Image in dask: {image.shape}, ndim: {image.ndim}, chunksize {image.chunksize}")

    # create pyramid lazy objects
    logger.info(f"Generating pyramid for {image_path} with shape {image.shape}")
    pyramid = [image]
    while min(pyramid[-1].shape[1:]) > min_size:
        downsample_factors = {0: 1, 1: 3, 2: 3}
        downsampled = da.coarsen(da.mean, pyramid[-1], downsample_factors, trim_excess=True).astype(image.dtype)
        pyramid.append(downsampled)
    logger.info(f"Generated {len(pyramid)} pyramid levels.")

    # write pyramid
    logger.info(f"Writing pyramid to {output_path} using TiffWriter...")
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        for i, level in enumerate(pyramid):
            logger.debug(f"Writing level {i} with shape {level.shape}")
            tif.write(
                level,
                subfiletype= 1 if i > 0 else 0,
                photometric= "minisblack",
                compression= "zlib",
                description= ome_xml_bytes if i == 0 else None,
                subifds=(len(pyramid) - 1) if i == 0 else None,
            )
    logger.success(f"Successfully created pyramid '{output_path}' ✨")

def main():
    """Main function to parse arguments and run the batch processing."""
    parser = argparse.ArgumentParser(description="Create a multi-resolution OME-TIFF pyramid.")
    parser.add_argument("-i", "--input", required=True, help="Input file or directory.")
    parser.add_argument("-o", "--output", required=True, help="Output directory.")
    parser.add_argument("-m", "--min_size", type=int, default=512, help="Minimum size of the smallest pyramid level.")
    parser.add_argument("-f", "--force", dest="force", action="store_true", help="Force overwrite of existing files.")
    args = parser.parse_args()

    func_kwargs = {'min_size': args.min_size}

    batch_process(
        process_func = pyramidize, 
        input_path = args.input, 
        output_path = args.output,
        func_kwargs = func_kwargs,
        force = args.force,
        output_suffix = "_pyramid"
        )

if __name__ == "__main__":
    main()
