import dask.array as da
import tifffile
import numpy as np
from dask_image import imread
import ome_types
import argparse
from loguru import logger
from ome_types import model

from .utils import batch_process, describe_image
from .utils import read_metadata, create_basic_ome


def binning(image_path, output_path, binning_factor:int = 2):

    # parse input
    if not isinstance(binning_factor, int):
        raise ValueError("binning_factor must be an integer")

    # describe input
    describe_image(image_path)

    # load and change metadata
    ome_metadata = read_metadata(image_path)
    # unsure how to change binning
    ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8')

    # load image, assume CYX
    image = imread.imread(image_path)
    image = image.rechunk(chunks={0:1, 1:512, 2:512})
    logger.info(f"Image shape after squeeze: {image.shape}, ndim: {image.ndim}")

    axes_to_bin = {1: binning_factor, 2: binning_factor}
    output_image = da.coarsen(np.mean, image, axes_to_bin, trim_excess=True)

    logger.info(f"Writing binned image to '{output_path}'...")
    tifffile.imwrite(output_path, output_image, bigtiff=True, description=ome_xml_bytes)
    logger.success(f"Successfully created '{output_path}' ✨")

def main():
    parser = argparse.ArgumentParser(description="Scale the data type of a 3D image or a directory of images.")
    parser.add_argument("-i", "--input", dest="input", type=str, required=True, help="Input image file or directory path.")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Output file or directory path.")
    parser.add_argument("-b", "--binning", dest="binning_factor", type=str, default=2, help="Target binning. Defaults to 2.")
    parser.add_argument("-f", "--force", dest="force", action="store_true", help="Force overwrite of existing files.")

    args = parser.parse_args()
    func_kwargs = {'output_binning': args.output_binning}
    batch_process(
        process_func=binning,
        input_path=args.input,
        output_path=args.output,
        func_kwargs=func_kwargs,
        force=args.force,
        output_suffix="_bin"
    )

if __name__ == "__main__":
    main()
