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

    ###### FOR NOW WE IGNORE BINNING METADATA ######

    # STR_TO_OME_BINNING_MAP = {
    #     "1x1": model.Binning.ONEBYONE, "2x2": model.Binning.TWOBYTWO,
    #     "4x4": model.Binning.FOURBYFOUR, "8x8": model.Binning.EIGHTBYEIGHT,
    # }

    # OME_TO_INT_MAP = {v: int(k.split('x')[0]) for k, v in STR_TO_OME_BINNING_MAP.items()}

    # try:
    #     target_ome_binning = STR_TO_OME_BINNING_MAP[output_binning]
    #     target_bin_factor = OME_TO_INT_MAP[target_ome_binning]
    # except KeyError:
    #     logger.error(f"Error: Unsupported or unknown dtype name '{output_binning}'.")
    #     logger.error(f"Supported types are: {list(STR_TO_OME_BINNING_MAP.keys())}")
    #     return

    # logger.info(f"Reading image '{image_path}', binning to {(output_binning)}")
    
    # metadata = read_metadata(image_path)
    # ome_obj = metadata["ome_metadata"]

    # if ome_obj is None:
    #     logger.warning("No OME metadata found — creating basic OME-XML.")
    #     ome_obj = create_basic_ome(image)

    # current_ome_binning = ome_obj.images[0].binning #not sure will work
    # current_bin_factor = OME_TO_INT_MAP.get(current_ome_binning, 1)

    # if target_bin_factor < current_bin_factor:
    #     logger.warning(f"Cannot bin down from {current_bin_factor}x{current_bin_factor} to {target_bin_factor}x{target_bin_factor}.")
    #     logger.warning("This operation would require upsampling, which is not supported. Aborting.")
    #     return
    # if target_bin_factor == current_bin_factor:
    #     logger.info(f"Image is already at the desired {target_bin_factor}x{target_bin_factor} binning. No processing needed.")
    #     return
    # else:
    #     if target_bin_factor % current_bin_factor != 0:
    #         logger.error(f"Invalid binning transition: Cannot cleanly go from {current_bin_factor}x{current_bin_factor} to {target_bin_factor}x{target_bin_factor}.")
    #         return
    #     bin_factor_to_apply = target_bin_factor // current_bin_factor
    #     logger.info(f"Applying a relative {bin_factor_to_apply}x{bin_factor_to_apply} binning to reach target.")

    # ome_obj.images[0].binning = target_ome_binning
    # ome_xml_bytes = ome_types.to_xml(ome_obj).encode('utf-8')
    # logger.info(f"Successfully updated Binning to '{target_ome_binning}'.")

    describe_image(image_path)
    image = imread.imread(image_path).rechunk('auto')

    image = image.squeeze()
    logger.info(f"Image shape after squeeze: {image.shape}, ndim: {image.ndim}")

    if image.ndim == 2:
        axes_to_bin = {0: binning_factor, 1: binning_factor}
        output_image = da.coarsen(np.mean, image, axes_to_bin, trim_excess=True)
    elif image.ndim == 3:
        #assumes CYX
        axes_to_bin = {1: binning_factor, 2: binning_factor}
        output_image = da.coarsen(np.mean, image, axes_to_bin, trim_excess=True)
    else:
        logger.error(f"Unsupported image dimensions: {image.ndim}. Can only bin 2D or 3D images.")
        return

    logger.info(f"Writing binned image to '{output_path}'...")
    tifffile.imwrite(
        output_path, 
        output_image, 
        bigtiff=True, 
        # description=ome_xml_bytes
        )
    logger.success(f"Successfully created '{output_path}' ✨")

def main():
    parser = argparse.ArgumentParser(description="Scale the data type of a 3D image or a directory of images.")
    parser.add_argument("-i", "--input", dest="input", type=str, required=True, help="Input image file or directory path.")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Output file or directory path.")
    parser.add_argument(
        "-b", "--binning", 
        dest="binning_factor", 
        type=str, 
        default=2, 
        help="Target binning (e.g., 2, 3, 4). Defaults to 2."
    )
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
