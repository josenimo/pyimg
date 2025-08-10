import dask.array as da
import tifffile
import numpy as np
from dask_image import imread
import ome_types
import argparse
from loguru import logger
from ome_types.model import PixelType

from .utils import read_metadata, create_basic_ome, batch_process, describe_image

def scale_dtype(image_path, output_path, output_dtype="uint8"):

    NUMPY_TO_OME_TYPE_MAP = {
        "int8": PixelType.INT8,
        "int16": PixelType.INT16,
        "int32": PixelType.INT32,
        "uint8": PixelType.UINT8,
        "uint16": PixelType.UINT16,
        "uint32": PixelType.UINT32,
        "float32": PixelType.FLOAT,
        "float64": PixelType.DOUBLE,
        "complex64": PixelType.COMPLEXFLOAT,
        "complex128": PixelType.COMPLEXDOUBLE,
        "bool": PixelType.BIT,
    }

    try:
        ome_pixel_type = NUMPY_TO_OME_TYPE_MAP[output_dtype]
    except KeyError:
        print(f"Error: Unsupported or unknown dtype name '{output_dtype}'.")
        print(f"Supported types are: {list(NUMPY_TO_OME_TYPE_MAP.keys())}")
        return

    logger.info(f"Reading image '{image_path}', targeting output type: {np.dtype(output_dtype).name}")
    describe_image(image_path)
    image = imread.imread(image_path).rechunk('auto')

    metadata = read_metadata(image_path)
    ome_obj = metadata["ome_metadata"]

    if ome_obj is None:
        logger.warning("No OME metadata found — creating basic OME-XML.")
        ome_obj = create_basic_ome(image)

    ome_obj.images[0].pixels.type = ome_pixel_type
    ome_xml_bytes = ome_types.to_xml(ome_obj).encode('utf-8')
    logger.info(f"Successfully updated Pixels.Type to '{np.dtype(output_dtype).name}'.")

    image = image.squeeze() # to remove singleton dimensions
    logger.info(f"Image shape after squeeze: {image.shape}, ndim: {image.ndim}")

    if image.ndim == 3:
        min_max_axes = (1, 2)
        channel_axis = 0
    elif image.ndim == 2:
        min_max_axes = (0, 1)
        channel_axis = None
    else:
        raise ValueError(f"Unsupported image dimensions after squeezing: {image.ndim}. Expected 2 or 3 dimensions.")

    logger.info("Calculating min/max for all channels...")
    channel_mins, channel_maxs = da.compute(image.min(axis=min_max_axes), image.max(axis=min_max_axes))
    
    if channel_axis is not None:
        mins = channel_mins.reshape(-1, 1, 1)
        maxs = channel_maxs.reshape(-1, 1, 1)
    else: # 2D image, no channel dimension
        mins = channel_mins
        maxs = channel_maxs

    scales = maxs - mins
    scales[scales == 0] = 1

    max_val_of_dtype = np.iinfo(output_dtype).max
    normalized_image = (image.astype(np.float32) - mins) / scales
    output_image = (normalized_image * max_val_of_dtype).astype(output_dtype)

    logger.info(f"Writing scaled image to '{output_path}'...")

    tifffile.imwrite(
        output_path,
        output_image,
        bigtiff=True,
        description=ome_xml_bytes
    )
    logger.success(f"Successfully created '{output_path}' ✨")

def main():
    parser = argparse.ArgumentParser(description="Scale the data type of a 3D image or a directory of images.")
    parser.add_argument("-i", "--input", dest="input", type=str, required=True, help="Input image file or directory path.")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Output file or directory path.")
    parser.add_argument(
        "-d", "--dtype", 
        dest="output_dtype", 
        type=str, 
        default="uint8", 
        help="Target data type (e.g., uint8, uint16). Defaults to uint8."
    )
    parser.add_argument("-f", "--force", dest="force", action="store_true", help="Force overwrite of existing files.")

    args = parser.parse_args()

    func_kwargs = {'output_dtype': args.output_dtype}

    batch_process(
        process_func=scale_dtype,
        input_path=args.input,
        output_path=args.output,
        func_kwargs=func_kwargs,
        force=args.force,
        output_suffix="_scaled"
    )

if __name__ == "__main__":
    main()
