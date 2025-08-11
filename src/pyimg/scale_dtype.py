import dask.array as da
import tifffile
import numpy as np
from dask_image import imread
import ome_types
import argparse
from loguru import logger
from ome_types.model import PixelType

from pyimg.utils import batch_process, describe_image, read_metadata

def scale_dtype(image_path, output_path, output_dtype="uint8"):

    # parse inputs
    ome_pixel_type = parse_output_dtype(output_dtype)
    
    #describe input
    describe_image(image_path)
    
    #load and change image metadata
    ome_metadata = read_metadata(image_path)
    ome_metadata.images[0].pixels.type = ome_pixel_type
    ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8')
    logger.info(f"Successfully updated Pixels.Type to '{np.dtype(output_dtype).name}'.")

    # load image, ASSUME CYX
    image = imread.imread(image_path)
    image = image.rechunk(chunks={0:1, 1:512, 2:512})
    logger.info(f"Image shape after squeeze: {image.shape}, ndim: {image.ndim}")
    min_max_axes = (1, 2)

    logger.info("Calculating min/max for all channels...")
    channel_mins, channel_maxs = da.compute(image.min(axis=min_max_axes), image.max(axis=min_max_axes))
    mins = channel_mins.reshape(-1, 1, 1)
    maxs = channel_maxs.reshape(-1, 1, 1)
    scales = maxs - mins
    scales[scales == 0] = 1

    # rescaling
    max_val_of_dtype = np.iinfo(output_dtype).max
    normalized_image = (image.astype(np.float32) - mins) / scales
    output_image = (normalized_image * max_val_of_dtype).astype(output_dtype)

    logger.info(f"Writing scaled image to '{output_path}'...")
    tifffile.imwrite( output_path, output_image, bigtiff=True, description=ome_xml_bytes)
    logger.success(f"Successfully created '{output_path}' ✨")

def main():
    parser = argparse.ArgumentParser(description="Scale the data type of a 3D image or a directory of images.")
    parser.add_argument("-i", "--input", dest="input", type=str, required=True, help="Input image file or directory path.")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Output file or directory path.")
    parser.add_argument("-d", "--dtype", dest="output_dtype", type=str, default="uint8", help="Target data type. Defaults to uint8.")
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

def parse_output_dtype(output_dtype):

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
        return ome_pixel_type
    except KeyError:
        print(f"Error: Unsupported or unknown dtype name '{output_dtype}'.")
        print(f"Supported types are: {list(NUMPY_TO_OME_TYPE_MAP.keys())}")
        return

if __name__ == "__main__":
    main()
