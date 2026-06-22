import dask.array as da
import tifffile
from dask_image import imread
import ome_types
import numpy as np
import argparse
from loguru import logger
from ome_types.model import OME, Image, Pixels, Channel
from pyimg.scale_dtype import parse_output_dtype

from pyimg.utils import batch_process, describe_image, read_metadata

def binning(image_path, output_path, binning_factor:int = 2):

    # parse input
    if not isinstance(binning_factor, int):
        raise ValueError("binning_factor must be an integer")

    # describe input
    describe_image(image_path)

    # load image
    image = imread.imread(image_path)
    ndim = image.ndim
    shape = image.shape
    logger.info(f"Image shape: {shape}, ndim: {ndim}")

    # dynamic rechunking
    chunks = {}
    if ndim >= 2:
        for i in range(ndim - 2):
            chunks[i] = 1
        chunks[ndim - 2] = 512
        chunks[ndim - 1] = 512
    else:
        chunks = {0: 512}
    
    logger.info(f"Dask chunks before rechunking: {image.chunksize}")
    image = image.rechunk(chunks=chunks)
    logger.info(f"Dask chunks after rechunking: {image.chunksize}")

    # perform binning on spatial dimensions
    if ndim >= 2:
        axes_to_bin = {ndim - 2: binning_factor, ndim - 1: binning_factor}
    else:
        axes_to_bin = {0: binning_factor}
    
    logger.info(f"Coarsening image with factors {axes_to_bin}...")
    output_image = da.coarsen(da.mean, image, axes_to_bin, trim_excess=True).astype(image.dtype)
    out_shape = output_image.shape
    logger.info(f"Binned image shape: {out_shape}")

    # load and change metadata
    ome_metadata = read_metadata(image_path)
    if ome_metadata is None:
        logger.warning(
            "\n"
            "################################################################################\n"
            "LOUD WARNING: Input image does not contain OME-XML metadata! Creating placeholder metadata.\n"
            "################################################################################\n"
        )
        # Create placeholder metadata based on output image dimensions
        ome_pixel_type = parse_output_dtype(np.dtype(output_image.dtype).name)
        
        size_x = out_shape[-1]
        size_y = out_shape[-2] if ndim >= 2 else 1
        size_c = out_shape[-3] if ndim >= 3 else 1
        size_z = out_shape[-4] if ndim >= 4 else 1
        size_t = out_shape[-5] if ndim >= 5 else 1
        
        channels = [Channel(id=f"Channel:0:{i}", name=f"Channel_{i}") for i in range(size_c)]
        
        pixels = Pixels(
            id="Pixels:0",
            dimension_order="XYCZT",
            type=ome_pixel_type,
            size_x=size_x,
            size_y=size_y,
            size_c=size_c,
            size_z=size_z,
            size_t=size_t,
            channels=channels
        )
        
        ome_metadata = OME()
        ome_metadata.images.append(Image(id="Image:0", pixels=pixels))
    else:
        # Update existing metadata
        pixels = ome_metadata.images[0].pixels
        
        # Update sizes
        pixels.size_x = out_shape[-1]
        if ndim >= 2:
            pixels.size_y = out_shape[-2]
            
        # Coarsen physical pixel size if present
        if pixels.physical_size_x is not None:
            pixels.physical_size_x *= binning_factor
        if pixels.physical_size_y is not None:
            pixels.physical_size_y *= binning_factor

    ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8')

    logger.info(f"Writing binned image to '{output_path}'...")
    tifffile.imwrite(output_path, output_image, bigtiff=True, description=ome_xml_bytes)
    logger.success(f"Successfully created '{output_path}' ✨")

def main():
    parser = argparse.ArgumentParser(description="Scale the data type of a 3D image or a directory of images.")
    parser.add_argument("-i", "--input", dest="input", type=str, required=True, help="Input image file or directory path.")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Output file or directory path.")
    parser.add_argument("-b", "--binning", dest="binning_factor", type=int, default=2, help="Target binning. Defaults to 2.")
    parser.add_argument("-f", "--force", dest="force", action="store_true", help="Force overwrite of existing files.")

    args = parser.parse_args()
    func_kwargs = {'binning_factor': args.binning_factor}
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
