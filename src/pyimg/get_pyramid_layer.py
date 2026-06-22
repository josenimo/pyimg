import tifffile
import ome_types
import argparse
from loguru import logger

from pyimg.utils import batch_process, describe_image, read_metadata
def get_pyramid_layer(image_path, output_path, level):
    import dask.array as da

    # parse input
    if not isinstance(level, int):
        raise ValueError("level must be an integer")
    
    # describe input
    describe_image(image_path)

    # load and change metadata
    ome_metadata = read_metadata(image_path)
    ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8') if ome_metadata else None

    logger.info(f"Reading image: {image_path}")
    with tifffile.TiffFile(image_path) as tif:
        store = tif.series[0].levels[level].aszarr()
        image = da.from_zarr(store)
        ndim = image.ndim
        logger.info(f"Image shape: {image.shape}, ndim: {ndim}")

        logger.info(f"Writing layer image to '{output_path}'...")
        write_kwargs = {"bigtiff": True, "description": ome_xml_bytes}
        if ndim >= 2:
            write_kwargs["tile"] = (512, 512)
            write_kwargs["compression"] = "zlib"
        tifffile.imwrite(output_path, image, **write_kwargs)
        logger.success(f"Successfully created '{output_path}' ✨")

def main():
    parser = argparse.ArgumentParser(description="Multiscale ome.tif from numpy array")
    parser.add_argument("-r", "--input", dest="input", action="store", required=True)
    parser.add_argument("-o", "--output", dest="output", action="store", required=True)
    parser.add_argument("-l", "--level", dest="level", action="store", required=True, type=int, help="Level of pyramid to extract")
    parser.add_argument("-f", "--force", dest="force", action="store", help="Force overwrite of existing files.")
    args = parser.parse_args()

    func_kwargs = {'level': args.level}
    batch_process(
        process_func=get_pyramid_layer,
        input_path=args.input,
        output_path=args.output,
        func_kwargs=func_kwargs,
        force=args.force,
        output_suffix=f"_layer{args.level}"
    )

if __name__ == "__main__":
    main()
