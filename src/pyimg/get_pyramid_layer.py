import tifffile
import ome_types
import argparse
from loguru import logger

from pyimg.utils import batch_process, describe_image, read_metadata
from pyimg.utils.tiff_zarr import check_level, spatial_axes, write_level_tiled


def _rescale_ome_to_level(ome_metadata, series: tifffile.TiffPageSeries, level: int):
    """Update OME Pixels sizes and physical pixel sizes to match ``level``."""
    y_ax, x_ax = spatial_axes(series.axes)
    full_shape = series.levels[0].shape
    level_shape = series.levels[level].shape
    pixels = ome_metadata.images[0].pixels
    pixels.size_x = level_shape[x_ax]
    pixels.size_y = level_shape[y_ax]
    if pixels.physical_size_x is not None:
        pixels.physical_size_x *= full_shape[x_ax] / level_shape[x_ax]
    if pixels.physical_size_y is not None:
        pixels.physical_size_y *= full_shape[y_ax] / level_shape[y_ax]


def get_pyramid_layer(image_path, output_path, level):
    # parse input
    if not isinstance(level, int):
        raise ValueError("level must be an integer")

    with tifffile.TiffFile(image_path) as tif:
        check_level(tif, level)

    # describe input
    describe_image(image_path)

    # load and change metadata
    ome_metadata = read_metadata(image_path)

    logger.info(f"Reading level {level} of image: {image_path}")
    with tifffile.TiffFile(image_path) as tif:
        if ome_metadata is not None:
            _rescale_ome_to_level(ome_metadata, tif.series[0], level)
        ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8') if ome_metadata else None

        logger.info(f"Writing layer image to '{output_path}'...")
        write_level_tiled(tif, output_path, level=level, compression="zlib", description=ome_xml_bytes)
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
