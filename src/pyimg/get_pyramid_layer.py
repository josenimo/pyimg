import tifffile
import ome_types
import argparse
from loguru import logger

from pyimg.utils import batch_process, describe_image, read_metadata

def get_pyramid_layer(image_path, output_path, level):
    
    # parse input
    if not isinstance(level, int):
        raise ValueError("binning_factor must be an integer")
    
    # describe input
    describe_image(image_path)

    # load and change metadata
    ome_metadata = read_metadata(image_path)
    #change pixel to new layer?
    ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8')

    logger.info(f"Reading image: {image_path}")
    with tifffile.TiffFile(image_path) as tif:
        image = tif.series[0].levels[level].asarray() #lazy?
    logger.info(f"Image shape: {image.shape}")

    logger.info(f"Writing layer image to '{output_path}'...")
    tifffile.imwrite(output_path, output_path, bigtiff=True, description=ome_xml_bytes)
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
