import argparse
from pathlib import Path
import dask.array as da
from dask_image import imread
import tifffile
from loguru import logger
import ome_types

from pyimg.utils import batch_process, describe_image, read_metadata

def check_compression(image_path: str) -> dict[int, str]:
    """Checks the compression schema of all pages in a TIFF/OME-TIFF image.

    Args:
        image_path: Path to the input image file.

    Returns:
        A dictionary mapping page indices to their compression schema names.
    """
    logger.info(f"Checking compression schema for: {image_path}")
    schemas = {}
    with tifffile.TiffFile(image_path) as tif:
        for i, page in enumerate(tif.pages):
            schemas[i] = page.compression.name
            logger.info(f"Page {i}: shape={page.shape}, dtype={page.dtype}, compression={page.compression.name}")
    return schemas

def compress_image(image_path: str, output_path: str, compression: str = "zlib"):
    """Converts/re-compresses a TIFF image to the desired compression schema using tiled, out-of-core writing.

    Args:
        image_path: Path to the input image file.
        output_path: Path to write the re-compressed image.
        compression: The target compression algorithm (zlib, lzw, jpeg, uncompressed).
    """
    describe_image(image_path)

    # Map user compression algorithm string to tifffile compression argument
    comp_map = {
        "zlib": "zlib",
        "lzw": "lzw",
        "jpeg": "jpeg",
        "uncompressed": None,
        "none": None,
    }
    
    comp_key = compression.lower().strip()
    if comp_key not in comp_map:
        logger.warning(f"Unknown compression '{compression}', passing through directly to tifffile.")
        target_compression = compression
    else:
        target_compression = comp_map[comp_key]

    # Load image lazily
    image = imread.imread(image_path)
    ndim = image.ndim

    # Rechunk to standard 512x512 spatial blocks for out-of-core computation
    logger.info(f"Dask chunks before rechunking: {image.chunksize}")
    chunks = {}
    if ndim >= 2:
        for i in range(ndim - 2):
            chunks[i] = 1
        chunks[ndim - 2] = 512
        chunks[ndim - 1] = 512
    else:
        chunks = {0: 512}
    image = image.rechunk(chunks=chunks)
    logger.info(f"Dask chunks after rechunking: {image.chunksize}")

    # Read and preserve OME metadata
    ome_metadata = read_metadata(image_path)
    ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8') if ome_metadata else None

    logger.info(f"Writing compressed image to '{output_path}' using '{compression}'...")
    write_kwargs = {"bigtiff": True, "description": ome_xml_bytes}
    if ndim >= 2:
        write_kwargs["tile"] = (512, 512)
        if target_compression:
            write_kwargs["compression"] = target_compression
    else:
        if target_compression:
            write_kwargs["compression"] = target_compression

    tifffile.imwrite(output_path, image, **write_kwargs)
    logger.success(f"Successfully created '{output_path}' ✨")

def main():
    parser = argparse.ArgumentParser(description="Check compression schema or re-compress a TIFF/OME-TIFF image.")
    parser.add_argument("-i", "--input", dest="input", type=str, required=True, help="Input image file or directory path.")
    parser.add_argument("-o", "--output", dest="output", type=str, help="Output file or directory path (required if not --check).")
    parser.add_argument("-c", "--compression", dest="compression", type=str, default="zlib", help="Target compression algorithm (zlib, lzw, jpeg, uncompressed). Defaults to zlib.")
    parser.add_argument("--check", dest="check", action="store_true", help="Only check existing compression schemas without compressing.")
    parser.add_argument("-f", "--force", dest="force", action="store_true", help="Force overwrite of existing files.")

    args = parser.parse_args()

    if args.check:
        input_path = Path(args.input).expanduser()
        if input_path.is_file():
            check_compression(str(input_path))
        elif input_path.is_dir():
            for f in input_path.glob("**/*.tif*"):
                check_compression(str(f))
        return

    if not args.output:
        parser.error("--output is required unless --check is specified.")

    func_kwargs = {'compression': args.compression}
    batch_process(
        process_func=compress_image,
        input_path=args.input,
        output_path=args.output,
        func_kwargs=func_kwargs,
        force=args.force,
        output_suffix="_compressed"
    )

if __name__ == "__main__":
    main()
