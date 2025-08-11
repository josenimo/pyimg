from pathlib import Path
import dask.array as da
import dask_image.imread
from loguru import logger
import ome_types
import tifffile

def describe_image(file_path: Path):
    """
    Calculates and logs basic features and metadata for a TIFF image
    using Dask for out-of-core computation.
    """
    logger.info(f"\n--- Describing: {file_path} ---")

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    allowed_extensions = ['.tif', '.tiff']
    if file_path.suffix.lower() not in allowed_extensions:
        logger.error(f"Unsupported file type: '{file_path.suffix}'. Please provide a .tif or .tiff file.")
        return

    try:
        # --- Section 1: File Structure (Pyramid Check) ---
        logger.info("\n## FILE STRUCTURE ##")
        with tifffile.TiffFile(str(file_path)) as tif:
            series = tif.series[0]
            is_pyramid = len(series.levels) > 1
            
            if is_pyramid:
                logger.info(f"Image is a pyramid")
                num_levels = len(tif.series[0].levels)
                logger.info(f"  - Levels: {num_levels}")
                # Log the shape of each pyramid level
                shapes = [level.shape for level in tif.series[0].levels]
                logger.info(f"  - Level Shapes: {shapes}")

        dask_arr = dask_image.imread.imread(str(file_path))
        logger.info(f"Successfully loaded image with Dask.")

        # OME METADATA
        try:
            ome = ome_types.from_tiff(file_path)
            for i, image in enumerate(ome.images):

                pixels = image.pixels
                phys_x = f"{pixels.physical_size_x} {pixels.physical_size_x_unit.value}" if pixels.physical_size_x else "N/A"
                pixels_info = (
                    f"  - Image [{i}]: '{image.name or 'Unnamed'}'\n"
                    f"    - Dimensions (Order: {pixels.dimension_order.value}): "
                    f"X={pixels.size_x}, Y={pixels.size_y}, Z={pixels.size_z}, C={pixels.size_c}, T={pixels.size_t}\n"
                    f"    - Pixel Type: {pixels.type.value}\n"
                    f"    - Pixel Size (X): {phys_x}"
                )
                logger.info(pixels_info)

                if  pixels.channels:
                    logger.info(f"    - Channels ({len(pixels.channels)}):")
                    for chan in pixels.channels:
                        color_hex = f"#{chan.color}" if chan.color else "N/A"
                        logger.info(f"      - Name: '{chan.name or 'Unnamed'}', Color: {color_hex}")

        except ValueError:
            # This is common if the file is a standard TIFF without OME-XML
            logger.info("-> No OME-XML metadata found in the file.")


        # --- Section 2: Dask Array Properties ---
        logger.info("\n## ARRAY PROPERTIES ##")
        logger.info(f"-> Shape: {dask_arr.shape}")
        logger.info(f"-> Data Type: {dask_arr.dtype}")
        logger.info(f"-> Dask Chunks: {dask_arr.chunksize}")
        logger.info(f"-> On-Disk Size: {dask_arr.nbytes / 1e9:.3f} GB")

        # --- Section 3: Pixel Statistics ---
        logger.info("\n## PIXEL STATISTICS ##")
        logger.info("-> Computing min, max, and mean...")
        min_val, max_val, mean_val = da.min(dask_arr), da.max(dask_arr), da.mean(dask_arr)
        
        # This triggers the actual computation
        min_res, max_res, mean_res = da.compute(min_val, max_val, mean_val)
        
        logger.info(f"-> Min Pixel Value: {min_res}")
        logger.info(f"-> Max Pixel Value: {max_res}")
        logger.info(f"-> Mean Pixel Value: {mean_res:.2f}")
        logger.info("-" * (len(file_path.name) + 18))

    except Exception as e:
        logger.info(f"Failed to process {file_path}: {e}")