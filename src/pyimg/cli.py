import click
import json
from pathlib import Path
from loguru import logger

from pyimg.scale_dtype import scale_dtype
from pyimg.binning import binning as binning_func
from pyimg.pyramidize import pyramidize as pyramidize_func
from pyimg.get_pyramid_layer import get_pyramid_layer
from pyimg.project import merge_channels_dask, config_from_csv, read_metadata
from pyimg.compress import check_compression, compress_image
from pyimg.utils import batch_process, describe_image

@click.group()
def cli():
    """pyimg: Scientific image analysis and optimization CLI."""
    pass

@cli.command("scale")
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True), help="Input image file or directory path.")
@click.option("-o", "--output", "output_path", required=True, type=click.Path(), help="Output file or directory path.")
@click.option("-d", "--dtype", "output_dtype", default="uint8", show_default=True, help="Target data type (e.g. uint8, uint16).")
@click.option("-f", "--force", is_flag=True, help="Force overwrite of existing files.")
def scale(input_path, output_path, output_dtype, force):
    """Scale the data type (dynamic range) of an image or a directory of images."""
    func_kwargs = {'output_dtype': output_dtype}
    batch_process(
        process_func=scale_dtype,
        input_path=input_path,
        output_path=output_path,
        func_kwargs=func_kwargs,
        force=force,
        output_suffix="_scaled"
    )

@cli.command("bin")
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True), help="Input image file or directory path.")
@click.option("-o", "--output", "output_path", required=True, type=click.Path(), help="Output file or directory path.")
@click.option("-b", "--binning", "binning_factor", default=2, show_default=True, type=int, help="Downsampling binning factor (integer).")
@click.option("-f", "--force", is_flag=True, help="Force overwrite of existing files.")
def bin_cmd(input_path, output_path, binning_factor, force):
    """Downsample image dimensions by a given integer factor."""
    func_kwargs = {'binning_factor': binning_factor}
    batch_process(
        process_func=binning_func,
        input_path=input_path,
        output_path=output_path,
        func_kwargs=func_kwargs,
        force=force,
        output_suffix="_bin"
    )

@cli.command("pyramid")
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True), help="Input image file or directory path.")
@click.option("-o", "--output", "output_path", required=True, type=click.Path(), help="Output file or directory path.")
@click.option("-m", "--min-size", "min_size", default=512, show_default=True, type=int, help="Minimum spatial dimension of smallest pyramid level.")
@click.option("-f", "--force", is_flag=True, help="Force overwrite of existing files.")
def pyramid(input_path, output_path, min_size, force):
    """Create a multi-resolution OME-TIFF pyramid."""
    func_kwargs = {'min_size': min_size}
    batch_process(
        process_func=pyramidize_func,
        input_path=input_path,
        output_path=output_path,
        func_kwargs=func_kwargs,
        force=force,
        output_suffix="_pyramid"
    )

@cli.command("layer")
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True), help="Input pyramidal image file or directory.")
@click.option("-o", "--output", "output_path", required=True, type=click.Path(), help="Output file or directory path.")
@click.option("-l", "--level", "level", required=True, type=int, help="Resolution level to extract (0-indexed, where 0 is highest resolution).")
@click.option("-f", "--force", is_flag=True, help="Force overwrite of existing files.")
def layer(input_path, output_path, level, force):
    """Extract a specific resolution level from a pyramidal TIFF image."""
    func_kwargs = {'level': level}
    batch_process(
        process_func=get_pyramid_layer,
        input_path=input_path,
        output_path=output_path,
        func_kwargs=func_kwargs,
        force=force,
        output_suffix=f"_layer{level}"
    )

@cli.command("project")
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True), help="Input image file or directory.")
@click.option("-o", "--output", "output_path", required=True, type=click.Path(), help="Output file or directory path.")
@click.option("--markers-csv", "markers_csv", required=True, type=click.Path(exists=True), help="Path to markers CSV config file.")
@click.option("--projection-method", "projection_method", default="sum", show_default=True, type=click.Choice(["sum", "mean", "max"]), help="The projection method to apply.")
@click.option("-f", "--force", is_flag=True, help="Force overwrite of existing files.")
def project(input_path, output_path, markers_csv, projection_method, force):
    """Merge subsets of channels based on a markers CSV file config."""
    from dask_image import imread
    import tifffile
    
    def process_image(image_path, output_path):
        describe_image(image_path)
        try:
            projection_config = config_from_csv(markers_csv, projection_method)
        except Exception:
            return

        if not projection_config:
            logger.warning("Projection config is empty. No output will be generated.")
            return

        img = imread.imread(image_path)
        while img.ndim > 3 and img.shape[0] == 1:
            img = img.squeeze(axis=0)
        
        # Rechunk to standard spatial blocks
        logger.info(f"Dask chunks before rechunking: {img.chunksize}")
        ndim = img.ndim
        chunks = {}
        if ndim >= 2:
            for i in range(ndim - 2):
                chunks[i] = 1
            chunks[ndim - 2] = 512
            chunks[ndim - 1] = 512
        else:
            chunks = {0: 512}
        img = img.rechunk(chunks=chunks)
        logger.info(f"Dask chunks after rechunking: {img.chunksize}")

        projected_img = merge_channels_dask(
            img,
            projection_config=projection_config,
            scale_before_project=True,
            scale_quantiles=(0.01, 0.999)
        )
        
        try:
            ome_metadata = read_metadata(image_path)
            if ome_metadata:
                ome_metadata.images[0].pixels.size_c = len(projection_config)
                ome_metadata.images[0].pixels.channels = ome_metadata.images[0].pixels.channels[:len(projection_config)]
                ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8')
            else:
                ome_xml_bytes = None
        except Exception as e:
            logger.warning(f"Could not update OME metadata: {e}")
            ome_xml_bytes = None

        import dask
        from pyimg.project import get_optimal_num_workers
        logger.info(f"Writing projected image to '{output_path}'...")
        write_kwargs = {"bigtiff": True, "description": ome_xml_bytes}
        if projected_img.ndim >= 2:
            write_kwargs["tile"] = (512, 512)
            write_kwargs["compression"] = "zlib"
        
        num_workers = get_optimal_num_workers()
        logger.info(f"Using {num_workers} Dask threads based on available memory limit.")
        with dask.config.set(num_workers=num_workers):
            tifffile.imwrite(output_path, projected_img, **write_kwargs)
        logger.success(f"Successfully created '{output_path}' ✨")

    batch_process(
        process_func=process_image,
        input_path=input_path,
        output_path=output_path,
        func_kwargs={},
        force=force,
        output_suffix="_projected"
    )

@cli.command("compress")
@click.option("-i", "--input", "input_path", required=True, type=click.Path(exists=True), help="Input image file or directory.")
@click.option("-o", "--output", "output_path", type=click.Path(), help="Output file or directory path (required if not --check).")
@click.option("-c", "--compression", "compression", default="zlib", show_default=True, help="Target compression algorithm (zlib, lzw, jpeg, uncompressed).")
@click.option("--check", is_flag=True, help="Only check existing compression schemas without compressing.")
@click.option("-f", "--force", is_flag=True, help="Force overwrite of existing files.")
def compress(input_path, output_path, compression, check, force):
    """Check image compression schema or convert compression using tiled out-of-core writing."""
    if check:
        in_p = Path(input_path).expanduser()
        if in_p.is_file():
            check_compression(str(in_p))
        elif in_p.is_dir():
            for f in in_p.glob("**/*.tif*"):
                check_compression(str(f))
        return

    if not output_path:
        click.echo("Error: --output is required unless --check is specified.", err=True)
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit(1)

    func_kwargs = {'compression': compression}
    batch_process(
        process_func=compress_image,
        input_path=input_path,
        output_path=output_path,
        func_kwargs=func_kwargs,
        force=force,
        output_suffix="_compressed"
    )

@cli.command("benchmark")
@click.option("-i", "--input", "input_path", required=False, type=click.Path(exists=True), help="Input image file. Defaults to TD_15_TNBC_subset.ome.tif.")
@click.option("-o", "--output", "output_path", required=False, type=click.Path(), help="Output file or directory path.")
@click.option("--task", "task_name", default="all", show_default=True, type=click.Choice(["all", "scale", "bin", "pyramid", "layer", "project", "compress"]), help="The operation to benchmark. 'all' runs the entire suite.")
@click.option("--task-args", "task_args_json", default="{}", show_default=True, help="JSON string representing keyword arguments for the task.")
@click.option("--keep-images", is_flag=True, help="Keep benchmark output images for human QC.")
def benchmark(input_path, output_path, task_name, task_args_json, keep_images):
    """Run computational benchmarking suite for resource profiling."""
    from pyimg.benchmark import run_benchmark, run_benchmarking_suite
    try:
        task_args = json.loads(task_args_json)
    except json.JSONDecodeError as e:
        click.echo(f"Error parsing --task-args as JSON: {e}", err=True)
        return

    if task_name == "all":
        run_benchmarking_suite(input_path, output_path, keep_images=keep_images)
    else:
        if not output_path:
            click.echo("Error: --output is required for profiling individual tasks.", err=True)
            return
        run_benchmark(input_path, output_path, task_name, task_args)

if __name__ == "__main__":
    cli()
