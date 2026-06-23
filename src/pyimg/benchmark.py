import time
import threading
import psutil
import os
import tempfile
from pathlib import Path
from loguru import logger

from pyimg.scale_dtype import scale_dtype
from pyimg.binning import binning as binning_func
from pyimg.pyramidize import pyramidize as pyramidize_func
from pyimg.get_pyramid_layer import get_pyramid_layer
from pyimg.compress import compress_image

class MemoryTracker:
    """Utility class to track peak memory usage of the current process using a background thread."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.peak_memory = 0
        self.running = False
        self._thread = None
        self.process = psutil.Process(os.getpid())

    def start(self):
        """Starts tracking memory usage."""
        self.peak_memory = self.process.memory_info().rss
        self.running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def _run(self):
        while self.running:
            try:
                mem = self.process.memory_info().rss
                if mem > self.peak_memory:
                    self.peak_memory = mem
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> int:
        """Stops tracking memory usage and returns peak usage in bytes."""
        self.running = False
        if self._thread:
            self._thread.join()
        try:
            mem = self.process.memory_info().rss
            if mem > self.peak_memory:
                self.peak_memory = mem
        except Exception:
            pass
        return self.peak_memory

def run_benchmark(input_path: str, output_path: str, task_name: str, task_kwargs: dict):
    """Profiles resource usage of a processing task on an image.

    Args:
        input_path: Path to the input image file.
        output_path: Path to write the output image file.
        task_name: The operation to profile (scale, bin, pyramid, layer, project, compress).
        task_kwargs: Dictionary of keyword arguments to pass to the function.
    """
    logger.info(f"Initializing benchmark for task: {task_name}")
    
    # 1. Initialize tracker
    tracker = MemoryTracker()
    tracker.start()
    
    # 2. Track timing
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    
    try:
        if task_name == "scale":
            scale_dtype(input_path, output_path, **task_kwargs)
        elif task_name == "bin":
            binning_func(input_path, output_path, **task_kwargs)
        elif task_name == "pyramid":
            pyramidize_func(input_path, output_path, **task_kwargs)
        elif task_name == "layer":
            get_pyramid_layer(input_path, output_path, **task_kwargs)
        elif task_name == "compress":
            compress_image(input_path, output_path, **task_kwargs)
        elif task_name == "project":
            from dask_image import imread
            import tifffile
            from pyimg.project import merge_channels_dask, config_from_csv, read_metadata
            
            markers_csv = task_kwargs.get("markers_csv")
            projection_method = task_kwargs.get("projection_method", "sum")
            if not markers_csv:
                raise ValueError("markers_csv is required for project benchmark")
            
            projection_config = config_from_csv(markers_csv, projection_method)
            img = imread.imread(input_path)
            while img.ndim > 3 and img.shape[0] == 1:
                img = img.squeeze(axis=0)
            
            # Rechunk to standard spatial blocks
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
            
            projected_img = merge_channels_dask(
                img,
                projection_config=projection_config,
                scale_before_project=True,
                scale_quantiles=(0.01, 0.999)
            )
            
            ome_metadata = read_metadata(input_path)
            if ome_metadata:
                ome_metadata.images[0].pixels.size_c = len(projection_config)
                ome_metadata.images[0].pixels.channels = ome_metadata.images[0].pixels.channels[:len(projection_config)]
                import ome_types
                ome_xml_bytes = ome_types.to_xml(ome_metadata).encode('utf-8')
            else:
                ome_xml_bytes = None

            import dask
            from pyimg.project import get_optimal_num_workers
            write_kwargs = {"bigtiff": True, "description": ome_xml_bytes}
            if projected_img.ndim >= 2:
                write_kwargs["tile"] = (512, 512)
                write_kwargs["compression"] = "zlib"
            
            num_workers = get_optimal_num_workers()
            logger.info(f"Using {num_workers} Dask threads based on available memory limit.")
            with dask.config.set(num_workers=num_workers):
                tifffile.imwrite(output_path, projected_img, **write_kwargs)
        else:
            raise ValueError(f"Unsupported benchmark task: {task_name}")

    except Exception as e:
        end_cpu = time.process_time()
        end_wall = time.perf_counter()
        tracker.stop()
        logger.error(f"Benchmark failed during {task_name} execution: {e}")
        raise e

    end_cpu = time.process_time()
    end_wall = time.perf_counter()
    peak_mem = tracker.stop()
    
    wall_duration = end_wall - start_wall
    cpu_duration = end_cpu - start_cpu
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    logger.info("\n" + "="*50 + "\n" +
                "           BENCHMARK PERFORMANCE REPORT           \n" +
                "="*50 + f"\n"
                f"Task Name:          {task_name}\n"
                f"Input Path:         {input_path}\n"
                f"Output Path:        {output_path}\n"
                f"Wall-clock Time:    {wall_duration:.4f} seconds\n"
                f"CPU Time:           {cpu_duration:.4f} seconds\n"
                f"Peak RAM Usage:     {peak_mem_mb:.2f} MB\n" +
                "="*50 + "\n")
    return {
        "wall_time": wall_duration,
        "cpu_time": cpu_duration,
        "peak_mem_mb": peak_mem_mb
    }

def run_benchmarking_suite(input_path: str = None, output_dir: str = None, keep_images: bool = False):
    """Runs a benchmarking suite of all tasks on the specified input image."""
    import pandas as pd
    import tifffile
    
    # 1. Resolve input path
    if not input_path:
        cwd_path = Path("test_data/TD_15_TNBC_subset.ome.tif")
        file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "TD_15_TNBC_subset.ome.tif"
        if cwd_path.exists():
            input_path = str(cwd_path.resolve())
        elif file_path.exists():
            input_path = str(file_path.resolve())
        else:
            raise FileNotFoundError(
                "Default benchmark image 'test_data/TD_15_TNBC_subset.ome.tif' not found. "
                "Please provide a valid --input path."
            )
    else:
        input_path = str(Path(input_path).resolve())
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found at: {input_path}")

    # 2. Resolve output directory
    if not output_dir:
        output_dir = str(Path(input_path).parent)
    else:
        output_dir = str(Path(output_dir).resolve())
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting benchmarking suite on: {input_path}")
    logger.info(f"Benchmark output directory: {output_dir}")

    stem = Path(input_path).name.split('.')[0]
    
    # Define output files for each task to allow immediate tracking & cleanup
    scale_out = Path(output_dir) / f"{stem}_bench_scaled.ome.tif"
    bin_out = Path(output_dir) / f"{stem}_bench_binned.ome.tif"
    pyr_out = Path(output_dir) / f"{stem}_bench_pyramid.ome.tif"
    lay_out = Path(output_dir) / f"{stem}_bench_layer.ome.tif"
    proj_out = Path(output_dir) / f"{stem}_bench_projected.ome.tif"
    comp_out = Path(output_dir) / f"{stem}_bench_compressed.ome.tif"

    temp_csv_path = None
    results = []

    try:
        # Determine number of channels in input image to build a valid markers CSV
        with tifffile.TiffFile(input_path) as tif:
            shape = tif.series[0].shape
        
        # Squeezed dimension layout logic: assume channels is first or second dimension if 3D/4D
        if len(shape) == 3:
            num_channels = shape[0]
        elif len(shape) == 4:
            num_channels = shape[1]
        else:
            num_channels = 1
            
        # Create temporary markers CSV
        rows = []
        for c in range(1, num_channels + 1):
            group_name = f"group_{1 if c <= max(1, num_channels // 2) else 2}"
            rows.append({"Channel_number": c, "projection": group_name})
        
        df_csv = pd.DataFrame(rows)
        temp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        temp_csv_path = temp_csv.name
        temp_csv.close()
        df_csv.to_csv(temp_csv_path, index=False)

        # Define tasks list
        tasks = [
            {
                "name": "scale",
                "input": input_path,
                "output": str(scale_out),
                "kwargs": {"output_dtype": "uint8"}
            },
            {
                "name": "bin",
                "input": input_path,
                "output": str(bin_out),
                "kwargs": {"binning_factor": 2}
            },
            {
                "name": "pyramid",
                "input": input_path,
                "output": str(pyr_out),
                "kwargs": {"min_size": 512}
            },
            {
                "name": "layer",
                "input": str(pyr_out),
                "output": str(lay_out),
                "kwargs": {"level": 1}
            },
            {
                "name": "project",
                "input": input_path,
                "output": str(proj_out),
                "kwargs": {"markers_csv": temp_csv_path, "projection_method": "sum"}
            },
            {
                "name": "compress",
                "input": input_path,
                "output": str(comp_out),
                "kwargs": {"compression": "lzw"}
            }
        ]

        # Execute tasks sequentially and track performance
        for task in tasks:
            # Check inputs exist
            if not os.path.exists(task["input"]):
                logger.error(f"Skipping task '{task['name']}': Input file '{task['input']}' does not exist.")
                continue

            # Dynamically adjust layer level if we are running the 'layer' task
            if task["name"] == "layer":
                try:
                    with tifffile.TiffFile(task["input"]) as tif:
                        num_levels = len(tif.series[0].levels)
                    task["kwargs"]["level"] = min(1, num_levels - 1)
                except Exception as e:
                    logger.warning(f"Could not determine pyramid levels, defaulting to level 0: {e}")
                    task["kwargs"]["level"] = 0

            metrics = run_benchmark(task["input"], task["output"], task["name"], task["kwargs"])
            results.append({
                "Task": task["name"],
                "Wall Time (s)": metrics["wall_time"],
                "CPU Time (s)": metrics["cpu_time"],
                "Peak Memory (MB)": metrics["peak_mem_mb"]
            })

            # Clean up task output immediately if no longer needed
            if not keep_images:
                if task["name"] == "scale" and scale_out.exists():
                    scale_out.unlink()
                elif task["name"] == "bin" and bin_out.exists():
                    bin_out.unlink()
                elif task["name"] == "layer":
                    if lay_out.exists():
                        lay_out.unlink()
                    if pyr_out.exists():
                        pyr_out.unlink()
                elif task["name"] == "project" and proj_out.exists():
                    proj_out.unlink()
                elif task["name"] == "compress" and comp_out.exists():
                    comp_out.unlink()

    finally:
        # Robust final cleanup in case of any failures
        if temp_csv_path and os.path.exists(temp_csv_path):
            try:
                os.unlink(temp_csv_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary markers CSV: {e}")

        if not keep_images:
            for path_to_clean in [scale_out, bin_out, pyr_out, lay_out, proj_out, comp_out]:
                if path_to_clean.exists():
                    try:
                        path_to_clean.unlink()
                    except Exception as e:
                        logger.warning(f"Could not delete temporary file {path_to_clean}: {e}")

    if not results:
        logger.error("No benchmarking tasks were completed successfully.")
        return

    # Print and save results
    df = pd.DataFrame(results)
    csv_results_path = Path(output_dir) / "benchmark_results.csv"
    df.to_csv(csv_results_path, index=False)
    logger.info(f"Saved benchmark results table to {csv_results_path}")

    print("\n" + "="*70)
    print("                     PYIMG BENCHMARK RESULTS")
    print("="*70)
    print(df.to_string(index=False, formatters={
        "Wall Time (s)": lambda x: f"{x:.2f}s",
        "CPU Time (s)": lambda x: f"{x:.2f}s",
        "Peak Memory (MB)": lambda x: f"{x:.1f} MB"
    }))
    print("="*70 + "\n")

    # Generate and save premium plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        df_timing = df.melt(id_vars="Task", value_vars=["Wall Time (s)", "CPU Time (s)"], 
                            var_name="Metric", value_name="Time (seconds)")

        palette_timing = ["#2A4B7C", "#4A90E2"]
        palette_mem = sns.color_palette("viridis", len(df))

        # 1. Timing Barplot
        sns.barplot(data=df_timing, x="Task", y="Time (seconds)", hue="Metric", ax=axes[0], palette=palette_timing)
        axes[0].set_title("Execution Time Comparison", fontsize=14, fontweight="bold", pad=15)
        axes[0].set_xlabel("Task", fontsize=12, labelpad=10)
        axes[0].set_ylabel("Time (seconds)", fontsize=12, labelpad=10)
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt="%.2fs", padding=3, fontsize=9)

        # 2. Peak RAM Barplot
        sns.barplot(data=df, x="Task", y="Peak Memory (MB)", hue="Task", ax=axes[1], palette=palette_mem, legend=False)
        axes[1].set_title("Peak RAM Footprint", fontsize=14, fontweight="bold", pad=15)
        axes[1].set_xlabel("Task", fontsize=12, labelpad=10)
        axes[1].set_ylabel("Peak RAM (MB)", fontsize=12, labelpad=10)
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt="%.1f MB", padding=3, fontsize=9)

        plt.suptitle(f"Performance Benchmark: {Path(input_path).name}", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout()

        png_results_path = Path(output_dir) / "benchmark_results.png"
        plt.savefig(png_results_path, dpi=300)
        plt.close()
        logger.info(f"Saved benchmark results plot to {png_results_path}")
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run computational benchmarking suite.")
    parser.add_argument("-i", "--input", help="Input image file. Defaults to TD_15_TNBC_subset.ome.tif.")
    parser.add_argument("-o", "--output-dir", help="Directory to save benchmark results. Defaults to the same directory as the input image.")
    parser.add_argument("--keep-images", action="store_true", help="Keep benchmark output images for human QC.")
    args = parser.parse_args()
    
    run_benchmarking_suite(input_path=args.input, output_dir=args.output_dir, keep_images=args.keep_images)
