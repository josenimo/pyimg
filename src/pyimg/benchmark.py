import time
import threading
import psutil
import os
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

            write_kwargs = {"bigtiff": True, "description": ome_xml_bytes}
            if projected_img.ndim >= 2:
                write_kwargs["tile"] = (512, 512)
                write_kwargs["compression"] = "zlib"
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
