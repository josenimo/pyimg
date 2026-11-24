"""Tile-chunked lazy access to TIFF levels.

dask-image maps each TIFF page to one dask chunk, so a whole-slide channel
plane (e.g. 48k x 87k uint16, ~8 GiB) is read in one piece. tifffile's zarr
store instead exposes the file's own tiles/strips as chunks, so reads only
touch the segments they need.
"""
import dask.array as da
import numpy as np
import tifffile
import zarr


def open_level_zarr(tif: tifffile.TiffFile, level: int = 0) -> zarr.Array:
    """Return a read-only zarr array for ``level`` of the first series.

    The array reads from ``tif``'s file handle, so ``tif`` must stay open
    while the array is used.
    """
    return zarr.open_array(tif.series[0].aszarr(level=level), mode="r")


def spatial_axes(axes: str) -> tuple[int, int]:
    """Return the indices of the Y and X axes in a tifffile axes string."""
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"Image axes '{axes}' have no Y/X spatial dimensions.")
    return axes.index("Y"), axes.index("X")


def open_level_dask(
    tif: tifffile.TiffFile, level: int = 0, target_chunk_bytes: int = 8 * 2**20
) -> da.Array:
    """Return a dask array for ``level`` whose chunks are groups of whole tiles.

    Spatial chunks are grown in multiples of the file's tile/strip size up to
    roughly ``target_chunk_bytes``; non-spatial axes keep the store's chunking.
    """
    z = open_level_zarr(tif, level)
    y_ax, x_ax = spatial_axes(tif.series[0].axes)
    chunks = list(z.chunks)
    per_pixel = np.dtype(z.dtype).itemsize * int(
        np.prod([c for i, c in enumerate(chunks) if i not in (y_ax, x_ax)])
    )
    # grow X first (up to 2048 px), then Y, each in whole source chunks
    for ax, cap in ((x_ax, 2048), (y_ax, None)):
        src = z.chunks[ax]
        other = chunks[x_ax if ax == y_ax else y_ax]
        limit = max(1, target_chunk_bytes // (per_pixel * other))
        if cap is not None:
            limit = min(limit, cap)
        chunks[ax] = min(z.shape[ax], max(src, (limit // src) * src))
    return da.from_zarr(z, chunks=tuple(chunks))
