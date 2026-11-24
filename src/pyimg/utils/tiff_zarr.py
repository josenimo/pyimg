"""Tile-chunked lazy access to TIFF levels.

dask-image maps each TIFF page to one dask chunk, so a whole-slide channel
plane (e.g. 48k x 87k uint16, ~8 GiB) is read in one piece. tifffile's zarr
store instead exposes the file's own tiles/strips as chunks, so reads only
touch the segments they need.
"""
import itertools

import dask.array as da
import numpy as np
import tifffile
import zarr
from loguru import logger

TILE_SIZE = 512
# caps tiles held by tifffile's encoder thread pool (its default is 512 MiB)
WRITE_BUFFER_BYTES = 64 * 2**20


def check_level(tif: tifffile.TiffFile, level: int) -> None:
    """Raise ValueError if ``level`` is not a pyramid level of the first series."""
    n_levels = len(tif.series[0].levels)
    if not 0 <= level < n_levels:
        raise ValueError(
            f"Cannot read level {level}: image has {n_levels} level(s) (0-{n_levels - 1})."
        )


def open_level_zarr(tif: tifffile.TiffFile, level: int = 0) -> zarr.Array:
    """Return a read-only zarr array for ``level`` of the first series.

    The array reads from ``tif``'s file handle, so ``tif`` must stay open
    while the array is used.
    """
    check_level(tif, level)
    # series.aszarr(level=...) yields an array; series.levels[i].aszarr() on a
    # multi-level series yields a zarr group for i == 0
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


def write_level_tiled(
    tif: tifffile.TiffFile,
    output_path,
    *,
    level: int = 0,
    compression=None,
    description: bytes | None = None,
) -> None:
    """Write one level of ``tif``'s first series as a tiled (Big)TIFF.

    Tiles are streamed from the source's own tiles/strips, so memory stays at a
    few tiles regardless of plane size. ``description`` (e.g. OME-XML) is
    written as the only ImageDescription.
    """
    source = open_level_zarr(tif, level)
    axes = tif.series[0].axes
    y_ax, x_ax = spatial_axes(axes)
    rgb = axes.endswith("YXS")
    if not (rgb or axes.endswith("YX")):
        raise ValueError(f"Unsupported axes '{axes}': expected the last axes to be YX or YXS.")
    leading = source.shape[:y_ax]
    size_y, size_x = source.shape[y_ax], source.shape[x_ax]
    logger.info(f"Source level {level}: axes {axes}, shape {source.shape}, chunks {source.chunks}")

    def tiles():
        for idx in itertools.product(*(range(n) for n in leading)):
            for y in range(0, size_y, TILE_SIZE):
                for x in range(0, size_x, TILE_SIZE):
                    yield source[idx + (slice(y, y + TILE_SIZE), slice(x, x + TILE_SIZE))]

    tifffile.imwrite(
        output_path,
        tiles(),
        shape=source.shape,
        dtype=source.dtype,
        photometric="rgb" if rgb else "minisblack",
        tile=(TILE_SIZE, TILE_SIZE),
        compression=compression,
        bigtiff=True,
        description=description,
        # no tifffile "shaped" JSON description: it would shadow the OME-XML
        metadata=None,
        buffersize=WRITE_BUFFER_BYTES,
    )
