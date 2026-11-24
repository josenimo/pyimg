"""Interoperability checks between pyimg outputs and spatialdata.

pyimg monkeypatches tifffile's zarr store on import, so these tests make sure
that patch coexists with spatialdata's own zarr I/O and that pyimg-produced
OME-TIFFs can be loaded into a SpatialData object and round-tripped to disk.
"""
import dask.array as da
import numpy as np
import pytest
import tifffile

import pyimg  # noqa: F401  (applies tifffile.zarr monkeypatch before spatialdata loads)
from pyimg.pyramidize import pyramidize

sd = pytest.importorskip("spatialdata", minversion="0.9.0")
from spatialdata.models import Image2DModel  # noqa: E402


def _load_levels(path):
    """Lazily load every pyramid level the same way pyimg does (aszarr -> dask)."""
    with tifffile.TiffFile(path) as tif:
        n_levels = len(tif.series[0].levels)
    levels = []
    for i in range(n_levels):
        store = tifffile.imread(path, aszarr=True, level=i)
        levels.append(da.from_zarr(store))
    return levels


def test_pyramidize_output_roundtrips_through_spatialdata(tmp_path):
    channel_names = ["DAPI", "CD3", "CD20"]
    # One page per channel (minisblack), as written by microscopes / bioformats.
    src = tmp_path / "src.ome.tif"
    data = np.random.default_rng(0).integers(0, 1000, (3, 1200, 1200), dtype=np.uint16)
    tifffile.imwrite(
        src,
        data,
        photometric="minisblack",
        ome=True,
        metadata={
            "axes": "CYX",
            "Channel": {"Name": channel_names},
            "PhysicalSizeX": 0.5,
            "PhysicalSizeY": 0.5,
        },
    )
    pyramid_path = tmp_path / "src_pyramid.ome.tif"
    pyramidize(str(src), str(pyramid_path), min_size=200)

    levels = _load_levels(pyramid_path)
    assert [lvl.shape for lvl in levels] == [(3, 1200, 1200), (3, 400, 400), (3, 133, 133)]

    image = Image2DModel.parse(
        levels[0],
        dims=("c", "y", "x"),
        c_coords=channel_names,
        scale_factors=[3, 3],
        chunks=(1, 256, 256),
    )
    sdata = sd.SpatialData(images={"pyimg_image": image})

    zarr_path = tmp_path / "pyimg.zarr"
    sdata.write(zarr_path)
    reread = sd.read_zarr(zarr_path)

    node = reread.images["pyimg_image"]
    scales = sorted(node.keys())
    assert len(scales) == 3
    full_res = node[scales[0]]["image"]
    assert list(full_res.coords["c"].values) == channel_names
    assert full_res.dtype == np.uint16
    np.testing.assert_array_equal(full_res.data.compute(), levels[0].compute())
    np.testing.assert_array_equal(full_res.data.compute(), data)
