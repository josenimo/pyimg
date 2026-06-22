import os
import pytest
import numpy as np
import tifffile
import pandas as pd
from pathlib import Path
from click.testing import CliRunner

from pyimg.cli import cli
from pyimg.utils.generate_output_paths import get_stem_and_suffixes, generate_output_paths
from pyimg.compress import check_compression, compress_image
from pyimg.get_pyramid_layer import get_pyramid_layer
from tests.helpers import create_test_image

def test_get_stem_and_suffixes():
    """Test extracting the stem and suffixes correctly including double extensions."""
    p1 = Path("image.ome.tif")
    stem, ext = get_stem_and_suffixes(p1)
    assert stem == "image"
    assert ext == ".ome.tif"

    p2 = Path("my_photo.ome.tiff")
    stem, ext = get_stem_and_suffixes(p2)
    assert stem == "my_photo"
    assert ext == ".ome.tiff"

    p3 = Path("data.tif")
    stem, ext = get_stem_and_suffixes(p3)
    assert stem == "data"
    assert ext == ".tif"

def test_generate_output_paths_directories(tmp_path):
    """Test mapping directories to output path while preserving tree structure."""
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    sub_dir = in_dir / "subdir"
    sub_dir.mkdir()

    f1 = in_dir / "img1.ome.tif"
    f2 = sub_dir / "img2.tif"

    # Create dummy files
    f1.write_text("dummy")
    f2.write_text("dummy")

    input_paths = [f1, f2]
    out_dir = tmp_path / "outputs"

    out_paths = generate_output_paths(input_paths, str(in_dir), str(out_dir), "_scaled")
    
    assert len(out_paths) == 2
    assert out_paths[0] == out_dir / "img1_scaled.ome.tif"
    assert out_paths[1] == out_dir / "subdir" / "img2_scaled.tif"
    
    # Check that directories were created
    assert (out_dir / "subdir").exists()

def test_compress_convert(tmp_path):
    """Test check_compression and compress_image."""
    img_path = tmp_path / "test_compress.tif"
    create_test_image(img_path, (100, 100), dtype=np.uint8, has_metadata=True)

    # Initial check
    schemas = check_compression(str(img_path))
    assert 0 in schemas

    # Re-compress to LZW
    out_path = tmp_path / "test_lzw.tif"
    compress_image(str(img_path), str(out_path), compression="lzw")

    assert out_path.exists()
    schemas_lzw = check_compression(str(out_path))
    assert schemas_lzw[0] == "LZW"

def test_lazy_layer_extraction(tmp_path):
    """Test get_pyramid_layer which uses lazy loading via aszarr()."""
    img_path = tmp_path / "pyramid.ome.tif"
    
    # Write a pyramidal OME-TIFF
    data1 = np.ones((100, 100), dtype='uint8')
    data2 = np.ones((50, 50), dtype='uint8')
    with tifffile.TiffWriter(img_path) as tif:
        tif.write(data1, subfiletype=0, subifds=1)
        tif.write(data2, subfiletype=1)

    out_path = tmp_path / "layer1.tif"
    get_pyramid_layer(str(img_path), str(out_path), level=1)

    assert out_path.exists()
    with tifffile.TiffFile(out_path) as tif:
        assert tif.series[0].shape == (50, 50)

def test_cli_subcommands(tmp_path):
    """Test click CLI subcommands scale, bin, pyramid, layer, project, and compress."""
    runner = CliRunner()
    
    # 1. Test Scale
    in_img = tmp_path / "cli_scale.tif"
    create_test_image(in_img, (50, 50), dtype=np.uint16)
    out_img = tmp_path / "cli_scale_out.tif"
    
    result = runner.invoke(cli, ["scale", "-i", str(in_img), "-o", str(out_img), "-d", "uint8", "-f"])
    assert result.exit_code == 0
    assert out_img.exists()
    
    # 2. Test Bin
    out_bin = tmp_path / "cli_bin_out.tif"
    result = runner.invoke(cli, ["bin", "-i", str(in_img), "-o", str(out_bin), "-b", "2", "-f"])
    assert result.exit_code == 0
    assert out_bin.exists()

    # 3. Test Pyramid
    out_pyr = tmp_path / "cli_pyr_out.tif"
    result = runner.invoke(cli, ["pyramid", "-i", str(in_img), "-o", str(out_pyr), "-m", "10", "-f"])
    assert result.exit_code == 0
    assert out_pyr.exists()

    # 4. Test Layer
    out_lay = tmp_path / "cli_lay_out.tif"
    result = runner.invoke(cli, ["layer", "-i", str(out_pyr), "-o", str(out_lay), "-l", "1", "-f"])
    assert result.exit_code == 0
    assert out_lay.exists()

    # 5. Test Compress
    out_comp = tmp_path / "cli_comp_out.tif"
    result = runner.invoke(cli, ["compress", "-i", str(in_img), "-o", str(out_comp), "-c", "lzw", "-f"])
    assert result.exit_code == 0
    assert out_comp.exists()

    # 6. Test Project
    in_proj = tmp_path / "cli_proj.tif"
    # Create 3-channel image
    create_test_image(in_proj, (3, 50, 50), dtype=np.uint16, layout="CYX")
    
    # Create markers CSV
    csv_path = tmp_path / "markers.csv"
    df = pd.DataFrame({
        "Channel_number": [1, 2, 3],
        "projection": ["groupA", "groupA", "groupB"]
    })
    df.to_csv(csv_path, index=False)
    
    out_proj = tmp_path / "cli_proj_out.tif"
    result = runner.invoke(cli, [
        "project", 
        "-i", str(in_proj), 
        "-o", str(out_proj), 
        "--markers-csv", str(csv_path), 
        "--projection-method", "sum", 
        "-f"
    ])
    assert result.exit_code == 0
    assert out_proj.exists()

    # 7. Test Benchmark
    out_bench = tmp_path / "cli_bench_out.tif"
    result = runner.invoke(cli, [
        "benchmark",
        "-i", str(in_img),
        "-o", str(out_bench),
        "--task", "scale",
        "--task-args", '{"output_dtype": "uint8"}'
    ])
    assert result.exit_code == 0
