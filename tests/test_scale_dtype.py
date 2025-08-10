import pytest
import numpy as np
import tifffile
import ome_types

from pyimg.scale_dtype import scale_dtype
from pyimg.utils.read_metadata import read_metadata

@pytest.fixture
def test_data(tmp_path):
    """Create a temporary directory with test images for pytest."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()

    for i in range(3):
        image_path = data_dir / f"test_image_{i}.tif"
        image_data = np.random.randint(0, 1000, size=(1, 1, 3, 50, 50), dtype=np.uint16) # (T, Z, C, Y, X)
        
        ome = ome_types.model.OME()
        ome.images.append(
            ome_types.model.Image(
                id=f'Image:{i}',
                pixels=ome_types.model.Pixels(
                    id=f'Pixels:{i}',
                    dimension_order='XYCZT',
                    type='uint16',
                    size_x=50,
                    size_y=50,
                    size_c=3,
                    size_z=1,
                    size_t=1,
                    physical_size_x=0.5 + i,
                    physical_size_y=0.5 + i,
                    physical_size_x_unit='µm',
                    physical_size_y_unit='µm',
                    channels=[
                        ome_types.model.Channel(id=f'Channel:{i}:0', name=f'DAPI_{i}'),
                        ome_types.model.Channel(id=f'Channel:{i}:1', name=f'FITC_{i}'),
                        ome_types.model.Channel(id=f'Channel:{i}:2', name=f'Cy5_{i}')
                    ]
                )
            )
        )
        ome_xml = ome_types.to_xml(ome)
        tifffile.imwrite(image_path, image_data, description=ome_xml.encode('utf-8'))
    
    return data_dir, output_dir

def test_single_file_scaling(test_data):
    """Test scaling a single file to uint8."""
    data_dir, output_dir = test_data
    input_path = data_dir / "test_image_0.tif"
    output_path = output_dir / "scaled_image.tif"
    
    scale_dtype(str(input_path), str(output_path), output_dtype="uint8")

    assert output_path.exists()
    with tifffile.TiffFile(output_path) as tif:
        assert tif.series[0].dtype == np.uint8

def test_metadata_preservation(test_data):
    """Test that OME metadata is preserved after scaling."""
    data_dir, output_dir = test_data
    input_path = data_dir / "test_image_1.tif"
    output_path = output_dir / "scaled_image_meta.tif"

    scale_dtype(str(input_path), str(output_path), output_dtype="uint8")
    output_metadata = read_metadata(str(output_path))

    assert output_metadata['pixel_size'] == pytest.approx(1.5)
    assert output_metadata['channel_names'] == ['DAPI_1', 'FITC_1', 'Cy5_1']