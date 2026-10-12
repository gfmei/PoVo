import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image


def write_sensor_file(path, truncate=False, frames=1):
    color = io.BytesIO()
    Image.new('RGB', (4, 3), (30, 60, 90)).save(color, format='JPEG')
    depth = np.array([[0, 1000, 2500], [1200, 65535, 10]], dtype='<u2')
    color_bytes = color.getvalue()
    depth_bytes = zlib.compress(depth.tobytes())
    identity = np.eye(4, dtype='<f4').tobytes()
    sensor_name = b'StructureSensor'
    with path.open('wb') as handle:
        handle.write(struct.pack('<IQ', 4, len(sensor_name)))
        handle.write(sensor_name)
        handle.write(identity * 4)
        handle.write(struct.pack('<iiIIIIfQ', 2, 1, 4, 3, 3, 2, 1000., frames))
        for index in range(frames):
            pose = np.eye(4, dtype='<f4')
            pose[0, 3] = index
            handle.write(pose.tobytes())
            handle.write(struct.pack('<QQQQ', 10, 11, len(color_bytes), len(depth_bytes)))
            handle.write(color_bytes)
            handle.write(depth_bytes[:-1] if truncate else depth_bytes)
    return depth


def test_sensor_data_reads_and_exports_python3_binary_frames(tmp_path):
    pytest.importorskip('imageio')
    pytest.importorskip('png')
    import cv2
    from datasets.preprocess.SensorData import SensorData

    path = tmp_path / 'scene.sens'
    expected_depth = write_sensor_file(path)
    sensor = SensorData(str(path))
    assert sensor.sensor_name == 'StructureSensor'
    assert sensor.num_frames == len(sensor.frames) == 1
    frame = sensor.frames[0]
    assert isinstance(frame.color_data, bytes)
    assert isinstance(frame.depth_data, bytes)
    assert frame.decompress_color(sensor.color_compression_type).shape == (3, 4, 3)
    np.testing.assert_array_equal(
        np.frombuffer(frame.decompress_depth(sensor.depth_compression_type), dtype='<u2').reshape(2, 3),
        expected_depth,
    )
    sensor.export_depth_images(str(tmp_path / 'depth'))
    sensor.export_color_images(str(tmp_path / 'color'))
    sensor.export_poses(str(tmp_path / 'pose'))
    sensor.export_intrinsics(str(tmp_path / 'intrinsic'))
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / 'depth/0.png'), -1), expected_depth)
    np.testing.assert_array_equal(np.loadtxt(tmp_path / 'pose/0.txt'), np.eye(4))
    np.testing.assert_array_equal(np.loadtxt(tmp_path / 'intrinsic/intrinsic_depth.txt'), np.eye(4))
    assert Image.open(tmp_path / 'color/0.jpg').size == (4, 3)


def test_sensor_data_rejects_truncated_frame(tmp_path):
    pytest.importorskip('imageio')
    pytest.importorskip('png')
    from datasets.preprocess.SensorData import SensorData

    path = tmp_path / 'truncated.sens'
    write_sensor_file(path, truncate=True)
    with pytest.raises(EOFError, match='Truncated'):
        SensorData(str(path))


def test_sensor_metadata_only_load_reads_selected_payloads_exactly(tmp_path):
    from datasets.preprocess.SensorData import SensorData

    path = tmp_path / 'scene.sens'
    write_sensor_file(path, frames=2)
    eager = SensorData(str(path))
    indexed = SensorData(str(path), load_frames=False)
    assert not hasattr(indexed.frames[0], 'color_data')
    selected = indexed.read_frames([1])
    assert set(selected) == {1}
    assert selected[1].color_data == eager.frames[1].color_data
    assert selected[1].depth_data == eager.frames[1].depth_data
    np.testing.assert_array_equal(selected[1].camera_to_world, eager.frames[1].camera_to_world)
    assert indexed.read_frames([]) == {}
    with pytest.raises(ValueError, match='outside'):
        indexed.read_frames([2])


def test_sensor_metadata_only_load_rejects_truncated_payload(tmp_path):
    from datasets.preprocess.SensorData import SensorData

    path = tmp_path / 'truncated.sens'
    write_sensor_file(path, truncate=True)
    with pytest.raises(EOFError, match='Truncated'):
        SensorData(str(path), load_frames=False)
