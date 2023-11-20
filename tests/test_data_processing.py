import numpy as np
import pytest

from moonshot.data_processing.processing import tesselate_im, \
    global_coords_from_lat_lon, lat_lon_from_global_coords, \
    local_coords_from_global, global_coords_from_local, \
    bounding_box_from_lat_lon, global_dims_from_local


# Test for tesselate_im function.
@pytest.mark.parametrize("im, tile_size, expected_output", [
    (np.zeros((10, 10, 3)), 2, ((5, 5, 2, 2, 3), 5, 5)),
    (np.zeros((12, 12, 3)), 2, ((6, 6, 2, 2, 3), 6, 6)),
    (np.zeros((11, 11, 3)), 3, ((3, 3, 3, 3, 3), 3, 3))])
def test_tesselate_im(im, tile_size, expected_output):
    tiles, nrows, ncols = tesselate_im(im, tile_size)
    tiles_shape_ex, nrows_ex, ncols_ex = expected_output

    # Check if tiles have the correct shape
    assert tiles_shape_ex == tiles.shape

    # Ensure the correct number of rows and columns are generated
    assert nrows == nrows_ex
    assert ncols == ncols_ex


@pytest.mark.parametrize(
    ("lon", "lat", "min_lon", "max_lon",
     "min_lat", "max_lat", "expected_output"),
    [(0, 0, -180, 180, -90, 90, (0.5, 0.5)),
     (90, 45, -180, 180, -90, 90, (0.75, 0.25)),
     (-90, -45, -180, 180, -90, 90, (0.25, 0.75)),
     (180, 90, -180, 180, -90, 90, (1.0, 0.0)),
     (-180, -90, -180, 180, -90, 90, (0.0, 1.0))])
def test_global_coords_from_lat_lon(lon, lat, min_lon, max_lon, min_lat,
                                    max_lat, expected_output):

    output = global_coords_from_lat_lon(lon, lat, min_lon, max_lon,
                                        min_lat, max_lat)
    assert output == expected_output


# The inputs for this function are just inverses of the preceeding function
@pytest.mark.parametrize(
    "X, Y, min_lon, max_lon, min_lat, max_lat, expected_output",
    [
        (0.5, 0.5, -180, 180, -90, 90, (0.0, 0.0)),
        (0.75, 0.25, -180, 180, -90, 90, (90.0, 45.0)),
        (0.25, 0.75, -180, 180, -90, 90, (-90.0, -45.0)),
        (1.0, 0.0, -180, 180, -90, 90, (180.0, 90.0)),
        (0.0, 1.0, -180, 180, -90, 90, (-180.0, -90.0)),
    ]
)
def test_lat_lon_from_global_coords(X, Y, min_lon, max_lon, min_lat, max_lat,
                                    expected_output):
    output = lat_lon_from_global_coords(X, Y, min_lon, max_lon,
                                        min_lat, max_lat)
    assert output == expected_output


@pytest.mark.parametrize(
    "X, Y, tile_size, im_nH, im_nW, expected_output",
    [
        (0.5, 0.5, 1, 3, 4, (1, 0.0, 2, 0.5)),
        (0.5, 0.5, 1, 4, 3, (2, 0.5, 1, 0.0)),
        (0.25, 0.25, 1, 5, 7, (1, 0.75, 1, 0.25)),
    ]
)
def test_local_coords_from_global(X, Y, tile_size, im_nH, im_nW,
                                  expected_output):
    output = local_coords_from_global(X, Y, tile_size, im_nH, im_nW)
    assert output == expected_output


@pytest.mark.parametrize(
    "i, x, j, y, tile_size, im_nH, im_nW, expected_output",
    [(1, 0.0, 2, 0.5, 1, 3, 4, (0.5, 0.5)),
     (2, 0.5, 1, 0.0, 1, 4, 3, (0.5, 0.5)),
     (1, 0.75, 1, 0.25, 1, 5, 7, (0.25, 0.25))])
def test_global_coords_from_local(i, x, j, y, tile_size, im_nH, im_nW,
                                  expected_output):

    assert global_coords_from_local(i, x, j, y,
                                    tile_size, im_nH, im_nW) == expected_output


@pytest.mark.parametrize("w, h, tile_size, im_nW, im_nH, expected_output", [
    (2, 3, 4, 5, 6, (1.6, 2.0)),
    (3, 2, 1, 4, 5, (0.75, 0.4))
])
def test_global_dims_from_local(w, h, tile_size, im_nW, im_nH,
                                expected_output):
    result = global_dims_from_local(w, h, tile_size, im_nW, im_nH)
    assert result == expected_output


@pytest.mark.parametrize(
    "lat, lon, diam, tile_size, m_per_pix,expected_output",
    [(0, 90, 100, 10, 100, (100.0, 100.0)),
     (0, 45, 200, 10, 100, (200.0, 200.0))])
def test_bounding_box_from_lat_lon(lat, lon, diam, tile_size, m_per_pix,
                                   expected_output):
    result = bounding_box_from_lat_lon(lat, lon, diam, tile_size, m_per_pix)
    assert result == expected_output
