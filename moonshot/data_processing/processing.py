import numpy as np
import cv2
import pandas as pd
import os

# Utility functions used at various stages of processing


def tesselate_im(im, tile_size):
    """Returns: a multi-dimensional tensor (number of vertical tiles,number of
    horizontal tiles,number of pixels per tile,number of pixels per tile,
    channels) number of vertical tiles, number of horizontal tiles. data is
    invalid if the tile size is greater than the image size # add more tests
    like that Parameters ---------- img: numpy array
        Image as a numpy array
    tile_size: int
        Size of the tile
    Returns
    -------
    numpy array,int,int
    """
    im_nH, im_nW, im_nC = im.shape

    nrows = im_nH // tile_size
    ncols = im_nW // tile_size

    shape = (nrows, ncols, tile_size, tile_size, im_nC)
    strides = (im_nW*im_nC*tile_size,
               im_nC*tile_size,
               im_nC*im_nW,
               im_nC,
               1)

    tiles = np.lib.stride_tricks.as_strided(im,
                                            shape=shape,
                                            strides=strides)

    return tiles, nrows, ncols


# Conversions between lon, lat and global coordinates X, Y
def global_coords_from_lat_lon(lon, lat, min_lon, max_lon,
                               min_lat, max_lat):
    """
    Converts lat and lon coordinates the fractional global coordinates.
    Parameters
    ----------
    lon: float
        Longitude of the crater
    lat: float
        Latitude of the crater
    min_lon: float
        Minimum Longitude of the image
    max_lon: float
        Maximum Longitude of the image
    min_lat: float
        Minimum Latitude of the image
    max_lat: float
        Maximum Latitude of the image
    Returns
    -------
    float,float
    """

    X = (lon - min_lon) / (max_lon - min_lon)
    # This conversion is slightly unorthodox due to the sign convention
    Y = (lat - max_lat) / (min_lat - max_lat)

    # X and Y are fractional
    return X, Y


def lat_lon_from_global_coords(X, Y, min_lon, max_lon,
                               min_lat, max_lat):
    """
    Converts fractional global coordinates to lat and lon coordinates.
    Parameters
    ----------
    lon: float
        Longitude of the crater
    lat: float
        Latitude of the crater
    min_lon: float
        Minimum Longitude of the image
    max_lon: float
        Maximum Longitude of the image
    min_lat: float
        Minimum Latitude of the image
    max_lat: float
        Maximum Latitude of the image
    Returns
    -------
    float
    """

    lon = X * (max_lon - min_lon) + min_lon
    # This conversion is slightly unorthodox due to the sign convention
    lat = Y * (min_lat - max_lat) + max_lat

    return lon, lat


# Conversions between local and global coordinates
def local_coords_from_global(X, Y, tile_size, im_nH, im_nW):
    """
    Converts fractional global coordinates to fractional local coordinates.
    Parameters
    ----------
    X: float
        Horizontal fractional global coordinate
    Y: float
        Vertical fractional global coordinate
    tile_size: float
        Size of the tile
    im_nH: float
        Number of pixels in the vertical axis
    im_nW: float
        Number of pixels in the horizontal axis
    Returns
    -------
    float, float
    """
    j, x = np.divmod(X*im_nW, tile_size)
    x /= tile_size
    i, y = np.divmod(Y*im_nH, tile_size)
    y /= tile_size

    return i.astype(int), x, j.astype(int), y


def global_coords_from_local(i, x, j, y, tile_size, im_nH, im_nW):
    """
    Converts fractional local coordinates to fractional global coordinates.
    Parameters
    ----------
    i: int
        tile number in the horizontal direction
    x: float
        Horizontal local coordinate
    j: float
        tile number in the horizontal direction
    y: int
        Vertical local coordinate
    tile_size: float
        Size of the tile
    im_nH: float
        Number of pixels in the vertical axis
    im_nW: float
        Number of pixels in the horizontal axis
    Returns
    -------
    float, float
    """
    X = (j + x) * tile_size/im_nW
    Y = (i + y) * tile_size/im_nH
    return X, Y


def global_dims_from_local(w, h, tile_size, im_nW, im_nH):
    """
    Returns global width and height from local (tile-specific) values
    Parameters
    ----------
    w: float
        Longitude of the crater
    h: float
        Latitude of the crater
    tile_size: float
        Size of the tile
    im_nW: float
        Number of pixels in the horizontal axis
    im_nH: float
        Number of pixels in the vertical axis
    Returns
    -------
    float, float
    """
    W = (w*tile_size)/im_nW
    H = (h*tile_size)/im_nH

    return W, H


def bounding_box_from_lat_lon(lat, lon, diam, tile_size, m_per_pix):
    """Returns bounding boxes from latitude, longitude, diameter of the crater
    and the used tile size and resolution.
    Parameters
    ----------
    lon: float
        Longitude of the crater
    lat: float
        Latitude of the crater
    diam: float
        Diameter of crater in km
    tile_size: float
        Size of the tile
    m_per_pix: float
        Length in meters per pixel
    Returns
    -------
    float, float
    """
    # input diameter is in km
    # convert to
    diam_pix = 1000 * diam/m_per_pix

    bb_height = diam_pix/tile_size
    bb_width = diam_pix/(tile_size * np.cos(np.deg2rad(lat)))

    return bb_height, bb_width


class lunarPreprocessor():
    """
    Preprocessing pipeline for training data creation for the Moon.
    """
    def __init__(self, im_path, labels_path, split_name, m_per_pix, tile_size):
        """"
        Parameters
        ----------
        im_path: string
            Path to training images directory.
        labels_path: string
            Path to lunar crater database.
        split_name: string
            Lunar image A,B,C,D
        m_per_pix: float
            Length in meters per pixel
        tile_size: float
            Size of the tile
        """

        self.im = cv2.imread(im_path)
        self.labels = pd.read_csv(labels_path, index_col=[0])

        self.im_nH, self.im_nW, self.im_nC = self.im.shape
        self.tile_size = tile_size
        self.m_per_pix = m_per_pix
        self.split_name = split_name

        range_dict = {'A': [-45.0, 0.0, -180.0, -90.0],
                      'B': [0.0, 45.0, -180.0, -90.0],
                      'C': [-45.0, 0.0, -90.0, 0.0],
                      'D': [0.0, 45.0, -90.0, 0.0]}

        range_limits = range_dict[self.split_name]

        self.min_lat = range_limits[0]
        self.max_lat = range_limits[1]
        self.min_lon = range_limits[2]
        self.max_lon = range_limits[3]

    def adjust_edge_boundary_boxes(self, coords, dims):
        """
        Adjust the boundary box parameters for boxes that span more than a
        single tile.
        Parameters
        ----------
        coords: floats
            Local fractional coordinate
        dims: floats
            Local fractional size
        Returns
        -------
        float, float
        """

        # Check positive boundary
        mask = coords + 0.5 * dims > 1.0
        delta = np.where(mask, coords + 0.5 * dims - 1.0, 0.0)
        coords -= 0.5 * delta
        dims -= delta

        # Check negative boundary
        mask = coords - 0.5 * dims < 0.0
        delta = np.where(mask, 0.5 * dims - coords, 0.0)
        coords += 0.5 * delta
        dims -= delta

        return coords, dims

    def create_train_data(self, output_dir):
        """ Construct the training dataset in the specified directory.
        Parameters
        ----------
        output_dir: string
            Path to output directory
        """

        # Construct image and label directories
        im_dir = os.path.join(output_dir, 'image')
        label_dir = os.path.join(output_dir, 'label')

        paths = [im_dir, label_dir]

        for p in paths:
            if not os.path.exists(p):
                print(f'Creating directory {p}')
                os.makedirs(p)

        # Filter out labels not corresponding to the investigated image
        lat = self.labels[['LAT_CIRC_IMG']].to_numpy()
        lon = self.labels[['LON_CIRC_IMG']].to_numpy()
        diam = self.labels[['DIAM_CIRC_IMG']].to_numpy()

        # Tesselate the image
        self.tiles, self.nrows, self.ncols = tesselate_im(self.im,
                                                          self.tile_size)

        X, Y = global_coords_from_lat_lon(lon, lat, self.min_lon, self.max_lon,
                                          self.min_lat, self.max_lat)
        i, x, j, y = local_coords_from_global(X, Y, self.tile_size,
                                              self.im_nH, self.im_nW)
        h, w = bounding_box_from_lat_lon(lat, lon, diam,
                                         self.tile_size, self.m_per_pix)

        print(x)
        # Adjust bounding boxes
        x, w = self.adjust_edge_boundary_boxes(x, w)
        y, h = self.adjust_edge_boundary_boxes(y, h)

        print('Creating training data ...')

        # Colate arrays in a dataframe
        df = pd.DataFrame({'i': i.reshape((-1,)),
                           'j': j.reshape((-1,)),
                           'x': x.reshape((-1,)),
                           'y': y.reshape((-1,)),
                           'w': w.reshape((-1,)),
                           'h': h.reshape((-1,))})
        # Filter out craters with any dimension higher than width tile
        df = df[(df['w'] < 1.0) & (df['h'] < 1.0)]

        # Group the labels using tile indices
        grouped_labels = df.groupby(['i', 'j'])

        for (i_g, j_g), group in grouped_labels:
            i_g = int(i_g)
            j_g = int(j_g)

            if (0 <= i_g and 0 <= j_g and i_g < self.nrows
                    and j_g < self.ncols):

                print(i_g, j_g)
                # Save labels
                label_name = f'Lunar_{self.split_name}_{i_g}_{j_g}'
                label_path = os.path.join(label_dir, label_name)

                group.iloc[:, 2:].to_csv(label_path, index=False, header=False)

                # Save tiles
                tile_to_save = self.tiles[i_g, j_g]

                tile_name = f'Lunar_{self.split_name}_{i_g}_{j_g}.png'
                tile_path = os.path.join(im_dir, tile_name)

                cv2.imwrite(tile_path, tile_to_save)

        print('Finished')


class lunarPostprocessor():
    '''
    Used for tiling images back together

    '''

    def __init__(self, input_dir, output_dir, output_filename,
                 tile_size, nrows, ncols, im_nW, im_nH):
        """  Parameters
        ----------
        label_dir: string
            Path to labels directory
        tile_size: float
            Size of the tile
        im_nW: float
            Number of pixels in the horizontal axis
        im_nH: float
            Number of pixels in the vertical axis
        nrows: int
            Number of vertical tiles
        ncols: int
            Number of horizontal tiles
        """

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.tile_size = tile_size
        self.nrows = nrows
        self.ncols = ncols
        self.im_nW = im_nW
        self.im_nH = im_nH

        self.nc = 3  # number of channels is hardcoded

    def load_labels(self):
        """
        Returns the csv with the global coordinates for each crater detected in
        the test image.
        Returns
        -------
        pandas DataFrame
        """
        # initialise first row with zeros - to be deleted later
        global_data = np.zeros((1, 4))

        for filename in os.listdir(self.input_dir):
            if filename.endswith(".csv"):
                # Extract i and j from the filename
                i, j = filename.split(".")[0].split("_")[-2:]
                # Convert to int
                i, j = int(i), int(j)

                filepath = os.path.join(self.input_dir, filename)
                tile_label = pd.read_csv(filepath)

                x = tile_label.iloc[:, 0].to_numpy()
                y = tile_label.iloc[:, 1].to_numpy()
                w = tile_label.iloc[:, 2].to_numpy()
                h = tile_label.iloc[:, 3].to_numpy()

                X, Y = global_coords_from_local(i, x, j, y, self.tile_size,
                                                self.im_nH, self.im_nW)
                W, H = global_dims_from_local(w, h, self.tile_size,
                                              self.im_nW, self.im_nH)

                local_data = np.column_stack((X, Y, W, H))
                global_data = np.vstack((global_data, local_data))

        # Drop first row used when initialising the global matrix
        global_data = global_data[1:, :]

        # Assign the results to a pandas dataframe
        df = pd.DataFrame(data=global_data, columns=['x', 'y', 'w', 'h'])

        df.to_csv(os.path.join(self.output_dir, self.output_filename + '.csv'))


if __name__ == "__main__":
    input_dir = os.path.join(os.getcwd(), 'test_data')  # Current dir for test

    tile_size = 416
    nrows = 32
    ncols = 65

    im_nH = 27291
    im_nW = 13645

    P = lunarPostprocessor(input_dir, tile_size, nrows, ncols, im_nW, im_nH)
    df = P.load_labels()

    print(df)
