import cv2
import pandas as pd

__all__ = ['generate_bounding_boxes', 'plot_box']


def plot_box(img, data, color, line_thickness=1, origin='center'):
    """
    Function to plot a bounding box
    an an image.
    ----------
    img: Pillow Image
        Image that just be justed.
    data: pd.Series
        Data should include: [x, y, w, h].
    color: tuple
        Tuple to set color.
    line_thickness: int
        Line thickness.
    origin: string
        The origin of the bounding box.
    """

    if origin == 'center':
        cv2.rectangle(
            img,
            (data['x']-(data['w']//2), data['y']-(data['h']//2)),
            (1+data['x']+(data['w']//2), data['y']+(data['h']//2)),
            color,
            line_thickness)

    elif origin == 'top-left':
        cv2.rectangle(
            img,
            (data['x']-(data['w']), data['y']-(data['h'])),
            (1+data['x']+(data['w']), data['y']+(data['h'])),
            color,
            line_thickness)

    else:
        raise Exception(
            'The specified origin is not yet implemented.')


def generate_bounding_boxes(inputPath,
                            outputPath,
                            df1,
                            df2=None,
                            df1IsFractional=True,
                            df2IsFractional=True):
    """
    Plots bounding boxes in an image. It
    can take two DataFrames as an input
    and plots the bounding boxes from both
    DataFrames in a different color.
    Parameters
    ----------
    inputPath: string or cv2 image
        Path to the image file.
    outputPath: string
        Path where to store the image. If None
        return cv2 image instance.
    df1: DataFrame
        DataFrame with information about
        bounding boxes.
    df2: DataFrame
        Second DataFrame with information
        about bounding boxes.
    df1IsFractional: boolean
        Whether the first DataFrame consists
        of fractional values.
    df2IsFractional: boolean
        Whether the first DataFrame consists
        of fractional values.
    """

    # read image
    if isinstance(inputPath, str):
        img = cv2.imread(inputPath)

    else:
        img = inputPath

    height, width, _ = img.shape
    df1 = df1.copy()

    # need to convert fractional to int for cv2
    if df1IsFractional:
        df1['x'] = df1['x'] * width
        df1['y'] = df1['y'] * height
        df1['w'] = df1['w'] * width
        df1['h'] = df1['h'] * height

        df1[['x', 'y', 'w', 'h']] = df1[[
            'x', 'y', 'w', 'h']].astype('int')

    # plot predicted data
    for _, row in df1.iterrows():
        plot_box(img, row, (0, 255, 0))

    # plot true data
    if isinstance(df2, pd.DataFrame):
        df2 = df2.copy()

        # need to convert fractional to int for cv2
        if df2IsFractional:
            df2['w'] = df2['w'] * width
            df2['h'] = df2['h'] * height
            df2['x'] = df2['x'] * width
            df2['y'] = df2['y'] * height
            df2[['x', 'y', 'w', 'h']] = df2[[
                'x', 'y', 'w', 'h']].astype('int')

        for _, row in df2.iterrows():
            plot_box(img, row, (255, 0, 0))

    if outputPath is None:
        return img

    cv2.imwrite(outputPath, img)
