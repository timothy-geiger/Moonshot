import os
import cv2
import glob
import pandas as pd
from PIL import Image
from moonshot.gui.backend.detect import detect_craters
from moonshot.gui.backend.visualization import generate_bounding_boxes
from moonshot.gui.backend.statistics import generate_statistics
from moonshot.data_processing.processing import (tesselate_im,
                                                 lunarPostprocessor)


__all__ = ['generate_output']


def _convert_df(df,
                imgFile):
    """
    Converts the dataframe that can be created
    with the results of the 'detectCraters'
    function (contains information about bounding
    boxes). It creates new columns width and
    height. The method changes x, y, w, h into
    fractional lengths.
    Parameters
    ----------
    df: DataFrame
        Dataframe which is to be converted.
    imgFile: string
        The path to the image file that
        corresponds to the bounding boxes.
    Returns
    -------
    DataFrame
        The converted DataFrame
    """

    # open image to get width and height
    img = Image.open(imgFile)

    # convert to fractional image width / height
    df['xmin'] = df['xmin'] / img.width
    df['xmax'] = df['xmax'] / img.width

    df['ymin'] = df['ymin'] / img.height
    df['ymax'] = df['ymax'] / img.height

    # calculate width and height of bounding box
    df['w'] = df['xmax'] - df['xmin']
    df['h'] = df['ymax'] - df['ymin']

    # rename columns
    df = df.rename(columns={'xmin': 'x', 'ymin': 'y'})

    df['x'] = df['x'] + (df['w'] / 2)
    df['y'] = df['y'] + (df['h'] / 2)

    return df[['x', 'y', 'w', 'h']]


def _identify_output(inputPath):
    """
    Checks if the input folder has the
    right format and if ground truth
    data is provided.
    Parameters
    ----------
    inputPath: string
        The path to the input directory.
    Returns
    -------
    boolean
        Whether ground truth data is provided.
    """
    inputPathImgs = os.path.join(inputPath, 'images')
    inputPathLabels = os.path.join(inputPath, 'labels')

    # check if images folder exists in input folder
    if not os.path.exists(inputPathImgs):
        raise Exception(
            'The specified input folder does not contain an "images" folder.')

    numberOfImgFiles = len(os.listdir(inputPathImgs))

    # check if input folder is empty
    if numberOfImgFiles == 0:
        raise Exception(
            'The specified input folder does contains ' +
            'an empty "images" folder.')

    # get groundTruth data if available
    hasLabels = False

    # check if labels folder exists in input folder (ground truth data exists)
    if os.path.exists(inputPathLabels):
        numberOfLabelFiles = len(os.listdir(inputPathLabels))

        # check if labels folder is not empty
        if numberOfLabelFiles != 0:

            # check if theire is the same number of input
            # images and input labels
            if numberOfImgFiles != numberOfLabelFiles:
                raise Exception(
                    'The number of images and labels does not match.')

            hasLabels = True

    return hasLabels


def _generate(inputPath, outputPath, hasLabels, planet):
    """
    Generates the output. It will create new
    folders (if necessary) in the specified
    output directory, to store the results.
    Parameters
    ----------
    inputPath: string
        The path to the input directory.
    outputPath: string
        The path to the output directory.
    hasLabels: boolean
        Whether the input folder contains ground
        truth data.
    planet: string
        The name to the planet ('mars' or 'moon').
    """

    # Output Paths
    outputPathImgs = os.path.join(outputPath, 'images')
    outputPathDetections = os.path.join(outputPath, 'detections')
    outputPathStatistics = os.path.join(outputPath, 'statistics')

    # check if folders exsists in output folder
    # if do not exist -> create directory
    # if they do exist -> delete contents

    # create folders if they do not exist
    for path in [outputPathImgs, outputPathDetections, outputPathStatistics]:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, '*'))

            for f in files:
                os.remove(f)

        else:
            os.makedirs(path)

    # add input dir to save tiled images
    # and add output dir for predictions for
    # tiled images (.csv)
    # if mars -> remove folders
    inputPathTiledImages = os.path.join(inputPath, 'tiled_images')
    outputPathTiledDetections = os.path.join(outputPath, 'tiled_detections')

    for path in [inputPathTiledImages, outputPathTiledDetections]:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, '*'))

            for f in files:

                # deleting all files in input tiles dir
                if os.path.isdir(f):
                    files_inner = glob.glob(os.path.join(f, '*'))

                    for fi in files_inner:
                        os.remove(fi)

                    os.rmdir(f)

                else:
                    os.remove(f)

            if planet == 'mars':
                os.rmdir(path)

        else:
            if planet == 'moon':
                os.makedirs(path)

    # if planet is moon, tile images in
    # input/images/ and save them under
    # input/tiled_images/. Change path to take
    # images form from input/images/ to
    # input/tiled_images/
    nrows_list = []
    ncols_list = []

    if planet == 'moon':
        # changeing paths
        inputPathImgsNotTiled = os.path.join(inputPath, 'images')
        inputPathImgs = os.path.join(inputPath, 'tiled_images')

        outputPathDetections, outputPathTiledDetections = \
            outputPathTiledDetections, outputPathDetections

        for file in os.listdir(inputPathImgsNotTiled):
            img = cv2.imread(os.path.join(inputPathImgsNotTiled, file))
            tiles, nrows, ncols = tesselate_im(img, 416)

            nrows_list.append(nrows)
            ncols_list.append(ncols)

            name, ending = os.path.splitext(file)

            # create dir to save tiled image for every
            # input image
            os.makedirs(os.path.join(inputPathImgs, name))

            for i in range(nrows):
                for j in range(ncols):
                    cv2.imwrite(
                        os.path.join(inputPathImgs,
                                     name,
                                     name+'_'+str(i)+'_'+str(j)+ending),
                        tiles[i, j])

    else:
        inputPathImgs = os.path.join(inputPath, 'images')

    if planet == 'mars':
        # get only the filename of the images (without extension)
        allImgFiles = glob.glob(os.path.join(inputPathImgs, '*'))
        imgFileBases = [os.path.basename(imgPath) for imgPath in allImgFiles]
        inputFileNames = [os.path.splitext(base)[0] for base in imgFileBases]

        # detect craters
        results = detect_craters(inputPathImgs, planet)

        for i in range(len(results.pandas().xyxy)):

            # generate csv-files
            df = results.pandas().xyxy[i]

            # --- Create CSV files with bounding boxes --- #
            # convert df into fractional values and add cols width and height
            df_converted = _convert_df(df, allImgFiles[i])
            df_converted.to_csv(os.path.join(
                outputPathDetections, inputFileNames[i] + '.csv'))

            # generate img-files
            # path to save image at
            outputPathImg = os.path.join(
                outputPathImgs,
                inputFileNames[i] + os.path.splitext(imgFileBases[i])[1])

            # input folder has non empty labels (ground truth data)
            if hasLabels:

                # --- Bounding Boxes (with ground truth data) --- #
                # path take take ground truth data
                inputPathLabel = os.path.join(
                    inputPath, 'labels', inputFileNames[i] + '.csv')

                # read csv file
                df_label = pd.read_csv(inputPathLabel, names=[
                                       'x', 'y', 'w', 'h'], header=None)

                # if header was given in csv
                if df_label['h'].dtype == 'object':
                    df_label = pd.read_csv(inputPathLabel, names=[
                                           'x', 'y', 'w', 'h'])

                # check if x column is from type string
                # -> then class idx is in csv file
                if df_label['x'].dtype == 'object':
                    dfSplitted = df_label['x'].str.split(" ", expand=True)
                    df_label['x'] = dfSplitted[1].astype('float64')

                # generate Bounding Boxes with true data
                generate_bounding_boxes(
                    allImgFiles[i], outputPathImg, df_converted, df_label)

                # --- Statistics (when ground truth data available) --- #
                #
                outputPathStatistic = os.path.join(
                    outputPathStatistics, inputFileNames[i] + '.csv')

                # generate statistics (only if ground truth data is available)
                generate_statistics(df_label, df_converted,
                                    outputPathStatistic)

            # input folder has no labels folder (ground truth data)
            else:
                # --- Bounding Boxes (no ground truth data) --- #
                # generate Bounding Boxes without true data
                generate_bounding_boxes(
                    allImgFiles[i], outputPathImg, df_converted)

    # moon input
    else:
        # iterate over orgiginal images
        for orgIdx, file in enumerate(os.listdir(inputPathImgsNotTiled)):
            name, ending = os.path.splitext(file)

            # create dir to save detections
            os.makedirs(os.path.join(outputPathDetections, name))

            # get orgiginal image size
            img = cv2.imread(os.path.join(inputPathImgsNotTiled, file))
            height, width, _ = img.shape

            allImgFiles = glob.glob(os.path.join(inputPathImgs, name, '*'))
            imgFileBases = [os.path.basename(imgPath)
                            for imgPath in allImgFiles]
            inputFileNames = [os.path.splitext(
                base)[0] for base in imgFileBases]

            chunk_size = 10

            for chunkIdx in range(0, len(allImgFiles), chunk_size):
                print(chunkIdx)
                chunk_tiled_images = allImgFiles[chunkIdx:chunkIdx+chunk_size]
                results = detect_craters(chunk_tiled_images, planet)

                for i in range(len(results.pandas().xyxy)):
                    # generate csv-files
                    df = results.pandas().xyxy[i]

                    # --- Create CSV files with bounding boxes --- #
                    # convert df into fractional values and add
                    # cols width and height
                    df_converted = _convert_df(df, allImgFiles[chunkIdx + i])
                    df_converted.to_csv(os.path.join(
                        outputPathDetections, name,
                        inputFileNames[chunkIdx + i] + '.csv'),
                        index=False)

            preprocessor = lunarPostprocessor(
                os.path.join(outputPathDetections, name),
                outputPathTiledDetections,
                name,
                416,
                nrows_list[orgIdx],
                ncols_list[orgIdx],
                width,
                height)

            preprocessor.load_labels()

            if hasLabels:
                inputPathLabel = os.path.join(
                    inputPath, 'labels', name + '.csv')

                df_label = pd.read_csv(inputPathLabel, names=[
                                       'x', 'y', 'w', 'h'], header=None)

                # if header was given in csv
                if df_label['h'].dtype == 'object':
                    df_label.columns = df_label.iloc[0]
                    df_label = df_label[1:]
                    df_label = df_label.astype('float64')

                # check if x column is from type string
                # -> then class idx is in csv file
                if df_label['x'].dtype == 'object':
                    dfSplitted = df_label['x'].str.split(" ", expand=True)
                    df_label['x'] = dfSplitted[1].astype('float64')

                df_pred = pd.read_csv(os.path.join(outputPathTiledDetections,
                                                   name + '.csv'),
                                      index_col=0)

                # output path
                outputPathStatistic = os.path.join(
                    outputPathStatistics, name + '.csv')

                # generate statistics (only if ground truth data is available)
                generate_statistics(df_label,
                                    df_pred,
                                    outputPathStatistic)


def generate_output(inputPath, outputPath, planet):
    """
    Generates the output based on the structure
    of the input folder. If the input folder
    does not contain ground truth data, only
    image files with the bounding boxes of the
    detected craters will be created as well
    as csv files with [x, y, w, h] of the bounding
    boxes in fractional values for every input
    image. If ground truth data is given, the actual
    bounding boxes will be displayed and a csv file
    with statistics (TP, FN, FP) will be created
    for every image.
    Parameters
    ----------
    inputPath: string
        The path to the input directory.
    outputPath: string
        The path to the output directory.
    planet: string
        The name to the planet ('mars' or 'moon').
    """
    hasLabels = _identify_output(inputPath)
    _generate(inputPath, outputPath, hasLabels, planet.lower())
