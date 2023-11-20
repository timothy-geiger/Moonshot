import numpy as np
import pandas as pd


__all__ = ['calculate_iou', 'get_statistics', 'generate_statistics']


def calculate_iou(trueRow, predRow):
    """
    Calculates the IOU based on two
    bounding boxes in the format [x, y, w, h]
    Parameters
    ----------
    trueRow: pd.Series
        Row of the first bounding box.
    predRow: pd.Series
        Row of the second bounding box.
    Returns
    -------
    float
        The calculatet IOU.
    """

    # determine x and y coordinates of the intersection rectangle
    xA = max(predRow['x'] - predRow['w'] / 2, trueRow['x'] - trueRow['w'] / 2)
    yA = max(predRow['y'] - predRow['h'] / 2, trueRow['y'] - trueRow['h'] / 2)
    xB = min(predRow['x'] + predRow['w'] / 2, trueRow['x'] + trueRow['w'] / 2)
    yB = min(predRow['y'] + predRow['h'] / 2, trueRow['y'] + trueRow['h'] / 2)

    # compute iou
    intersectionArea = max(0., xB - xA) * max(0., yB - yA)

    # compute the area of both the prediction and ground-truth rectangles
    predArea = predRow['w'] * predRow['h']
    trueArea = trueRow['w'] * trueRow['h']

    iou = intersectionArea / float(predArea + trueArea - intersectionArea)

    return iou


def get_statistics(trueDF, predDF, iou_thresold=0.5):
    """
    Calculates statistics (TP, FN, FP) for
    detected craters vs. true craters.
    ----------
    trueRow: pd.DataFrame
        DataFrame of the true bounding boxes.
    predRow: pd.DataFrame
        DataFrame of the detected bounding boxes.
    iou_thresold: float
        The iou threshold.
    Returns
    -------
    float
        The calculated IOU.
    """

    tp, fp, fn = 0, 0, 0

    for _, trueRow in trueDF.iterrows():
        hasFound = False

        for _, predRow in predDF.iterrows():
            iou = calculate_iou(trueRow, predRow)

            if iou >= iou_thresold:
                tp += 1
                hasFound = True
                break

        if not hasFound:
            fn += 1

    fp = len(predDF) - tp

    return tp, fp, fn


def generate_statistics(trueDF, predDF, outputPath, iou_thresold=0.5):
    """
    Calculates statistics for detected craters
    vs. true craters.
    ----------
    trueRow: pd.DataFrame
        DataFrame of the true bounding boxes.
    predRow: pd.DataFrame
        DataFrame of the detected bounding boxes.
    outputPath: string
        Path to a directory where to store the metrics.
    iou_thresold: float
        The iou threshold.
    Returns
    -------
    float
        The calculatet IOU.
    """

    # get metrics
    tp, fp, fn = get_statistics(trueDF, predDF, iou_thresold)

    # generate DF
    df = pd.DataFrame(np.array([[tp, fp, fn]]), columns=['tp', 'fp', 'fn'])

    # save DF
    df.to_csv(outputPath)
