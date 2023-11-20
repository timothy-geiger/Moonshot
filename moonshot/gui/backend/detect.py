import os
import glob
import torch

__all__ = ['detect_craters']


def detect_craters(inputPathImgs,
                   planet):
    """
    Uses trained model to detect craters in
    images. It can detect craters on the mars
    and on the moon.
    Parameters
    ----------
    inputPath: string
        Path to the directory of images.
    planet: string
        Either 'mars' or 'moon'.
    Returns
    -------
    object
        Detected craters by model.
    """

    # get weights for planet
    if planet == 'mars':
        weightsPath = './weights/mars/best.pt'

    else:
        weightsPath = './weights/moon/best.pt'

    # call model
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=weightsPath, _verbose=False)
    model.iou = 0.5

    if torch.cuda.is_available():
        model.cuda()

    # get images from input folder
    if not isinstance(inputPathImgs, list):
        inputImgPaths = glob.glob(os.path.join(inputPathImgs, '*'))

    else:
        inputImgPaths = inputPathImgs

    results = model(inputImgPaths, size=416)

    return results
