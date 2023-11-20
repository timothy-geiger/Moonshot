import pytest
import numpy as np
import pandas as pd

from moonshot.gui.backend import statistics

# Tesing distance function


class TestStatistics(object):

    # test valid input datatypes
    @pytest.mark.parametrize('trueRow, predRow, result', [
        (
            pd.Series(data={'x': 0.1, 'y': 0.1, 'w': 0.1, 'h': 0.1}),
            pd.Series(data={'x': 0.5, 'y': 0.5, 'w': 0.1, 'h': 0.1}),
            0.0
        ),
        (
            pd.Series(data={'x': 0.1, 'y': 0.1, 'w': 0.1, 'h': 0.1}),
            pd.Series(data={'x': 0.1, 'y': 0.1, 'w': 0.1, 'h': 0.1}),
            1.0
        ),
        (
            pd.Series(data={'x': 0.1, 'y': 0.1, 'w': 0.1, 'h': 0.1}),
            pd.Series(data={'x': 0.1, 'y': 0.1, 'w': 0.1, 'h': 0.2}),
            0.5
        ),
        (
            pd.Series(data={'x': 0.1, 'y': 0.1, 'w': 0.1, 'h': 0.1}),
            pd.Series(data={'x': 0.1, 'y': 0.1, 'w': 0.2, 'h': 0.2}),
            0.25
        )
    ])
    def testCalculationOfIou(self, trueRow, predRow, result):
        output = statistics.calculate_iou(trueRow, predRow)
        assert (np.isclose(output, result)).all()

    # test valid input datatypes

    @pytest.mark.parametrize('trueRow, predRow, iou_thresold, ' +
                             'tpRes, fpRes, fnRes', [
                                 (
                                     pd.DataFrame(
                                         np.array([[0.1, 0.1, 0.1, 0.1]]),
                                         columns=['x', 'y', 'w', 'h']),
                                     pd.DataFrame(
                                         np.array([[0.1, 0.1, 0.1, 0.1]]),
                                         columns=['x', 'y', 'w', 'h']),
                                     0.5,
                                     1, 0, 0
                                 ),
                                 (
                                     pd.DataFrame(
                                         np.array([[0.1, 0.1, 0.1, 0.1]]),
                                         columns=['x', 'y', 'w', 'h']),
                                     pd.DataFrame(
                                         np.array([[0.1, 0.1, 0.1, 0.1]]),
                                         columns=['x', 'y', 'w', 'h']),
                                     0.2,
                                     1, 0, 0
                                 ),
                             ])
    def testCalculationMetrics(self, trueRow, predRow, iou_thresold,
                               tpRes, fpRes, fnRes):
        tp, fp, fn = statistics.get_statistics(trueRow, predRow, iou_thresold)

        assert (np.isclose(tp, tpRes)).all()
        assert (np.isclose(fp, fpRes)).all()
        assert (np.isclose(fn, fnRes)).all()
