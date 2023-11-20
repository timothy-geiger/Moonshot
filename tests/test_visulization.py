import pytest
import numpy as np
import pandas as pd

from moonshot.gui.backend import visualization

# Tesing distance function


class TestVisualization(object):

    # test valid input datatypes
    @pytest.mark.parametrize('data, line_thickness, shouldBeSame', [
        (
            pd.Series(data={'x': 0, 'y': 0, 'w': 20, 'h': 20}),
            1,
            False
        ),
        (
            pd.Series(data={'x': 0, 'y': 0, 'w': 20, 'h': 20}),
            0,
            False
        ),
        (
            pd.Series(data={'x': 0, 'y': 0, 'w': 20, 'h': 20}),
            1,
            False
        ),
        (
            pd.Series(data={'x': 0, 'y': 0, 'w': 20, 'h': 20}),
            10,
            False
        )
    ])
    def testSginleBoundingBoxPlotting(self, data,
                                      line_thickness, shouldBeSame):
        img = np.zeros((100, 256), dtype=np.uint8)

        output = visualization.plot_box(img,
                                        data,
                                        (255, 0, 0),
                                        line_thickness=line_thickness)

        assert (output == img).all() == shouldBeSame
