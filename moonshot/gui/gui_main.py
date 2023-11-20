import sys
import os
import cv2
import glob
import matplotlib.pyplot as plt
import random
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QGridLayout, QLineEdit,
                             QPushButton, QLabel, QGroupBox,
                             QComboBox, QTableWidget, QFileDialog,
                             QTableWidgetItem, QApplication,
                             QTabWidget, QMainWindow)
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QPixmap, QPalette, QColor, QImage
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas


from moonshot.gui.backend.output import generate_output
from moonshot.gui.backend.visualization import generate_bounding_boxes


class Predictor(QThread):
    """
    A class that handels the detection. It inherits
    from "QThread" in order to let it run in the background.

    Attributes
    ----------
    input_path : str
        The path where to get the data from (input folder).
    output_path : str
        The path where to store the data.
    planet : str
        The planet from the input image data.

    Methods
    -------
    run():
        Runs the detection.
    """

    def __init__(self, input_path, output_path, planet):
        super().__init__()
        self._input_path = input_path
        self._output_path = output_path
        self._planet = planet

    def run(self):
        generate_output(self._input_path, self._output_path, self._planet)


class MainWindow(QMainWindow):
    """
    The class for the GUI.

    Methods
    -------
    browse_folder(path, type):
        Opens a window so the user can select a folder.
    change_img(type):
        Changes the image when the controlls are used.
    display_image():
        Displays an image on the input and the
        corresponding detected image on the output tab
        if a prediction has been done.
    search_by_title():
        Search the image folder for a file.
    change_planet_selection():
        Method gets executed when user changes
        the planet. When mars is selected it will
        deactivate the optional input.
    predict_image():
        Call the Predictor class to make the
        detection in the background.
    updateBBTables():
        Updated the table that shows the bounding
        boxes informations.
    updateStatistics():
        Updates the frequency-size plot and the
        count of TP, FN and FP.
    freq_size_plot(path):
        Plots the frequency-size plot.
    show_predictions():
        When the program is done with the detection
        process, the images, tables and statistics
        must be updated.,
    """

    def __init__(self):
        super().__init__()

        # Params
        self.input_folder = ""
        self.output_folder = ""
        self.current_image_index = 0
        self.images = []
        self.did_prediction = False
        self.gt_available = False

        # Create a tab widget
        self.tabs = QTabWidget(self)

        # Create 3 tabs
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()
        tab4 = QWidget()

        # Set the tabs as the central widget
        self.setCentralWidget(self.tabs)

        # Add tabs to the tab widget
        self.tabs.addTab(tab1, "Input")
        self.tabs.addTab(tab2, "Output")
        self.tabs.addTab(tab3, "Statistics")
        self.tabs.addTab(tab4, "Help")

        # Create a grid layout for each tab
        self.grid_tab1 = QGridLayout(tab1)
        self.grid_tab2 = QGridLayout(tab2)
        self.grid_tab3 = QGridLayout(tab3)
        self.grid_tab4 = QGridLayout(tab4)

        # ---------------------- INPUT TAB ---------------------- #
        # ----------- Required inputs -----------

        self.required_box = QGroupBox("Required inputs", tab1)
        # Assign layout to group box
        self.required_box_layout = QGridLayout()

        # Add entries to the group box
        # Enter input directory
        self.input_path_entry = QLineEdit(self.required_box)
        self.required_box_layout.addWidget(self.input_path_entry, 4, 2)
        self.browse_button_input = QPushButton('...', self)
        self.browse_button_input.clicked.connect(
            lambda: self.browse_folder(self.input_path_entry, 'Input'))

        self.required_box_layout.addWidget(self.browse_button_input, 4, 3)
        self.browse_label = QLabel('Browse input folder: ', self)
        self.required_box_layout.addWidget(self.browse_label, 4, 0)

        # Add entries to the group box
        # Enter output directory
        self.output_path_entry = QLineEdit(self.required_box)
        self.required_box_layout.addWidget(self.output_path_entry, 5, 2)
        self.browse_button_output = QPushButton('...', self)
        self.browse_button_output.clicked.connect(
            lambda: self.browse_folder(self.output_path_entry, 'Output'))

        self.required_box_layout.addWidget(self.browse_button_output, 5, 3)
        self.browse_label = QLabel('Browse output folder: ', self)
        self.required_box_layout.addWidget(self.browse_label, 5, 0)

        # Dropdown for planet selection
        self.planet_dropdown = QComboBox(self)
        self.planet_dropdown.addItem("Moon")
        self.planet_dropdown.addItem("Mars")
        self.planet_dropdown.activated.connect(self.change_planet_selection)
        self.required_box_layout.addWidget(self.planet_dropdown, 6, 2)

        # Label for planet selection
        self.planet_label = QLabel('Select planet: ', self)
        self.required_box_layout.addWidget(self.planet_label, 6, 0)
        # Assign the group box layout to the group
        self.required_box.setLayout(self.required_box_layout)
        # Add the required input groupbox to tab1
        self.grid_tab1.addWidget(self.required_box, 0, 0, 10, 10)

        # ----------- Optional inputs ----------- #

        self.optional_box = QGroupBox("Optional inputs", tab1)
        # Assign layout to group box
        self.optional_box_layout = QGridLayout()

        # Label for real image size
        self.image_size_label = QLabel('Image size: ', self)
        self.optional_box_layout.addWidget(self.image_size_label, 6, 0)

        # Width entry
        self.width_entry = QLineEdit(self.optional_box)
        self.width_entry.setPlaceholderText("Width")
        self.optional_box_layout.addWidget(self.width_entry, 6, 2)

        # Height entry
        self.height_entry = QLineEdit(self.optional_box)
        self.height_entry.setPlaceholderText("Height")
        self.optional_box_layout.addWidget(self.height_entry, 6, 3)

        # Resolution label
        self.resolution_label = QLabel('Resolution [m/px]: ', self)
        self.optional_box_layout.addWidget(self.resolution_label, 6, 4)

        # Resolution entry
        self.resolution_entry = QLineEdit(self.optional_box)
        # self.resolution_entry.setPlaceholderText("100")
        self.optional_box_layout.addWidget(self.resolution_entry, 6, 5)

        # Planet radius label
        self.planet_radius_label = QLabel('Planet radius [km]: ', self)
        self.optional_box_layout.addWidget(self.planet_radius_label, 8, 4)

        # Resolution entry
        self.planet_radius_entry = QLineEdit(self.optional_box)
        # self._planet_radius_entry.setPlaceholderText("2137")
        self.optional_box_layout.addWidget(self.planet_radius_entry, 8, 5)

        # Label for real image location
        self.image_loc_label = QLabel('Image location: ', self)
        self.optional_box_layout.addWidget(self.image_loc_label, 8, 0)

        # Longitude entry
        self.longitude_entry = QLineEdit(self.optional_box)
        self.longitude_entry.setPlaceholderText("Longitude")
        self.optional_box_layout.addWidget(self.longitude_entry, 8, 2)

        # Latitude entry
        self.latitude_entry = QLineEdit(self.optional_box)
        self.latitude_entry.setPlaceholderText("Latitude")
        self.optional_box_layout.addWidget(self.latitude_entry, 8, 3)

        # Assign the group box layout to the group
        self.optional_box.setLayout(self.optional_box_layout)
        # Add the required input groupbox to tab1
        self.grid_tab1.addWidget(self.optional_box, 11, 0, 10, 10)

        # ----------- Input image preview ----------- #

        self.preview_box = QGroupBox("Preview", tab1)
        # Assign layout to group box
        self.preview_box_layout = QGridLayout()

        # Image preview
        self.preview_image = QLabel(self)
        empty_pixmap = QPixmap(400, 400)
        empty_pixmap.fill(Qt.transparent)

        self.preview_image.setPixmap(empty_pixmap)
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_box_layout.addWidget(self.preview_image, 0, 0, 3, 3)

        # Title search
        self.search_text = QLineEdit(self)
        self.search_text.setPlaceholderText("Search by title")
        self.preview_box_layout.addWidget(self.search_text, 5, 0)

        # Search button
        self.search_button = QPushButton('Search', self)
        self.search_button.clicked.connect(self.search_by_title)
        self.preview_box_layout.addWidget(self.search_button, 5, 2)

        # adding new layout for buttons
        self.preview_btn_controls_i = QGridLayout()
        self.preview_box_layout.addLayout(
            self.preview_btn_controls_i, 6, 0, 1, 3)

        # Prev Image
        self.prev_button = QPushButton('Previous image', self)
        self.prev_button.clicked.connect(lambda: self.change_img('prev'))
        self.preview_btn_controls_i.addWidget(self.prev_button, 0, 0, 1, 1)

        # Random image button
        self.random_button = QPushButton('Random image', self)
        self.random_button.clicked.connect(lambda: self.change_img('random'))
        self.preview_btn_controls_i.addWidget(self.random_button, 0, 1, 1, 3)

        # Prev Image
        self.next_button = QPushButton('Next image', self)
        self.next_button.clicked.connect(lambda: self.change_img('next'))
        self.preview_btn_controls_i.addWidget(self.next_button, 0, 4, 1, 1)

        # Predict button
        self.predict_button = QPushButton('Predict', self)
        self.predict_button.clicked.connect(self.predict_image)
        self.preview_box_layout.addWidget(self.predict_button, 7, 0, 1, 3)

        # Search result
        self.search_result_label = QLabel(self)
        self.preview_box_layout.addWidget(self.search_result_label, 8, 0)

        # Assign the group box layout to the group
        self.preview_box.setLayout(self.preview_box_layout)
        # Add the required input groupbox to tab1
        self.grid_tab1.addWidget(self.preview_box, 21, 0, 20, 10)

        # ---------------------- OUTPUT TAB ---------------------- #
        # ----------- Detection outputs ----------- #

        self.detection_box = QGroupBox("Real-World Locations", tab2)
        # Assign layout to group box
        self.detection_box_layout = QGridLayout()

        # Label for real-world locations
        self.physical_location_label = QLabel('Physical Locations: ', self)
        self.detection_box_layout.addWidget(self.physical_location_label, 4, 0)

        # Latitude entry
        self.latitude_entry = QLineEdit(self.detection_box)
        self.latitude_entry.setPlaceholderText("Latitude")
        self.detection_box_layout.addWidget(self.latitude_entry, 4, 1)

        # Longtitude entry
        self.longtitude_entry = QLineEdit(self.detection_box)
        self.longtitude_entry.setPlaceholderText("Longtitude")
        self.detection_box_layout.addWidget(self.longtitude_entry, 4, 2)

        # Label for size of crater
        self.size_of_crater_label = QLabel('Size of Crater [km]: ', self)
        self.detection_box_layout.addWidget(self.size_of_crater_label, 4, 3)

        # Size entry
        self.size_entry = QLineEdit(self.detection_box)
        self.size_entry.setPlaceholderText("Size")
        self.detection_box_layout.addWidget(self.size_entry, 4, 5)

        # Assign the group box layout to the group
        self.detection_box.setLayout(self.detection_box_layout)
        # Add the required output groupbox to tab2
        self.grid_tab2.addWidget(self.detection_box, 0, 0, 2, 10)

        # ----------- Output images preview ----------- #

        self.annotated_box = QGroupBox("Annotated Image", tab2)
        # Assign layout to group box
        self.annotated_box_layout = QGridLayout()

        # Image preview
        self.annotated_image = QLabel(self)
        empty_pixmap = QPixmap(400, 400)
        empty_pixmap.fill(Qt.transparent)

        self.annotated_image.setPixmap(empty_pixmap)
        self.annotated_image.setAlignment(Qt.AlignCenter)
        self.annotated_box_layout.addWidget(self.annotated_image, 0, 0, 3, 3)

        # adding new layout for buttons
        self.preview_btn_controls_o = QGridLayout()
        self.annotated_box_layout.addLayout(
            self.preview_btn_controls_o, 4, 0, 5, 3, Qt.AlignBottom)

        # Prev Image
        self.prev_button = QPushButton('Previous image', self)
        self.prev_button.clicked.connect(lambda: self.change_img('prev'))
        self.preview_btn_controls_o.addWidget(self.prev_button, 0, 0, 1, 1)

        # Random image button
        self.random_button = QPushButton('Random image', self)
        self.random_button.clicked.connect(lambda: self.change_img('random'))
        self.preview_btn_controls_o.addWidget(self.random_button, 0, 1, 1, 3)

        # Prev Image
        self.next_button = QPushButton('Next image', self)
        self.next_button.clicked.connect(lambda: self.change_img('next'))
        self.preview_btn_controls_o.addWidget(self.next_button, 0, 4, 1, 1)

        # Assign the group box layout to the group
        self.annotated_box.setLayout(self.annotated_box_layout)
        # Add the required output groupbox to tab2
        self.grid_tab2.addWidget(self.annotated_box, 5, 0, 20, 10)

        # ----------- Bounding boxes for CDM ----------- #

        self.bounding_box1 = QGroupBox("Bounding-Boxes for CDM", tab2)
        # Assign layout to group box
        self.bounding_box1_layout = QGridLayout()

        # Table for bounding boxes of CDM
        self.CDM_table = QTableWidget(0, 4)
        self.CDM_table.setHorizontalHeaderLabels(['Horizontal location',
                                                  'Vertical location',
                                                  'Width', 'Height'])
        self.header = self.CDM_table.horizontalHeader()
        self.font = self.header.font()
        self.font.setPointSize(12)
        self.header.setFont(self.font)
        self.bounding_box1_layout.addWidget(self.CDM_table, 0, 0)

        # Assign the group box layout to the group
        self.bounding_box1.setLayout(self.bounding_box1_layout)
        # Add the required output groupbox to tab2
        self.grid_tab2.addWidget(self.bounding_box1, 25, 0, 10, 10)

        # ----------- Bounding boxes for ground truth ----------- #

        self.bounding_box2 = QGroupBox("Bounding-Boxes for Ground Truth", tab2)
        # Assign layout to group box
        self.bounding_box2_layout = QGridLayout()

        # Table for bounding boxes of ground truth
        self.GT_table = QTableWidget(0, 4)
        self.GT_table.setHorizontalHeaderLabels(['Horizontal location',
                                                 'Vertical location',
                                                 'Width', 'Height'])
        self.header = self.GT_table.horizontalHeader()
        self.font = self.header.font()
        self.font.setPointSize(12)
        self.header.setFont(self.font)
        self.bounding_box2_layout.addWidget(self.GT_table, 3, 0)

        # Assign the group box layout to the group
        self.bounding_box2.setLayout(self.bounding_box2_layout)
        # Add the required output groupbox to tab2
        self.grid_tab2.addWidget(self.bounding_box2, 25, 5, 10, 5)
        self.bounding_box2.hide()

        # ---------------------- Statistics TAB ---------------------- #

        self.distribution_box = QGroupBox("Frequency Distribution", tab3)
        # Assign layout to group box
        self.distribution_box_layout = QGridLayout()

        self.figure = plt.figure()
        self.figure.patch.set_facecolor('None')
        self.size_freq_canvas = FigureCanvas(self.figure)
        self.size_freq_canvas.setStyleSheet("background-color:transparent;")
        self.distribution_box_layout.addWidget(
            self.size_freq_canvas, 0, 0, 3, 3)

        # Assign the group box layout to the group
        self.distribution_box.setLayout(self.distribution_box_layout)
        # Add the required statistics groupbox to tab3
        self.grid_tab3.addWidget(self.distribution_box, 0, 0, 25, 10)

        self.matrix_box = QGroupBox("Confusion Matrix", tab3)
        # Assign layout to group box
        self.matrix_box_layout = QGridLayout()

        # Label for TP
        self.tp_label = QLabel('Number of TP: ', self)
        self.matrix_box_layout.addWidget(self.tp_label, 12, 0)

        # Number of TP entry
        self.TP_entry = QLineEdit(self.matrix_box)
        self.TP_entry.setPlaceholderText("")
        self.matrix_box_layout.addWidget(self.TP_entry, 13, 0)

        # Label for FP
        self.fp_label = QLabel('Number of FP: ', self)
        self.matrix_box_layout.addWidget(self.fp_label, 12, 2)

        # Number of FP entry
        self.FP_entry = QLineEdit(self.matrix_box)
        self.FP_entry.setPlaceholderText("")
        self.matrix_box_layout.addWidget(self.FP_entry, 13, 2)

        # Label for FN
        self.fn_label = QLabel('Number of FN: ', self)
        self.matrix_box_layout.addWidget(self.fn_label, 12, 4)

        # Number of FN entry
        self.FN_entry = QLineEdit(self.matrix_box)
        self.FN_entry.setPlaceholderText("")
        self.matrix_box_layout.addWidget(self.FN_entry, 13, 4)

        # Assign the group box layout to the group
        self.matrix_box.setLayout(self.matrix_box_layout)
        # Add the required statistics groupbox to tab3
        self.grid_tab3.addWidget(self.matrix_box, 25, 0, 2, 10)

        # General settings
        # self.setGeometry(500, 500, 500, 500)
        self.setWindowTitle('Automatic Crater Detection')
        self.show()

        # ---------------------- HELP TAB ---------------------- #
        self.help_box = QGroupBox("Help", tab4)
        # Assign layout to group box
        self.help_box_layout = QGridLayout()

        # Add text
        self.help_text = QLabel()
        self.help_text.setWordWrap(True)
        self.help_text.setText('This is a tool which is aimed to ' +
                               'automatically detect impact craters ' +
                               'in images of planetary surfaces and ' +
                               'deriving from this a crater-size ' +
                               'frequency distribution. There are ' +
                               'three main ' +
                               'parts for this tool, "Input", "Output" and ' +
                               '"Statistics".<br><br>In the input part, the ' +
                               'user has to select an input folder and ' +
                               'an output folder. The Input folder must ' +
                               'contain a subfolder called "images". This ' +
                               'folder should only contain image files. ' +
                               'Optionally the user can provide ground ' +
                               'truth data in another subfolder called ' +
                               '"labels" inside the input folder. ' +
                               'If the folder is selected, a preview of ' +
                               'a random image inside the ' +
                               '"input_folder/images/" will show up in the ' +
                               'preview window. With the controlls under ' +
                               'the preview, the user can navigate through ' +
                               'different images. ' +
                               'Then the user must choose an output folder. ' +
                               'After that the planet for which the craters ' +
                               'should be detected can be changed. ' +
                               'There are two planet you can choose, ' +
                               'Mars and Moon. If you choose ' +
                               'the Moon to predict, you can type the ' +
                               'optional input part which will give you ' +
                               'the real-world locations in output part. ' +
                               '<br><br>' +
                               'In the output part, you can see the ' +
                               'annotated image after predicting. There ' +
                               'are also some location and size ' +
                               'information for the bounding boxes below ' +
                               'the image. If the ground truth is provided, ' +
                               'there will be two types of bounding boxes, ' +
                               'the red is for ground truth and the green ' +
                               'is for CDM. If the user changes the image ' +
                               'in the preview window (on the output tab) ' +
                               ', the image on the input tab will change ' +
                               'accordingly.<br><br>' +
                               'In the statistics part, we can ' +
                               'see the cumulative crater size-frequency ' +
                               'distribution of detected craters and the ' +
                               'number of True Positive, False Negative ' +
                               'and False Positive detections.')
        self.help_box_layout.addWidget(
            self.help_text, 0, 0, 25, 10, Qt.AlignTop)

        # Assign the group box layout to the group
        self.help_box.setLayout(self.help_box_layout)
        # Add the required statistics groupbox to tab3
        self.grid_tab4.addWidget(self.help_box, 0, 0, 25, 10)

    # Methods

    def browse_folder(self, path, type):
        """
        Opens a dialog so the user can
        chose a folder.
        Parameters
        ----------
        path: string
            Text input where the selected
            path should be displayed.
        type: string
            'Input' or 'Output'.
        """

        options = QFileDialog.Options()
        # options |= QFileDialog.ReadOnly
        folder = QFileDialog.getExistingDirectory(
            self, "Select " + type + " Folder", "", options=options)

        if folder != '':
            if type == 'Input':
                self.input_folder = folder

                # check if images folder exists
                if os.path.exists(os.path.join(self.input_folder, 'images')):
                    path.setText(folder)
                    self.search_result_label.setText('')

                    self.images = os.listdir(
                        os.path.join(self.input_folder, 'images'))

                    # Once the folder is selected, display the first picture
                    self.display_image()

                else:
                    palette = QPalette()
                    palette.setColor(QPalette.WindowText, QColor('red'))
                    self.search_result_label.setPalette(palette)
                    self.search_result_label.setText(
                        'Input folder does not contain an "images" folder.')

            else:
                self.output_folder = folder
                path.setText(folder)

    def change_img(self, type):
        """
        Changes the image in the input
        and output tab, thought the
        controlls under the image.
        Parameters
        ----------
        type: string
            Either 'random', 'next' or
            'prev'.
        """

        if len(self.images) != 0:
            if type == 'random':
                self.current_image_index = random.randint(
                    0, len(self.images) - 1)

            elif (type == 'next'):
                self.current_image_index = (
                    self.current_image_index + 1) % len(self.images)

            elif (type == 'prev'):
                self.current_image_index -= 1

                if self.current_image_index < 0:
                    self.current_image_index = len(self.images) - 1

            self.display_image()

            if self.did_prediction:
                self.updateBBTables()

            self.search_result_label.setText('')

        else:
            palette = QPalette()
            palette.setColor(QPalette.WindowText, QColor('red'))
            self.search_result_label.setPalette(palette)
            self.search_result_label.setText('Please specify an input folder.')

    def display_image(self):
        """
        Displays the current image to both
        preview screens.
        """
        image_file_path = os.path.join(
            self.input_folder, 'images', self.images[self.current_image_index])

        pixmap = QPixmap(image_file_path)
        pixmapScaled = pixmap.scaledToHeight(400)
        self.preview_image.setPixmap(pixmapScaled)

        if self.did_prediction:
            # we are not using the image in the output
            # directory, because we are compressing the
            # image in the GUI to display it always in
            # the same size. However when compressing
            # the image, theire is a possibility that
            # the bounding boxes are not displayed
            # correctly
            dataFileName = os.path.splitext(
                self.images[self.current_image_index])[0] + '.csv'
            predPathLabel = os.path.join(
                self.output_folder, 'detections', dataFileName)

            pred_df = pd.read_csv(predPathLabel, usecols=['x', 'y', 'w', 'h'])

            cv2Img = cv2.imread(image_file_path)
            height, width, _ = cv2Img.shape

            ratio = 400 / height
            cv2ImgScaled = cv2.resize(cv2Img, (int(width * ratio), 400))

            if self.gt_available:
                inputPathLabel = os.path.join(
                    self.input_folder, 'labels', dataFileName)

                # read csv file
                df_label = pd.read_csv(inputPathLabel, names=[
                                       'x', 'y', 'w', 'h'])

                # check if x column is from type string
                # -> then class idx is in csv file
                if df_label['x'].dtype == 'object':
                    dfSplitted = df_label['x'].str.split(" ", expand=True)
                    df_label['x'] = dfSplitted[1].astype('float64')

                cvImg = generate_bounding_boxes(
                    cv2ImgScaled, None, pred_df, df_label)

            else:
                cvImg = generate_bounding_boxes(cv2ImgScaled, None, pred_df)

            # convert cv2 image, so it can
            # be displayed on the GUI
            height, width, channels = cvImg.shape
            bytesPerLine = channels * width
            qImg = QImage(cvImg.data, width, height,
                          bytesPerLine, QImage.Format_RGB888)

            # displaying image
            pixmap = QPixmap.fromImage(qImg)
            self.annotated_image.setPixmap(pixmap)

    def search_by_title(self):
        """
        Uses the title in the text
        box to search for an image
        in the "input/images/" folder.
        When the image exists, show it
        to the user. Otherwise show an
        error message.
        """

        palette = QPalette()
        try:
            title = self.search_text.text()
            title = title + '.png'
            self.current_image_index = self.images.index(title)
            # Set color to green and print the text
            palette.setColor(QPalette.WindowText, QColor('green'))
            self.search_result_label.setPalette(palette)
            self.search_result_label.setText('Displaying search image.')

            # Displaying the image
            self.display_image()

        except ValueError:
            # Set color to red and print the text
            palette.setColor(QPalette.WindowText, QColor('red'))
            self.search_result_label.setPalette(palette)
            self.search_result_label.setText(
                'Invalid search query. Try again.')

    def change_planet_selection(self):
        """
        Method gets called, when the user
        selects a different planet. If planet
        is 'mars', the optional input has
        to be deactivated.
        """
        if self.planet_dropdown.currentText() == 'Mars':
            disable = False

        else:
            disable = True

        # disable optional input when mars is selected
        self.width_entry.setEnabled(disable)
        self.height_entry.setEnabled(disable)
        self.resolution_entry.setEnabled(disable)
        self.planet_radius_entry.setEnabled(disable)
        self.longitude_entry.setEnabled(disable)
        self.latitude_entry.setEnabled(disable)

    def predict_image(self):
        """
        Calls the backend to do the detections.
        Additionally it checks for ground truth
        data. If theire is ground truth data
        show 2 tables in the output  tab.
        """

        input_folder = self.input_path_entry.text()
        output_folder = self.output_path_entry.text()
        planet = self.planet_dropdown.currentText()

        if input_folder == '' or output_folder == '':
            palette = QPalette()
            palette.setColor(QPalette.WindowText, QColor('red'))
            self.search_result_label.setPalette(palette)
            self.search_result_label.setText(
                'Please specify an input and output folder.')

        else:
            # empty image
            empty_pixmap = QPixmap(400, 400)
            empty_pixmap.fill(Qt.transparent)

            self.preview_image.setPixmap(empty_pixmap)
            self.annotated_image.setPixmap(empty_pixmap)

            # empty table
            self.CDM_table.setRowCount(0)
            self.GT_table.setRowCount(0)

            self.search_result_label.setText('')

            # check if grount truth data available
            if os.path.exists(os.path.join(self.input_folder, 'labels')):

                # check if folder is empty
                numLabels = len(os.listdir(
                    os.path.join(self.input_folder, 'labels')))

                if numLabels == len(self.images):
                    self.gt_available = True
                    self.bounding_box2.show()
                    self.grid_tab2.addWidget(self.bounding_box1, 25, 0, 10, 5)

                else:
                    self.gt_available = False
                    self.bounding_box2.hide()
                    self.grid_tab2.addWidget(self.bounding_box1, 25, 0, 10, 10)

            else:
                self.gt_available = False
                self.bounding_box2.hide()
                self.grid_tab2.addWidget(self.bounding_box1, 25, 0, 10, 10)

            self.predictor = Predictor(
                input_folder, output_folder, planet)

            # show predicted images when finsihed
            self.predictor.finished.connect(self.show_predictions)

            # switch to predictions tab
            self.tabs.setCurrentIndex(1)

            # start thread
            self.predictor.start()

    def show_predictions(self):
        """
        Shows the prediction, when it is done.
        """
        self.did_prediction = True
        self.display_image()
        self.updateBBTables()
        self.updateStatistics()

    def updateBBTables(self):
        """
        Updated the Tables with the
        bounding box data. If gound truth
        data is provided it does update two
        tables. Otherwise it updated
        only one table.
        """

        filename = os.path.splitext(self.images[self.current_image_index])[0]

        # csv file from model results always has an additional
        # column for the index
        pred_df = pd.read_csv(os.path.join(
            self.output_folder, 'detections', filename + '.csv'), index_col=0)

        self.CDM_table.setRowCount(0)
        self.GT_table.setRowCount(0)

        for i, row in pred_df.iterrows():
            self.CDM_table.insertRow(i)

            for j, item in enumerate(row):
                newitem = QTableWidgetItem(str(round(item, 4)))
                self.CDM_table.setItem(i, j, newitem)

        # if ground truth data is available
        if self.gt_available:
            # csv file from model results always has an additional
            # column for the index
            gt_df = pd.read_csv(os.path.join(
                self.input_folder, 'labels', filename + '.csv'))

            # no columns in csv specified
            if len(gt_df) == 0:
                gt_df = pd.read_csv(os.path.join(
                    self.input_folder, 'labels', filename + '.csv'),
                    header=None)

            else:
                if gt_df.iloc[0][1].dtype == 'float64':
                    gt_df = pd.read_csv(os.path.join(
                        self.input_folder, 'labels', filename + '.csv'),
                        header=None)

            # csv has more columns (e.g idx column, lat, lon, ...)
            # x, y, w, h is expected to be at position 1-5 if multiple
            # columns are given
            if len(gt_df.columns) != 4:
                gt_df = gt_df[gt_df.columns[1:5]]

            for i, row in gt_df.iterrows():
                self.GT_table.insertRow(i)

                for j, item in enumerate(row):
                    newitem = QTableWidgetItem(str(round(item, 6)))
                    self.GT_table.setItem(i, j, newitem)

    def updateStatistics(self):
        """
        Updated the statitics in the statistics
        tab: TP, FN, FP and distribution
        frequency.
        """

        # update TP, FN, FP if ground truth data is provided
        if self.gt_available:
            statistics_output_path = os.path.join(
                self.output_folder, 'statistics')
            statistic_files = os.listdir(statistics_output_path)

            combined = []

            for file in statistic_files:
                df = pd.read_csv(os.path.join(
                    statistics_output_path, file), index_col=None, header=0)
                combined.append(df)

            combinedDF = pd.concat(combined, axis=0, ignore_index=True)
            combinedDF = combinedDF.sum()

            self.TP_entry.setText(str(combinedDF['tp']))
            self.FP_entry.setText(str(combinedDF['fp']))
            self.FN_entry.setText(str(combinedDF['fn']))

        # reset TP, FN, FP if no ground truth data is available
        else:
            self.TP_entry = ''
            self.FP_entry = ''
            self.FN_entry = ''

        # show freq distribution plot
        self.freq_size_plot(os.path.join(
            self.output_folder, 'detections'))

    def freq_size_plot(self, path):
        """
        Shows the frequency size plot in the
        statistics tab.
        """

        files = []

        # Create a dataframe list by using a list comprehension
        files = [pd.read_csv(file)
                 for file in glob.glob(os.path.join(path, "*.csv"))]

        # Concatenate the list of DataFrames into one
        files_df = pd.concat(files)

        # get D
        w = files_df['w']
        h = files_df['h']
        files_df['d'] = w+h/2

        # histogram plot
        kwargs = dict(alpha=0.5, bins=100)

        # clearing old figure
        self.figure.clear()
        self.figure.patch.set_facecolor('white')

        # create an axis
        ax = self.figure.add_subplot(111)

        # plot data

        ax.set_title('Frequency-size Distribution')
        ax.set_xlabel('D')
        ax.set_ylabel('Frequency')
        ax.set_xscale(value='log')

        ax.hist(files_df['d'], **kwargs, color='b')

        # refresh canvas
        self.size_freq_canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
