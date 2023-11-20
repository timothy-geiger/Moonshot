# How to start and use the User Interface

This is a short GUID on how to start the User Interface. 

# Installation

First the user have to install the required packages:

```console
$ pip install -r requirements.txt
```

After that the moonshot package must be installed:

```console
$ pip install -r pip install -e .
```

Before running the GUI, the user must download
the weights:

```console
$ python3 utils/download_weights.py
```

To start the application, please execute the
following command:

```console
$ python3 moonshot/gui/gui_main.py
```

# How to use the GUI

After starting the application, the user has to
do 3 things.

<br>

1. First of all an input folder must
be specified. The input folder has to contain a
subdirectory called images. The user should place
the images he wants to use for the detection
into this folder. Optionally the user can provide
the labels. To do that he must create another
folder inside the input folder called "labels".
This subfolder must contain a csv file
for every image in the images folder. The name
has to be the same ecxept the file ending.

2. The second thing the user has to specify
is the output folder. <b>This folder has to
be empty!</b>

3. Now the user has to chose the planet he
wants to to the detection for.

<br>
<br>

Now the user can click on the predict button
to see the results.