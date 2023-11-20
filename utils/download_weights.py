import os
import gdown

# Where to fetch data from
WEIGHTS_MARS_URL = 'https://drive.google.com/drive/folders/' + \
    '1Q3LxmPnABlD9a5_tYgsimidXj84tJTGN?usp=share_link'

WEIGHTS_MOON_URL = 'https://drive.google.com/drive/folders/' + \
    '1FO56YQokKGI8KN-niFDnqGqxfsP3fqOB?usp=share_link'


# Where to save
WEIGHTS_MARS_DEST = os.sep.join(
    ['.', 'weights', 'mars'])

WEIGHTS_MOON_DEST = os.sep.join(
    ['.', 'weights', 'moon'])


# Download Model Data
print('Downloading mars weights to: ', WEIGHTS_MARS_DEST)
gdown.download_folder(url=WEIGHTS_MARS_URL,
                      output=WEIGHTS_MARS_DEST,
                      quiet=False)

print('Downloading moon weights to: ', WEIGHTS_MOON_DEST)
gdown.download_folder(url=WEIGHTS_MOON_URL,
                      output=WEIGHTS_MOON_DEST,
                      quiet=False)
