import os
from processing import lunarPreprocessor

if __name__ == "__main__":
    # Parameters
    moon_path = '../../data/moon'
    im_path = os.path.join(moon_path, 'images', 'Lunar_A.jpg')
    labels_path = os.path.join(moon_path, 'labels',
                               'lunar_crater_database_robbins_train.csv')
    split_name = 'A'
    m_per_pix = 100
    tile_size = 416

    P = lunarPreprocessor(im_path,
                          labels_path,
                          split_name,
                          m_per_pix,
                          tile_size)

    output_dir = '../../data/moon_new'

    P.create_train_data(output_dir)
