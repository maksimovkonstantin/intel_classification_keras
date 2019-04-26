import sys; sys.path.append('..')
import os
import numpy as np
from keras.applications import imagenet_utils
from keras.preprocessing.image import Iterator
# import skimage.io
import pandas as pd
from utils.helpers import load_img_fast_jpg

class BaseDatasetIterator(Iterator):
    def __init__(self,
                 images_dir,
                 preprocessing_function,
                 image_ids,
                 crop_shape,
                 labels_path,
                 random_transformer=None,
                 batch_size=1,
                 shuffle=True,
                 image_name_template=None,
                 ):
        self.images_dir = images_dir
        self.image_ids = image_ids
        self.image_name_template = image_name_template
        self.random_transformer = random_transformer
        self.crop_shape = crop_shape
        self.preprocessing_function = preprocessing_function
        self.labels_path = labels_path
        labels_df = pd.read_csv(labels_path)
        labels_df['image_id'] = labels_df['image_name'].apply(lambda x: int(x.split('.')[0]))
        self.labels_df = labels_df
        super(BaseDatasetIterator, self).__init__(len(self.image_ids), batch_size, shuffle, None)

    def pad_image(self, image, crop_shape):
        return NotImplementedError

    def transform_batch_y(self, batch_y):
        return batch_y

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = []
        batch_y = []

        for batch_index, image_index in enumerate(index_array):
            _id = self.image_ids[image_index]
            # df = self.labels_df.copy()
            df = self.labels_df
            # print(df)
            row_label_info = df[df['image_id'] == _id]
            # print(row_label_info)

            row_label_info = row_label_info.iloc[0, :]

            label = row_label_info['label']
            # print(label)
            # subpath = row_label_info['subpath']
            # final_labels = [label, 1 - label]
            final_labels = [0]*6
            final_labels[int(label)] = 1
            # print(final_labels)
            img_name = self.image_name_template.format(id=_id)
            path = os.path.join(self.images_dir, img_name)

            # image = skimage.io.imread(path)
            image = (load_img_fast_jpg(path)).astype(np.uint8)
            image = self.pad_image(image, self.crop_shape)
            if self.random_transformer is not None:
                data = self.random_transformer(image=image)
                crop_image = data['image']
            else:
                crop_image = image

            batch_x.append(crop_image)
            batch_y.append(final_labels)

        batch_x = np.array(batch_x, dtype="float32")
        batch_y = np.array(batch_y, dtype="float32")
        # print(batch_x.shape)
        if self.preprocessing_function:
            batch_x = imagenet_utils.preprocess_input(batch_x, mode=self.preprocessing_function)

        return self.transform_batch_x(batch_x), self.transform_batch_y(batch_y)

    def transform_batch_x(self, batch_x):
        return batch_x

    def next(self):

        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)


