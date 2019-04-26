import numpy as np
import pandas as pd
from .base_dataset import BaseDatasetIterator
from albumentations import RandomCrop, Resize, Compose, PadIfNeeded


class IntelDataset:
    def __init__(self,
                 images_dir,
                 folds_file,
                 labels_path,
                 fold=0,
                 fold_num=5
                 ):
        super().__init__()
        self.fold = fold
        self.folds_file = folds_file
        self.fold_num = fold_num
        self.images_dir = images_dir
        self.train_ids, self.val_ids = self.generate_ids()
        self.labels_path = labels_path
        print("Found {} train images".format(len(self.train_ids)))
        print("Found {} val images".format(len(self.val_ids)))

    def get_generator(self, image_ids, crop_shape, preprocessing_function='tf',
                      random_transformer=None, batch_size=16, shuffle=True):
        return IntelDatasetIterator(
            self.images_dir,
            preprocessing_function,
            image_ids,
            crop_shape,
            self.labels_path,
            random_transformer,
            batch_size,
            shuffle=shuffle,
            image_name_template="{id}.jpg"
        )

    def train_generator(self, crop_shape, preprocessing_function='tf', random_transformer=None, batch_size=16):
        return self.get_generator(self.train_ids, crop_shape, preprocessing_function,
                                  random_transformer, batch_size, True)

    def val_generator(self,crop_shape, preprocessing_function='tf', batch_size=1):
        return self.get_generator(self.val_ids, crop_shape, preprocessing_function, None, batch_size, False)

    def generate_ids(self):
        
        df = pd.read_csv(self.folds_file)
        
        val_ids = df[(df['fold'] == self.fold)]['image_id'].values
        train_ids = np.sort(df[(df['fold'] != self.fold)]['image_id'].values)
        return train_ids, val_ids


class IntelDatasetIterator(BaseDatasetIterator):

    def __init__(self,
                 images_dir,
                 preprocessing_function,
                 image_ids,
                 crop_shape,
                 labels_path,
                 
                 random_transformer=None,
                 batch_size=1,
                 shuffle=True,
                 image_name_template=None):
        super().__init__(images_dir,
                         preprocessing_function,
                         image_ids,
                         crop_shape,
                         labels_path,
                         
                         random_transformer,
                         batch_size,
                         shuffle,
                         image_name_template)

    def pad_image(self, image, crop_shape):
        composed = Compose([PadIfNeeded(crop_shape[0], crop_shape[1], p=1),
                            RandomCrop(crop_shape[0], crop_shape[1], p=1)], p=1)

        croped = composed(image=image)

        image_padded = croped['image']

        return image_padded
