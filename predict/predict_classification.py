import os
import sys; sys.path.append('..')

import torch

from train.params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from keras.applications.imagenet_utils import preprocess_input

from train.model_factory import make_model
from albumentations import PadIfNeeded, CenterCrop, Compose, Resize, RandomCrop
from os import path, listdir
import numpy as np
import timeit
import cv2
from tqdm import tqdm
import pandas as pd
# import skimage.io
# import skimage.color
from utils.helpers import load_img_fast_jpg

def preprocess_inputs(x):
    return preprocess_input(x, mode=args.preprocessing_function)


if __name__ == '__main__':

    test_folder = args.test_data
    cut_size = args.crop_size
    t0 = timeit.default_timer()
    crop_shape = args.crop_size
    weights = [os.path.join(args.models_dir, m) for m in args.models]
    models = []
    weights = weights[:]
    for w in weights[:]:
        model = make_model(args.network, input_shape=(cut_size, cut_size, 3), predict_flag=1)
        print("Building model {} from weights {} ".format(args.network, w))
        model.load_weights(w)
        models.append(model)
        # model.summary()


    to_process = sorted(listdir(test_folder))
    submit_df = {'image_name': [],
                 'label': []}
    final_tensor = []
    for d in tqdm(to_process[:]):
        
        final_mask = None
        
        fid = d
        
        img_path = path.join(test_folder, fid)
        img = (load_img_fast_jpg(img_path)).astype(np.uint8)
        #if len(img.shape) < 3:
        #    skimage.color.gray2rgb(img)
        composed = Compose([PadIfNeeded(crop_shape, crop_shape, p=1),
                            RandomCrop(crop_shape, crop_shape, p=1)], p=1)

        croped = composed(image=img)

        image_padded = croped['image']

        #if img.shape != (160, 160, 3):
        #    print('FUUFUFUFU')
        #    print(fid)
        #    print(image_padded.shape)
        inp0 = [image_padded]

        inp0.append(np.fliplr(image_padded))
        inp0.append(np.flipud(image_padded))

        inp0 = np.asarray(inp0)
        inp0 = preprocess_inputs(np.array(inp0, "float32"))

        n_augs = len(inp0)
        n_outputs = 6
        result = np.zeros((len(weights), n_augs, n_outputs),dtype=np.float64)
        for model_id, model in enumerate(models):
            pass
            if len(inp0.shape) < 4:
                continue
            pred0 = model.predict(inp0)
            # print(inp0.shape)

            # print(pred0.shape)
            # print(pred0)
            result[model_id, :, :] += pred0[:, :]
        # print(result)
        result = np.mean(result, axis=(0, 1))
        final_tensor.append(result)
        # print(result)
        final_class = np.argmax(result)
        # print(final_class)
        # print(final_class)
        submit_df['image_name'].append(fid)
        submit_df['label'].append(final_class)
    final_tensor = np.array(final_tensor)
    final_tensor = final_tensor.astype(np.float64)
    final_tensor = torch.from_numpy(final_tensor)
    print(final_tensor.shape)
    print(final_tensor.type())
    torch.save(final_tensor, '/project/submits/predict_tensor.pt')
    submit_df = pd.DataFrame(submit_df)
    submit_df = submit_df[['image_name', 'label']]
    submit_df.to_csv('/project/submits/submit.csv', index=False)
