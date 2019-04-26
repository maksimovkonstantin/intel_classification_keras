import torch
import numpy as np
import pandas as pd
from os import listdir
from tqdm import tqdm


if __name__ == '__main__':

    predicts = torch.load('/project/submits/balanced_tensor.pt')
    predicts = predicts.numpy()
    test_folder = '/project/input_data/test'
    to_process = sorted(listdir(test_folder))
   #  print(predicts.shape)
    submit_df = {'image_name': [],
                 'label': []}

    for file_i, d in enumerate(tqdm(to_process[:])):
        # print(file_i)
        file_predict = predicts[file_i]
        # print(file_predict)
        # print(np.sum(file_predict))
        # print(file_predict.shape)
        final_class = np.argmax(file_predict)
        submit_df['image_name'].append(d)
        submit_df['label'].append(final_class)

    submit_df = pd.DataFrame(submit_df)
    submit_df = submit_df[['image_name', 'label']]
    submit_df.to_csv('/project/submits/balanced_submit.csv', index=False)