import sys; sys.path.append('..')
import pandas as pd
from sklearn.model_selection import KFold
from train.params import args


if __name__ == '__main__':
    cv_total = args.n_folds
    seed = args.seed
    labels_path = args.labels
    folds_file = args.folds_file

    train_df = pd.read_csv(labels_path)
    classes = train_df['label'].drop_duplicates().tolist()
    to_concat = []
    for cls in classes:
        class_df = train_df[train_df['label'] == cls]
        class_df['image_name'] = class_df['image_name'].apply(lambda x: x.split('.')[0])
        target_df = class_df[['image_name']]
        target_df = target_df.drop_duplicates()
        target_df['fold'] = -1

        kf = KFold(n_splits=cv_total, random_state=seed, shuffle=True)
        for i, (train_index, evaluate_index) in enumerate(kf.split(target_df.index.values)):
            target_df['fold'].iloc[evaluate_index] = i
        to_concat.append(target_df)
    target_df = pd.concat(to_concat)
    result_df = target_df.sort_values(by='fold')
    result_df.columns = ['image_id', 'fold']
    
    result_df.to_csv(folds_file, index=False)
