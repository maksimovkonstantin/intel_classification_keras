import argparse
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--raw_data', default='/project/input_data/default_data')
arg('--train_data', default='/project/input_data/train')
arg('--test_data', default='/project/input_data/test')
arg('--labels', default='/project/input_data/train_labels.csv')
arg('--models_dir', default='/project/output_data/models_weights')
arg('--logs', default="/project/output_data/models_logs")
arg('--folds_file', default='/project/output_data/folds_split.csv')
arg('--n_folds', type=int, default=10)
arg('--seed', type=int, default=769)
arg('--gpu', default="0")
arg('--epochs', type=int, default=20)
arg('--crop_size', type=int, default=160)
arg('--fold', default='0')
arg('--batch_size', type=int, default=8)
arg('--network', default='seresnext50')
arg('--preprocessing_function', default='caffe')
arg('--alias', default='')
arg('--verbose', type=int, default=1)
arg('--num_workers', type=int, default=2)
arg('--optimizer', default="adam")
arg('--learning_rate', type=float, default=1e-3)
arg('--reduce_lr_patience', type=int, default=3)
arg('--reduce_lr_rate', type=float, default=0.5)
arg('--early_stopping', type=int, default=5)
arg('--loss_function', default='focal_loss')
arg('--models', nargs='+', default=['best_loss_cce_with_augs_seresnext50_fold_0.h5',
'best_loss_cce_with_augs_seresnext50_fold_1.h5',
'best_loss_cce_with_augs_seresnext50_fold_2.h5',
'best_loss_cce_with_augs_seresnext50_fold_3.h5'
#'best_loss_cce_with_augs_resnet18_fold_2.h5',
#'best_loss_cce_with_augs_resnet18_fold_3.h5',
#'best_loss_cce_with_augs_resnet18_fold_4.h5',
#'best_loss_cce_with_augs_resnet18_fold_5.h5',
#'best_loss_cce_with_augs_resnet18_fold_6.h5',
#'best_loss_cce_with_augs_resnet18_fold_7.h5',
#'best_loss_cce_with_augs_resnet18_fold_8.h5',
#'best_loss_cce_with_augs_resnet18_fold_9.h5'

                                    ])
arg('--augment', type=int, default=1)
arg('--save_period', type=int, default=1)
arg('--max_queue_size', type=int, default=100)
arg('--decay', type=float, default=0.000001)

args = parser.parse_args()