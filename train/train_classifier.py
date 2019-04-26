import sys; sys.path.append('..')
import gc
from params import args
from dataset.dataset import IntelDataset
from transforms import augmentations
from train.model_factory import make_model
from utils.losses import make_loss
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.metrics import categorical_accuracy
from keras.optimizers import RMSprop, Adam, SGD
import keras.backend as K
import keras_metrics as km


def main():
    folds = [int(f) for f in args.fold.split(",")]

    batch_size = args.batch_size

    if args.augment:
        transformer = augmentations()
    else:
        transformer = None

    optimizer_type = args.optimizer
    if args.preprocessing_function:
        preprocessor = args.preprocessing_function
    else:
        preprocessor = None

    models_dir = args.models_dir
    alias = args.alias
    network = args.network
    save_period = args.save_period
    loss_function = args.loss_function
    logs = args.logs

    for fold in folds:
        
        if optimizer_type == 'rmsprop':
            optimizer = RMSprop(lr=args.learning_rate, decay=float(args.decay))
        elif args.optimizer == 'adam':
            optimizer = Adam(lr=args.learning_rate, decay=float(args.decay))
        elif args.optimizer == 'amsgrad':
            optimizer = Adam(lr=args.learning_rate, decay=float(args.decay), amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = SGD(lr=args.learning_rate, momentum=0.9, nesterov=True, decay=float(args.decay))

        model = make_model(args.network, input_shape=(args.crop_size, args.crop_size, 3))
        
        dataset = IntelDataset(args.raw_data,
                                args.folds_file,
                                args.labels,
                                fold,
                                args.n_folds)
        
        train_generator = dataset.train_generator(
            (args.crop_size, args.crop_size),
            preprocessor,
            transformer,
            batch_size=batch_size)
        
        val_generator = dataset.val_generator((args.crop_size, args.crop_size),
                                              preprocessor, batch_size=1)
        
        callbacks = []
        # precision = km.binary_precision()
        # recall = km.binary_recall()
        metrics = [categorical_accuracy
                   # , precision, recall
                   ]

        best_loss_model_file = '{}/best_loss_{}_{}_{}_fold_{}.h5'.format(models_dir, loss_function, alias, network, fold)
        best_loss_model = ModelCheckpoint(best_loss_model_file,
                                          verbose=1,
                                          monitor='val_loss',
                                          mode='min',
                                          period=save_period,
                                          save_best_only=True,
                                          save_weights_only=True)
        callbacks.append(best_loss_model)

        tb = TensorBoard("{}/{}_{}_{}".format(logs, network, fold, loss_function))
        callbacks.append(tb)

        reducer = ReduceLROnPlateau(monitor='val_loss', factor=args.reduce_lr_rate, patience=args.reduce_lr_patience, min_lr=1e-6,
                                    epsilon=0.001, verbose=1,
                                    mode='min')
        callbacks.append(reducer)
        es = EarlyStopping(monitor='val_loss',  mode='min',  patience=args.early_stopping)
        callbacks.append(es)
        steps_per_epoch = len(dataset.train_ids) / args.batch_size + 1

        model.compile(loss=make_loss(loss_function),
                      optimizer=optimizer,
                      metrics=metrics
                      )
        
        validation_data = val_generator
        validation_steps = len(dataset.val_ids)
        max_queue_size = args.max_queue_size
        verbose = args.verbose
        num_workers = args.num_workers

        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,
            verbose=verbose,
            workers=num_workers)
        
        del model
        K.clear_session()
        gc.collect()


if __name__ == '__main__':
    main()
