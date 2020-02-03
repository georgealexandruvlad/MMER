import os
import argparse
import tensorflow as tf
from loaders.dataloader import DataLoader
from paths import Datasets
from tensorflow.keras.optimizers import Nadam
from loaders.dataloader import get_dataset_size, get_dataset_structure
from architectures.models import get_model
from paths import MODEL_PATH
from metadata import config
from tensorflow.keras.callbacks import ReduceLROnPlateau


class Trainer:

    def __init__(self, name, record_paths, config):
        self.record_paths = record_paths
        self.name = name
        self.config = config
        self.train_ds = None
        self.validation_ds = None
        self.test_ds = None

    def get_checkpoints(self):
        summary_dir = os.path.join(MODEL_PATH, self.name, 'summaries')
        checkpoint_path = os.path.join(MODEL_PATH, self.name, 'checkpoint')

        patience = self.config['patience']
        mode = self.config['mode']
        metric = self.config['metric']

        summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                                                       monitor=metric, mode=mode,
                                                                       save_best_only=True)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=metric, patience=patience, mode=mode)
        reduce_lr = ReduceLROnPlateau(monitor=metric, factor=0.2,
                                      patience=20, min_lr=0.0001)

        return [model_checkpoint_callback, early_stopping_callback, summary_callback, reduce_lr]

    def train(self, device: str, strategy_name='tpu'):

        # init strategy
        if strategy_name == 'tpu':
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
            print('TPU strategy selected.')
        else:
            strategy = tf.distribute.OneDeviceStrategy(device=device)
            print('GPU strategy selected.')

        # decode records to tensorflow datasets
        dl = DataLoader(self.record_paths)
        self.train_ds = dl.get_dataset('train', batch_size=self.config['train_batch_size'])
        self.validation_ds = dl.get_dataset('validation', batch_size=self.config['validation_batch_size'])

        # infer steps for fitting and data structure
        train_steps = get_dataset_size(self.train_ds)
        validation_steps = get_dataset_size(self.validation_ds)
        data_structure = get_dataset_structure(self.train_ds)

        callbacks = self.get_checkpoints()
        with strategy.scope():

            model = get_model(input_shape=data_structure, name=config['model'])
            optimizer = Nadam()
            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                          optimizer=optimizer,
                          metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall(),
                                   tf.keras.metrics.Precision()])
            model.summary()
            model.fit(self.train_ds.repeat(), epochs=self.config['epochs'], steps_per_epoch=train_steps,
                      validation_data=self.validation_ds.repeat(),
                      validation_steps=validation_steps, callbacks=callbacks)

    def evaluate(self, device: str, strategy_name, pretrained):
        # init strategy
        if strategy_name == 'tpu':
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
            print('TPU strategy selected.')
        else:
            strategy = tf.distribute.OneDeviceStrategy(device=device)
            print('GPU strategy selected.')

        dl = DataLoader(self.record_paths)
        self.test_ds = dl.get_dataset('test', batch_size=self.config['train_batch_size'])

        # infer steps for fitting and data structure
        test_steps = get_dataset_size(self.test_ds)
        data_structure = get_dataset_structure(self.test_ds)

        with strategy.scope():

            model = get_model(input_shape=data_structure, name="VGG16_tf")
            model.load_weights(os.path.join(MODEL_PATH, pretrained, 'checkpoint'))
            optimizer = Nadam()
            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                          optimizer=optimizer,
                          metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Recall(),
                                   tf.keras.metrics.Precision()])
            model.summary()
            model.evaluate(self.test_ds, steps=test_steps)

    def predict(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='gpu', help='Choose strategy tpu/gpu [default: gpu]')
    parser.add_argument('--version', default='', help='Add a version to output files')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--model', default='', help='Choose checkpoint folder')

    args = parser.parse_args()

    if args.pretrain:
        train = Datasets.KDEF["train_records"]
        validation = Datasets.KDEF["validation_records"]
        test = Datasets.KDEF["test_records"]
        config['model'] = 'VGG16_pretrain'
        config['train_batch_size'] = 64
        config['validation_batch_size'] = 64
    else:
        # train = Datasets.phase1["train_records"]
        # validation = Datasets.phase1["validation_records"]
        # test = Datasets.phase2["test_records"]
        train = Datasets.phase1["all_train_records"]
        validation = Datasets.phase1["all_validation_records"]
        test = Datasets.phase1["all_test_records"]

    files = {'train': train,
             'validation': validation,
             'test': test
             }

    if args.eval:
        trainer = Trainer('Model_{}'.format(args.version if args.version else 'default'), files, config)
        trainer.evaluate('/gpu:0', args.strategy, pretrained=args.model)
    else:
        trainer = Trainer('Model_{}'.format(args.version if args.version else 'default'), files, config)
        trainer.train('/gpu:0', args.strategy)
