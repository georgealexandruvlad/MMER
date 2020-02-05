from metadata import MAX_PAD_LEN
from utils.read import Datasets, walk_frames, display_frame
import tensorflow as tf
import numpy as np


class DataLoader:

    def __init__(self, files: dict):
        """"
        :param files: dict with train, validation, maybe test
        """
        self.files = files

    @staticmethod
    def decode(serialized_example, classes):
        # Prepare feature list; read encoded JPG images as bytes
        features = dict()
        features["class_label"] = tf.io.FixedLenFeature((), tf.int64)
        features["frames"] = tf.io.VarLenFeature(tf.string)
        features["num_frames"] = tf.io.FixedLenFeature((), tf.int64)

        # keep them just in case
        # features['height'] = tf.io.FixedLenFeature((), tf.int64)
        # features['width'] = tf.io.FixedLenFeature((), tf.int64)
        # features['channels'] = tf.io.FixedLenFeature((), tf.int64)
        # features['class_text'] = tf.io.FixedLenFeature([], tf.string, default_value='')
        # features['video_id'] = tf.io.FixedLenFeature([], tf.string, default_value='')
        # features['desc_id'] = tf.io.FixedLenFeature([], tf.string, default_value='')

        # Parse into tensors
        parsed_features = tf.io.parse_single_example(serialized_example, features)

        # Randomly sample offset from the valid range.
        delta = tf.math.maximum(tf.constant(1, dtype=tf.int32),
                                tf.cast(tf.math.ceil(parsed_features["num_frames"] / 64), dtype=tf.int32))
        offsets = tf.range(0, parsed_features["num_frames"], delta)

        # Decode the encoded JPG images
        images = tf.map_fn(
            lambda i: tf.image.decode_jpeg(parsed_features["frames"].values[i], channels=3), offsets, dtype=tf.uint8)

        label = tf.cast(parsed_features["class_label"], tf.int32)

        label = tf.one_hot(label, classes)
        images = images[:128]
        images = tf.image.resize(images, (300, 300), method=tf.image.ResizeMethod.AREA, antialias=True)
        images = tf.cast(images, dtype=tf.float32) * 1 / 255

        return images, label

    @staticmethod
    def decode_all(serialized_example, classes):
        # Prepare feature list; read encoded JPG images as bytes
        features = dict()
        features["class_label"] = tf.io.FixedLenFeature((), tf.int64)
        features["video_frames"] = tf.io.VarLenFeature(tf.string)
        features["video_num_frames"] = tf.io.FixedLenFeature((), tf.int64)

        features['audio_dim0'] = tf.io.FixedLenFeature((), tf.int64)
        features['audio_dim1'] = tf.io.FixedLenFeature((), tf.int64)
        features['audio_num_rows'] = tf.io.FixedLenFeature((), tf.int64)
        features['audio_num_columns'] = tf.io.FixedLenFeature((), tf.int64)
        features['audio_num_channels'] = tf.io.FixedLenFeature((), tf.int64)
        features['audio_info'] = tf.io.VarLenFeature(tf.float32)

        features['motion_num_frames'] = tf.io.FixedLenFeature((), tf.int64)
        features['motion_num_params'] = tf.io.FixedLenFeature((), tf.int64)
        features['motion_frames'] = tf.io.VarLenFeature(tf.float32)

        # keep them just in case
        # features['height'] = tf.io.FixedLenFeature((), tf.int64)
        # features['width'] = tf.io.FixedLenFeature((), tf.int64)
        # features['channels'] = tf.io.FixedLenFeature((), tf.int64)
        # features['class_text'] = tf.io.FixedLenFeature([], tf.string, default_value='')
        # features['video_id'] = tf.io.FixedLenFeature([], tf.string, default_value='')
        # features['desc_id'] = tf.io.FixedLenFeature([], tf.string, default_value='')

        # Parse into tensors
        parsed_features = tf.io.parse_single_example(serialized_example, features)

        # audio
        audio_dim0 = tf.cast(parsed_features['audio_dim0'], tf.int64)
        audio_dim1 = tf.cast(parsed_features['audio_dim1'], tf.int64)
        audio_num_rows = tf.cast(parsed_features['audio_num_rows'], tf.int64)
        audio_num_columns = tf.cast(parsed_features['audio_num_columns'], tf.int64)
        audio_num_channels = tf.cast(parsed_features['audio_num_channels'], tf.int64)

        audio_frames = tf.reshape(parsed_features["audio_info"].values, tf.stack([audio_dim0, audio_dim1]))
        audio_frames = tf.reshape(audio_frames, tf.stack([audio_num_rows, audio_num_columns, audio_num_channels]))
        audio_frames = tf.cast(audio_frames, dtype=tf.float32)

        # motion
        motion_num_frames = tf.cast(parsed_features['motion_num_frames'], tf.int64)
        motion_num_params = tf.cast(parsed_features['motion_num_params'], tf.int64)

        """
        motion_delta = tf.math.maximum(tf.constant(1, dtype=tf.int32),
                                       tf.cast(tf.math.ceil(parsed_features["motion_num_frames"] / 64), dtype=tf.int32))
        motion_offsets = tf.range(0, parsed_features["motion_num_frames"], motion_delta)

        motion_frames = tf.map_fn(
            lambda i: parsed_features["motion_frames"].values[i], motion_offsets, dtype=tf.float32)
        """
        motion_frames = tf.reshape(parsed_features["motion_frames"].values,
                                   tf.stack([motion_num_frames, motion_num_params]))

        motion_delta = tf.math.maximum(tf.constant(1, dtype=tf.int32),
                                       tf.cast(tf.math.ceil(parsed_features["motion_num_frames"] / 64), dtype=tf.int32))
        motion_offsets = tf.range(0, parsed_features["motion_num_frames"], motion_delta)

        motion_frames = tf.map_fn(
            lambda i: motion_frames[i], motion_offsets, dtype=tf.float32)

        motion_frames = motion_frames[:64]
        motion_frames = tf.cast(motion_frames, dtype=tf.float32)

        # Randomly sample offset from the valid range.
        delta = tf.math.maximum(tf.constant(1, dtype=tf.int32),
                                tf.cast(tf.math.ceil(parsed_features["video_num_frames"] / 64), dtype=tf.int32))
        offsets = tf.range(0, parsed_features["video_num_frames"], delta)

        # Decode the encoded JPG images
        video_frames = tf.map_fn(
            lambda i: tf.image.decode_jpeg(parsed_features["video_frames"].values[i], channels=3), offsets,
            dtype=tf.uint8)

        video_frames = video_frames[:64]
        video_frames = tf.image.resize(video_frames, (300, 300), method=tf.image.ResizeMethod.AREA, antialias=True)
        video_frames = tf.cast(video_frames, dtype=tf.float32) * 1 / 255

        label = tf.cast(parsed_features["class_label"], tf.int32)
        label = tf.one_hot(label, classes)

        return video_frames, audio_frames, motion_frames, label

    @staticmethod
    def decode_pretrain(serialized_example, classes):
        # Prepare feature list; read encoded JPG images as bytes
        features = dict()
        features["class_label"] = tf.io.FixedLenFeature((), tf.int64)
        features["image"] = tf.io.FixedLenFeature([], tf.string)

        # keep them just in case
        # features['height'] = tf.io.FixedLenFeature((), tf.int64)
        # features['width'] = tf.io.FixedLenFeature((), tf.int64)
        # features['channels'] = tf.io.FixedLenFeature((), tf.int64)

        # Parse into tensors
        parsed_features = tf.io.parse_single_example(serialized_example, features)

        # Decode the encoded JPG images
        images = tf.image.decode_jpeg(parsed_features["image"], channels=3)

        label = tf.cast(parsed_features["class_label"], tf.int32)

        label = tf.one_hot(label, classes)
        images = tf.image.resize(images, (300, 300), method=tf.image.ResizeMethod.AREA, antialias=True)
        images = tf.cast(images, dtype=tf.float32) * 1 / 255

        return images, label

    def get_dataset(self, dataset='train', batch_size=1, shuffle_size=128, classes=7):
        """
        Load a dataset and apply mapping, shuffling, batching, etc policies

        :param dataset: which dataset to load: train, validation
        :param batch_size: dim of a batch
        :param shuffle_size: dim of shuffling window
        :param classes: total number of classes for one hot encoding purposes
        :return: tensorflow dataset
        """

        file = self.files[dataset]
        dataset = tf.data.TFRecordDataset(file).prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(shuffle_size)
        if 'pretrain' in file:
            dataset = dataset.map(lambda x: self.decode_pretrain(x, classes),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.padded_batch(batch_size, padded_shapes=([300, 300, 3], [7]))
        else:
            dataset = dataset.map(lambda x: self.decode_all(x, classes),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.padded_batch(batch_size,
                                           padded_shapes=([64, 300, 300, 3], [40, MAX_PAD_LEN, 1], [64, 152], [7]))
            dataset = dataset.map(lambda video, audio, motion, classes: ((video, audio, motion), classes))
        return dataset


def get_dataset_size(ds):
    """
    Get size of dataset
    :param ds: dataset
    :return: the size of the dataset
    """
    size = 0
    for _ in ds:
        size += 1
    return size


def get_dataset_structure(ds):
    """
    Get shape of batches
    :param ds: dataset
    :return: shapes for batch, frames, height, width, channels, number of classes if valid dataset, otherwise None
    """
    for (data_video, data_audio, data_motion), labels in ds.take(1):
        batch, video_frames, video_height, video_width, video_channels = data_video.numpy().shape
        _, motion_frames, motion_features = data_motion.numpy().shape
        _, audio_num_rows, audio_num_columns, audio_num_channels = data_audio.numpy().shape
        classes = labels.numpy().shape[-1]

        d = dict()
        d["classes"] = classes
        d["batch"] = batch
        d["video_frames"] = video_frames
        d["video_height"] = video_height
        d["video_width"] = video_width
        d["video_channels"] = video_channels
        d["motion_frames"] = motion_frames
        d["motion_features"] = motion_features
        d['audio_num_rows'] = audio_num_rows
        d['audio_num_columns'] = audio_num_columns
        d['audio_num_channels'] = audio_num_channels
        return d

        # if len(data.numpy().shape) == 5:
        #     batch, frames, height, width, channels = data.numpy().shape
        #     classes = labels.numpy().shape[-1]
        #     d = dict()
        #     d["batch"] = batch
        #     d["frames"] = frames
        #     d["height"] = height
        #     d["width"] = width
        #     d["channels"] = channels
        #     d["classes"] = classes
        #     return d
        # elif len(data.numpy().shape) == 4:
        #     batch, height, width, channels = data.numpy().shape
        #     classes = labels.numpy().shape[-1]
        #     d = dict()
        #     d["batch"] = batch
        #     d["height"] = height
        #     d["width"] = width
        #     d["channels"] = channels
        #     d["classes"] = classes
        #     return d
    return None


if __name__ == '__main__':
    train = Datasets.phase1["all_train_records"]
    validation = Datasets.phase1["all_validation_records"]
    files = {'train': train,
             'validation': validation
             }

    # train = Datasets.KDEF["train_records"]
    # validation = Datasets.KDEF["validation_records"]
    # files = {'train': train,
    #          'validation': validation
    #          }

    dl = DataLoader(files)
    dataset = dl.get_dataset('train')
    print(get_dataset_structure(dataset))
    # print(get_dataset_size(dataset))
    # print(get_dataset_size(dataset))

    # for img, label in dataset.take(10):
    #     # print(label.numpy().shape)
    #     # print(img.numpy().shape)
    #     display_frame((img.numpy()[0] * 255).astype(np.uint8))
    # print(np.array_equal(img.numpy(), np.zeros((1, 50, 224, 224, 3))))
    # walk_frames((img.numpy()[0] * 255).astype(np.uint8))
    #     print(delta.numpy())
    #     print(frames.numpy())
    # # to walk the frames
    # if len(img.numpy() == 8):
    #     walk_frames(img.numpy())
