from paths import *
from utils.read import get_frames_from_video, read_img, get_frames_from_motion, get_wav_from_audio
from collections import defaultdict
from tqdm import tqdm
import tensorflow as tf
from loaders.dataset import Dataset
from metadata import *
from multiprocessing import Pool


# noinspection PyShadowingNames
class TfRecordData:

    def __init__(self, videos_path, motions_path, audios_path):
        self.videos_path = videos_path
        self.motions_path = motions_path
        self.audios_path = audios_path

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _float_list_feature(values):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_list_feature(values):
        """Wrapper for inserting bytes features into Example proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    def frames_example(self, frames, label, video_id, desc_id):

        features = dict()
        features['num_frames'] = self._int64_feature(frames.shape[0])
        features['height'] = self._int64_feature(frames.shape[1])
        features['width'] = self._int64_feature(frames.shape[2])
        features['channels'] = self._int64_feature(frames.shape[3])
        features['class_label'] = self._int64_feature(EMOTION_LABELS[label])
        features['class_text'] = self._bytes_feature(tf.compat.as_bytes(label))
        features['video_id'] = self._bytes_feature(tf.compat.as_bytes(video_id))
        features['desc_id'] = self._bytes_feature(tf.compat.as_bytes(desc_id))

        # Compress the frames using JPG and store in as a list of strings in 'frames'
        encoded_frames = [tf.image.encode_jpeg(frame).numpy() for frame in frames]

        features['frames'] = self._bytes_list_feature(encoded_frames)

        return tf.train.Example(features=tf.train.Features(feature=features))

    def all_data_object(self, video_frames, audio_info, motion_frames, label, video_id, audio_id, motion_id, desc_id):
        features = dict()

        features['class_label'] = self._int64_feature(EMOTION_LABELS[label])
        features['class_text'] = self._bytes_feature(tf.compat.as_bytes(label))
        features['desc_id'] = self._bytes_feature(tf.compat.as_bytes(desc_id))

        # audio
        features['audio_dim0'] = self._int64_feature(audio_info.shape[0])
        features['audio_dim1'] = self._int64_feature(audio_info.shape[1])
        features['audio_id'] = self._bytes_feature(tf.compat.as_bytes(audio_id))
        features['audio_num_rows'] = self._int64_feature(40)
        features['audio_num_columns'] = self._int64_feature(MAX_PAD_LEN)
        features['audio_num_channels'] = self._int64_feature(1)
        audio_info = audio_info.flatten()
        features['audio_info'] = self._float_list_feature(audio_info)
        # end of audio

        # video
        features['video_num_frames'] = self._int64_feature(video_frames.shape[0])
        features['video_height'] = self._int64_feature(video_frames.shape[1])
        features['video_width'] = self._int64_feature(video_frames.shape[2])
        features['video_channels'] = self._int64_feature(video_frames.shape[3])
        features['video_id'] = self._bytes_feature(tf.compat.as_bytes(video_id))

        # Compress the frames using JPG and store in as a list of strings in 'frames'
        encoded_frames = [tf.image.encode_jpeg(frame).numpy() for frame in video_frames]

        features['video_frames'] = self._bytes_list_feature(encoded_frames)
        # end of video

        # motion
        features['motion_num_frames'] = self._int64_feature(motion_frames.shape[0])
        features['motion_num_params'] = self._int64_feature(motion_frames.shape[1])
        features['motion_id'] = self._bytes_feature(tf.compat.as_bytes(motion_id))

        motion_frames = motion_frames.flatten()

        features['motion_frames'] = self._float_list_feature(motion_frames)

        # end of motion
        return tf.train.Example(features=tf.train.Features(feature=features))

    def to_records_all(self, data, record_file, resize=None, extract_faces_size=None):
        # make it faster -> open videos only once and slice from there
        group_data = defaultdict(list)
        for d in data:
            video_id = d['video']['filename']
            group_data[video_id].append(d)

        with tf.io.TFRecordWriter(record_file) as writer:
            for video_id, list_desc in group_data.items():

                audio_id = video_id.split('.')[0] + '.wav'
                video_path = os.path.join(self.videos_path, video_id)
                audio_path = os.path.join(self.audios_path, audio_id)

                all_frames = get_frames_from_video(video_path, resize)

                print('\n#####################################\nVideo {}'.format(video_id))
                for d in tqdm(list_desc):
                    motion_id = d['kinect']['filename']
                    kinect_frame_start = d['kinect']['frame_start']
                    kinect_frame_end = d['kinect']['frame_end']
                    motion_path = os.path.join(self.motions_path, motion_id)
                    motion_all_frames = get_frames_from_motion(motion_path)

                    video_id = d['video']['filename']
                    desc_id = d['filename']
                    frame_start = d['video']['frame_start']
                    frame_end = d['video']['frame_end']
                    time_start = d['video']['time_start']
                    time_end = d['video']['time_end']
                    label = d['emotion']

                    frames = all_frames[frame_start:frame_end]
                    audio_info = get_wav_from_audio(audio_path, time_start, time_end)
                    selected_motion_frames = motion_all_frames[kinect_frame_start:kinect_frame_end]

                    # # motion area - face extraction
                    # if extract_faces_size is not None:
                    #     frames = detect_faces(frames, extract_faces_size)

                    tf_example = self.all_data_object(frames, audio_info, selected_motion_frames, label, video_id,
                                                      audio_id, motion_id, desc_id)
                    writer.write(tf_example.SerializeToString())

    def image_example(self, image, label):

        features = dict()
        features['height'] = self._int64_feature(image.shape[0])
        features['width'] = self._int64_feature(image.shape[1])
        features['channels'] = self._int64_feature(image.shape[2])
        features['class_label'] = self._int64_feature(EMOTION_LABELS[label])
        features['class_text'] = self._bytes_feature(tf.compat.as_bytes(label))

        # Compress the image using JPG
        encoded_image = tf.image.encode_jpeg(image).numpy()

        features['image'] = self._bytes_feature(encoded_image)

        return tf.train.Example(features=tf.train.Features(feature=features))

    def motion_example(self, frames, label, motion_id, desc_id):
        features = dict()
        features['num_frames'] = self._int64_feature(frames.shape[0])
        features['num_params'] = self._int64_feature(frames.shape[1])
        features['class_label'] = self._int64_feature(EMOTION_LABELS[label])
        features['class_text'] = self._bytes_feature(tf.compat.as_bytes(label))
        features['video_id'] = self._bytes_feature(tf.compat.as_bytes(motion_id))
        features['desc_id'] = self._bytes_feature(tf.compat.as_bytes(desc_id))

        frames = frames.flatten()

        features['frames'] = self._float_list_feature(frames)

        return tf.train.Example(features=tf.train.Features(feature=features))

    def audio_example(self, audio_info, label, audio_id, desc_id):
        features = dict()

        features['audio_dim0'] = self._int64_feature(audio_info.shape[0])
        features['audio_dim1'] = self._int64_feature(audio_info.shape[1])
        features['class_label'] = self._int64_feature(EMOTION_LABELS[label])
        features['class_text'] = self._bytes_feature(tf.compat.as_bytes(label))
        features['audio_id'] = self._bytes_feature(tf.compat.as_bytes(audio_id))
        features['desc_id'] = self._bytes_feature(tf.compat.as_bytes(desc_id))

        features['num_rows'] = self._int64_feature(40)
        features['num_columns'] = self._int64_feature(MAX_PAD_LEN)
        features['num_channels'] = self._int64_feature(1)

        audio_info = audio_info.flatten()
        features['audio_info'] = self._float_list_feature(audio_info)

        return tf.train.Example(features=tf.train.Features(feature=features))

    def _video_generator(self, data, resize=None, bulk=False):
        # make it faster -> open videos only once and slice from there (use bulk=True)
        if bulk:
            _data = defaultdict(list)
            for d in data:
                video_id = d['video']['filename']
                _data[video_id].append(d)
        else:
            _data = data

        # speed up - few video openings
        if bulk:
            for video_id, list_desc in _data.items():
                video_path = os.path.join(self.videos_path, video_id)
                all_frames = get_frames_from_video(video_path, resize)

                print('\n#####################################\nVideo {}'.format(video_id))
                for d in tqdm(list_desc):
                    video_id = d['video']['filename']
                    desc_id = d['filename']
                    frame_start = d['video']['frame_start']
                    frame_end = d['video']['frame_end']
                    label = d['emotion']
                    frames = all_frames[frame_start:frame_end]
                    tf_example = self.frames_example(frames, label, video_id, desc_id)
                    yield tf_example
        else:
            # memory friendly, much slower
            for desc in tqdm(_data):
                video_path = os.path.join(self.videos_path, desc['video']['filename'])
                frames_interval = (desc['video']['frame_start'], desc['video']['frame_end'])
                frames = get_frames_from_video(video_path, resize, frames_interval)

                video_id = desc['video']['filename']
                desc_id = desc['filename']
                label = desc['emotion']
                tf_example = self.frames_example(frames, label, video_id, desc_id)
                yield tf_example

    def _image_generator(self, data, resize=None):
        for desc in tqdm(data):
            img_path = desc['image']
            img = read_img(img_path, resize)

            label = desc['emotion']
            tf_example = self.image_example(img, label)
            yield tf_example

    def _motion_generator(self, data, resize=None):
        for d in data:
            motion_id = d['kinect']['filename']
            motion_path = os.path.join(self.motions_path, motion_id)
            all_frames = get_frames_from_motion(motion_path)

            desc_id = d['filename']
            frame_start = d['kinect']['frame_start']
            frame_end = d['kinect']['frame_end']
            label = d['emotion']
            selected_frames = all_frames[frame_start:frame_end]

            tf_example = self.motion_example(selected_frames, label, motion_id, desc_id)
            yield tf_example

    # def to_records(self, data, record_file, resize=None, bulk=False, policy='desc'):
    #     with tf.io.TFRecordWriter(record_file) as writer:
    #
    #         if policy == 'desc':
    #             examples_generator = self._video_generator(data, resize, bulk)
    #         elif policy == ''
    #             examples_generator = self._video_generator(data, resize,
    #                                                        bulk) if not policy == 'pretrain' else self._image_generator(
    #                 data,
    #                 resize)
    #         for tf_example in examples_generator:
    #             writer.write(tf_example.SerializeToString())


if __name__ == '__main__':

    # data = 'phase2'
    # split = 0
    # video_fp = None
    #
    # ds = Dataset()
    # if data == 'phase1':
    #     desc_fp = Datasets.phase1['descriptors']
    #     video_fp = Datasets.phase1['videos']
    #     motion_fp = Datasets.phase1['motion_capture']
    #     train_record_fp = Datasets.phase1['train_records']
    #     validation_record_fp = Datasets.phase1['validation_records']
    #     test_record_fp = Datasets.phase2['test_records']
    #     ds.get_descriptor_data(desc_fp)
    #
    # elif data == 'phase2':
    #     desc_fp = Datasets.phase2['descriptors']
    #     video_fp = Datasets.phase2['videos']
    #     motion_fp = Datasets.phase2['motion_capture']
    #     train_record_fp = Datasets.phase2['train_records']
    #     validation_record_fp = Datasets.phase2['validation_records']
    #     test_record_fp = Datasets.phase2['test_records']
    #     ds.get_descriptor_data(desc_fp)
    #
    # elif data == 'pretrain':
    #     desc_fp = Datasets.KDEF['data']
    #     train_record_fp = Datasets.KDEF['train_records']
    #     validation_record_fp = Datasets.KDEF['validation_records']
    #     test_record_fp = Datasets.KDEF['test_records']
    #     ds.get_pretrain_data()
    #
    # else:
    #     raise ValueError()
    #
    # # split
    # if split == 0:
    #     splits = [None, None, ds.split(0)]
    # elif 0 < split < 1:
    #     splits = [*ds.split(split), None]
    # else:
    #     raise ValueError
    #
    # tfd = TfRecordData(video_fp)
    # for _split, _data, _fp in zip(['train', 'validation', 'test'], splits,
    #                               [train_record_fp, validation_record_fp, test_record_fp]):
    #     if _data is None:
    #         print('Skipping {}. No data.'.format(_split))
    #         continue
    #     print('Processing {} dataset. Output will be saved to {}'.format(_split, _fp))
    #     tfd.to_records(_data, _fp, resize=(600, 600), policy=data)

    split = 0.2
    desc_fp = Datasets.phase1['descriptors']
    video_fp = Datasets.phase1['videos']
    audios_fp = Datasets.phase1['audios']
    motion_fp = Datasets.phase1['motion_capture']

    train_record_fp = Datasets.phase1['all_train_records']
    validation_record_fp = Datasets.phase1['all_validation_records']
    test_record_fp = Datasets.phase1['all_test_records']

    ds = Dataset()
    ds.get_descriptor_data(desc_fp)
    splits = [*ds.split(split), None]

    tfd = TfRecordData(video_fp, motion_fp, audios_fp)
    for _split, _data, _fp in zip(['train', 'validation', 'test'], splits,
                                  [train_record_fp, validation_record_fp, test_record_fp]):
        if _data is None:
            print('Skipping {}. No data.'.format(_split))
            continue
        print('Processing {} dataset. Output will be saved to {}'.format(_split, _fp))
        tfd.to_records_all(_data, _fp, resize=(400, 400))
