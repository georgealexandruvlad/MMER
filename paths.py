import os

# BASE_PATH = os.path.dirname(__file__)
BASE_PATH = "gs://multimodal-emotion-recognition"

DATA_PATH = os.path.join(BASE_PATH, 'datasets')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
MODEL_PATH = os.path.join(BASE_PATH, 'checkpoints')
VGGFACE_DATA_PATH = os.path.join('D://test')
XML_PATH = os.path.join(BASE_PATH, 'utils', 'haarcascade_frontalface_default.xml')


class Datasets:
    phase1 = {
        'descriptors': os.path.join(DATA_PATH, 'phase_1', 'descriptors'),
        'motion_capture': os.path.join(DATA_PATH, 'phase_1', 'motion capture'),
        'videos': os.path.join(DATA_PATH, 'phase_1', 'videos'),
        'audios': os.path.join(DATA_PATH, 'phase_1', 'audios'),
        'train_desc': os.path.join(DATA_PATH, 'desc_train_phase_1.json'),
        'validation_desc': os.path.join(PROCESSED_PATH, 'desc_validation_phase_1.json'),
        'train_records': os.path.join(PROCESSED_PATH, 'train.tfrecords'),
        'validation_records': os.path.join(PROCESSED_PATH, 'validation.tfrecords'),
        'all_train_records': os.path.join(PROCESSED_PATH, 'all_train.tfrecords'),
        'all_validation_records': os.path.join(PROCESSED_PATH, 'all_validation.tfrecords'),
        'all_test_records': os.path.join(PROCESSED_PATH, 'all_test.tfrecords'),
    }

    phase2 = {
        'descriptors': os.path.join(DATA_PATH, 'phase_2', 'descriptors'),
        'motion_capture': os.path.join(DATA_PATH, 'phase_2', 'motion capture'),
        'videos': os.path.join(DATA_PATH, 'phase_2', 'videos'),
        'train_desc': os.path.join(PROCESSED_PATH, 'desc_train_phase_2.json'),
        'validation_desc': os.path.join(PROCESSED_PATH, 'desc_validation_phase_2.json'),
        'train_records': os.path.join(PROCESSED_PATH, 'train.tfrecords'),
        'validation_records': os.path.join(PROCESSED_PATH, 'validation.tfrecords'),
        'test_records': os.path.join(PROCESSED_PATH, 'test.tfrecords'),
    }

    KDEF = {
        'data': os.path.join(DATA_PATH, 'KDEF_and_AKDEF', 'KDEF'),
        'train_records': os.path.join(PROCESSED_PATH, 'train_pretrain.tfrecords'),
        'validation_records': os.path.join(PROCESSED_PATH, 'validation_pretrain.tfrecords'),
        'test_records': os.path.join(PROCESSED_PATH, 'test_pretrain.tfrecords'),

    }
