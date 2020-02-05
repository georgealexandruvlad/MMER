import os
import json
import numpy as np
from utils.read import get_files_from_folder
from sklearn.model_selection import train_test_split
from metadata import *
from paths import Datasets
import random


class Dataset:

    def __init__(self):
        self.data = None

    def get_descriptor_data(self, src, dst=None):
        descriptor_files = get_files_from_folder(src)

        descriptor_data = []
        for file in descriptor_files:
            d = {}
            filename, _ = os.path.splitext(os.path.basename(file))
            _, meta1, meta2 = filename.split('_')
            d["filename"] = filename
            d["gender"] = meta1[0]
            d["actor_index"] = int(meta1[1])
            d["emotion"] = EMOTION_DICT[meta2[:2]]
            d["emotion_index"] = int(meta2[-1])

            with open(file, 'r') as fp:
                fp.readline()
                video = fp.readline().strip()
                start = fp.readline().split(':')
                end = fp.readline().split(':')

                time_start, frame_start = float(start[0]), int(start[1])
                time_end, frame_end = float(end[0]), int(end[1])
                fp.readline()
                kinect = fp.readline().strip()
                kinect_frame_start = int(fp.readline())
                kinect_frame_end = int(fp.readline())

            if frame_end - frame_start == 0:
                continue

            d['video'] = {'filename': video,
                          'time_start': time_start,
                          'time_end': time_end,
                          'frame_start': frame_start,
                          'frame_end': frame_end,
                          }
            d['kinect'] = {'filename': kinect,
                           'frame_start': kinect_frame_start,
                           'frame_end': kinect_frame_end
                           }
            descriptor_data.append(d)

        if dst:
            with open(dst, 'w', encoding='utf-8') as fp:
                json.dump(descriptor_data, fp, ensure_ascii=False, indent=4)
                print('Descriptor data successfully written to file {}'.format(dst))

        self.data = descriptor_data
        return descriptor_data

    def get_pretrain_data(self):
        dirs = [os.path.join(Datasets.KDEF['data'], subdir) for subdir in
                os.listdir(Datasets.KDEF['data']) if os.path.isdir(os.path.join(Datasets.KDEF['data'], subdir))]
        data = []

        for subdir in dirs:
            for fpath in get_files_from_folder(subdir):
                d = {}
                filename, _ = os.path.splitext(os.path.basename(fpath))
                emotion = filename[4: 6]
                d["image"] = fpath
                d['emotion'] = EMOTION_DICT_PRETRAIN[emotion]
                d["gender"] = filename[1]
                d["angle"] = filename[-2:]
                data.append(d)
        self.data = data
        return self.data

    def split(self, test_size=0.2):
        if self.data is None:
            raise ValueError('Descriptor data not yet loaded. Call function {get_descriptor_data | get_pretrain_data}')

        if test_size == 0:
            return self.data

        male, female = [], []
        for d in self.data:
            if d['gender'] == 'M':
                male.append(d)
            elif d['gender'] == 'F':
                female.append(d)

        indices_male = np.arange(len(male))
        indices_female = np.arange(len(female))

        labels_male = np.array([EMOTION_LABELS[d['emotion']] for d in male])
        labels_female = np.array([EMOTION_LABELS[d['emotion']] for d in female])

        x_train_male, x_test_male, y_train_male, y_test_male = train_test_split(
            indices_male, labels_male, test_size=test_size, random_state=42, stratify=labels_male, shuffle=True)

        x_train_female, x_test_female, y_train_female, y_test_female = train_test_split(
            indices_female, labels_female, test_size=test_size, random_state=42, stratify=labels_female, shuffle=True)

        unique, counts = np.unique(y_train_male, return_counts=True)
        print('Label male distribution train set:', dict(zip(unique, counts)))

        unique, counts = np.unique(y_train_female, return_counts=True)
        print('Label female distribution train set:', dict(zip(unique, counts)))

        unique, counts = np.unique(y_test_male, return_counts=True)
        print('Label male distribution test set:', dict(zip(unique, counts)))

        unique, counts = np.unique(y_test_female, return_counts=True)
        print('Label female distribution test set:', dict(zip(unique, counts)))

        train_data = [male[idx] for idx in x_train_male]
        train_data += [female[idx] for idx in x_train_female]

        validation_data = [male[idx] for idx in x_test_male]
        validation_data += [female[idx] for idx in x_test_female]
        random.Random(42).shuffle(train_data)
        random.Random(42).shuffle(validation_data)
        return train_data, validation_data


if __name__ == '__main__':
    ds = Dataset()
    ds.get_descriptor_data(Datasets.phase1['descriptors'])
    # ds.get_pretrain_data()

    print(len(ds.data))
    dm, df = ds.split(0.2)

    M, F, O = 0, 0, 0
    for d in dm:
        if d['gender'] == 'M':
            M += 1
        elif d['gender'] == 'F':
            F += 1
        else:
            O += 1
    print(M, F, O)

    M, F, O = 0, 0, 0
    for d in df:
        if d['gender'] == 'M':
            M += 1
        elif d['gender'] == 'F':
            F += 1
        else:
            O += 1
    print(M, F, O)
