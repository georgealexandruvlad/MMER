from paths import Datasets
from collections import defaultdict
import matplotlib.pyplot as plt
from loaders.dataset import Dataset
import numpy as np


def no_frames_distribution(data):
    emotion_frames = defaultdict(int)
    emotion_times = defaultdict(int)
    frames = []
    times = []

    for d in data:
        frame_start = d['video']['frame_start']
        frame_end = d['video']['frame_end']
        time_start = d['video']['time_start']
        time_end = d['video']['time_end']
        emotion = d['emotion']

        delta_frames = frame_end - frame_start
        delta_time = time_end - time_start

        emotion_frames[emotion] += delta_frames
        emotion_times[emotion] += delta_time

        if delta_frames == 8:
            print(d['filename'])
            continue
        frames.append(delta_frames)
        times.append(delta_time)

    print('Min number of frames: {}'.format(np.array(frames).min()))
    print('Max number of frames: {}'.format(max(frames)))
    print('Mean number of frames: {}'.format(np.array(frames).mean()))
    print('Std number of frames: {}'.format(np.array(frames).std()))
    print('Frames per second: {}'.format((np.array(frames) / np.array(times)).mean()))

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    emotion_labels = sorted(emotion_frames.keys())

    # order is important
    axes[0, 0].bar(emotion_labels, [emotion_frames[key] for key in sorted(emotion_frames.keys(), reverse=True)])
    axes[0, 0].set_title('Frames per emotion')

    axes[0, 1].bar(emotion_labels, [emotion_frames[key] for key in sorted(emotion_times.keys(), reverse=True)])
    axes[0, 1].set_title('Time per emotion')

    axes[1, 0].plot(list(range(len(frames))), sorted(frames))
    axes[1, 0].set_title('Frames per sample')

    axes[1, 1].plot(list(range(len(times))), sorted(times))
    axes[1, 1].set_title('Seconds per sample')

    plt.show()


if __name__ == '__main__':
    desc_fp = Datasets.phase1['descriptors']
    data_desc = Dataset(desc_fp).get_descriptor_data()
    no_frames_distribution(data_desc)
