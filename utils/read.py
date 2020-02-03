import cv2
from os import listdir
from paths import *
import numpy as np
import tensorflow as tf
from utils.face_detector import detect_face
import wave
from utils.wavfilehelper import WavFileHelper, extract_features


def get_frames_all_videos(directory):
    video_files = get_files_from_folder(directory)
    videos = {os.path.basename(v_file): get_frames_from_video(v_file) for v_file in video_files}
    return videos


def read_img(src, resize=None):
    img = cv2.imread(src)
    if resize is not None and (resize[0] < img.shape[0] and resize[1] < img.shape[1]):
        return tf.image.resize(img, resize, tf.image.ResizeMethod.BILINEAR,
                               preserve_aspect_ratio=False).numpy().astype(np.uint8)
    return img.astype(np.uint8)


def get_frames_from_motion(src):
    with open(src, "r", encoding='utf-8') as f:
        lines = np.array([np.array([float(elem) for elem in line.rstrip().split(';')[:-1]], dtype=np.float) for line in
                          f.readlines()[1:]])
        return lines


def get_wav_from_audio(src, time_start, time_end):
    win = wave.open(src, 'rb')
    wavfilehelper = WavFileHelper()
    temp_paths = src.rsplit('\\', 1)

    temp_cut_path = temp_paths[0] + '\\' + 'temp_' + temp_paths[1]
    wout = wave.open(temp_cut_path, 'wb')

    s0, s1 = int(time_start * win.getframerate()), int(time_end * win.getframerate())
    win.readframes(s0)  # discard
    frames = win.readframes(s1 - s0)

    wout.setparams(win.getparams())
    wout.writeframes(frames)

    # data = wavfilehelper.read_file_properties(temp_cut_path)
    data_features = extract_features(temp_cut_path)

    # audiodata.append(data)

    win.close()
    wout.close()
    os.remove(temp_cut_path)

    return data_features


def get_frames_from_video(src, resize=None, frames_interval=None, face_area_detector_size=None):
    vidcap = cv2.VideoCapture(src)
    xml_abs_path = os.path.abspath(XML_PATH)
    face_cascade = cv2.CascadeClassifier(xml_abs_path)
    frames = []
    start, end = 0, 0
    frame_index = 0

    if frames_interval:
        start, end = frames_interval

    while vidcap.isOpened():
        success, frame = vidcap.read()
        if success:
            if frames_interval is not None:
                if frame_index < start:
                    frame_index += 1
                    continue
                elif frame_index > end:
                    break

            if resize:
                frame = tf.image.resize(frame, resize, tf.image.ResizeMethod.AREA,
                                        preserve_aspect_ratio=False, antialias=True).numpy().astype(np.uint8)
            if face_area_detector_size is not None:
                frame = detect_face(frame, face_area_detector_size, face_cascade)

            frames.append(frame)
            frame_index += 1
        else:
            break
    vidcap.release()
    return np.array(frames)


def display_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)


def walk_frames(frames):
    frame_index = 0

    def incr(i):
        if i == len(frames) - 1:
            return i
        return i + 1

    def decr(i):
        if i == 0:
            return i
        return i - 1

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    while True:

        key = cv2.waitKey(0) & 0xFF
        character = chr(int(key))

        if character == 'a':
            frame_index = decr(frame_index)
        elif character == 'd':
            frame_index = incr(frame_index)
        elif character == 'q':
            break
        cv2.imshow('image', frames[frame_index])
    cv2.destroyAllWindows()


def get_files_from_folder(directory):
    filenames = [os.path.join(directory, f) for f in listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return filenames


# def read()

if __name__ == '__main__':
    # for frames
    _files_video = get_files_from_folder(Datasets.phase1['videos'])
    _files_motion = get_files_from_folder(Datasets.phase1['motion_capture'])

    _frames = get_frames_from_video(_files_video[20], (600, 600), frames_interval=(20, 100))
    print(_frames[0].shape)
    walk_frames(_frames)

    # for images
    # img = read_img(os.path.join(Datasets.KDEF['data'], 'AF01', 'AF01AFS.jpg'), (400, 400))
    # print(img.shape)
    # display_frame(img)
