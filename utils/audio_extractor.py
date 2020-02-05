from paths import Datasets
import os
import subprocess

video_dir = Datasets.phase1['videos']
audio_dir = Datasets.phase1['audios']

if __name__ == '__main__':
    print(video_dir)
    for file in os.listdir(video_dir):
        wav_file_name = file.split('.')[0] + '.wav'
        command = "ffmpeg -i " + video_dir + '\\' + file + " -ab 160k -ac 2 -ar 44100 -vn " + audio_dir + '\\' + wav_file_name
        subprocess.call(command, shell=True)
