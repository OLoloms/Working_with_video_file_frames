import cv2
import librosa
import numpy as np


from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip

from PIL import Image


global_save_path = '.\\Frame\\'


def saveFrames(pathToVideo, frame_number, num_frames_to_save, save_path=global_save_path):

    cap = cv2.VideoCapture(pathToVideo)
    
    start_frame = max(0, frame_number)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # достаём аудиодорожку с видеофайла
    video_clip = VideoFileClip(pathToVideo)
    audio_file = video_clip.audio
    audio_file.write_audiofile('audio.wav')


    # определяем количество FPS и соответственно длительность одного кадра
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_duration = 1.0 / fps

    frame_count = frame_number
    
    temporary_audio_path = 'test.wav'

    noise_value = []

    
    while frame_count < frame_number + num_frames_to_save:
    
        ret, frame = cap.read()

        
        if not ret:
            break
        #

        frame_start_time = frame_count * frame_duration
        frame_end_time = (frame_count + 1) * frame_duration

        audio_clip = AudioFileClip('audio.wav').subclip(frame_start_time, frame_end_time)
        #audio_clip.write_audiofile(temporary_audio_path)
        audio_array = audio_clip.to_soundarray()


        # с использованием библиотеки librosa
        y = audio_array.mean(axis=1)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_mean_round = np.round(rms_mean*10, 3)
        noise_value.append(rms_mean_round)

       

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        scaled = cv2.resize(img, (640,480))
        
        gray_frame = Image.fromarray(scaled)
        gray_frame.save("{}{}.jpg".format(save_path, frame_count))
        
        frame_count += 1
    

    #
    noise_value_length = len(noise_value)
    with open(f'.\\Frame\\resultest.txt', 'w') as file:
        for index in range(noise_value_length):
            number_for_name = frame_number + index
            file.write('Frame - {}: {}\n'.format(number_for_name, noise_value[index]))          
    

    
global_save_path = '.\\Frame\\frame'

pathToVideo = "123.mp4"
pathToAudio = 'audio.wav'

saveFrames(pathToVideo, 5820, 10)




