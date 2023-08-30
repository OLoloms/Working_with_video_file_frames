import cv2
import librosa
import numpy as np
import matplotlib.pyplot as plt


from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip

from PIL import Image

default_save_path = ''

def saveFrames(pathToVideo, frame_number, num_frames_to_save, save_path=default_save_path):

    cap = cv2.VideoCapture(pathToVideo)
    
    start_frame = max(0, frame_number)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # достаём аудиодорожку с видеофайла
    video_clip = VideoFileClip(pathToVideo)
    audio_file = video_clip.audio


    # определяем количество FPS и соответственно длительность одного кадра
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_duration = 1.0 / fps

    frame_count = 0
    
    noise_value = []

    
    while frame_count < num_frames_to_save:
    
        ret, frame = cap.read()
        
        if not ret:
            break
        #

        frame_start_time = (frame_number + frame_count) * frame_duration
        frame_end_time = (frame_number + frame_count + 1) * frame_duration

        audio_clip = audio_file.subclip(frame_start_time, frame_end_time)
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
    
    with open('{}result.txt'.format(save_path), 'w') as file:
        
        for index in range(noise_value_length):
            file.write('Frame - {}: {}\n'.format(index, noise_value[index]))
        #
    #
    
    # Визуализируем данные
    
    colors = ['red' if value < 0.55 else 'green' for value in noise_value]
    
    #
    
    for i in range(num_frames_to_save):
        
        plt.plot([i , i], [0, noise_value[i]], color='gray', linestyle='--', linewidth=1)
        plt.scatter(i, noise_value[i], color='blue')
    #
    
    plt.axhline(y=0.55, color='red', linestyle='-', label='Noise_line')
    plt.ylim(0.0, 1.0)
    plt.title('Graph')
    plt.xlabel('Frame')
    plt.ylabel('Noise_value')
    plt.legend()
    plt.xticks(range(num_frames_to_save), rotation=45)
    plt.grid()

    plt.savefig('{}Graph.png'.format(save_path))


    
save_path = ''

pathToVideo = ''

saveFrames(pathToVideo, 5820, 30, save_path)




