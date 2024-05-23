import cv2
import numpy as np
import os
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

USE_GPU = False

def normalize_fps(path):
    abs_path = os.path.abspath(path)
    folder_path = os.path.join(os.path.dirname(abs_path), "fps")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    ffmpeg = 'ffmpeg -i "{}" -vf "fps=35" "{}"'

    if USE_GPU:
        ffmpeg = 'ffmpeg -hwaccel cuda -i "{}" -vf "fps=35" "{}"'

    new_name = os.path.join(folder_path, os.path.splitext(os.path.basename(abs_path))[0] + "_fps.mp4")
    command = ffmpeg.format(abs_path, new_name)

    os.system(command)

def join_video_audio(path):
    abs_path = os.path.abspath(path)
    fps_path = os.path.join(os.path.dirname(abs_path), "fps")
    final_path = os.path.join(os.path.dirname(abs_path), "final")

    if not os.path.exists(final_path):
        os.makedirs(final_path)

    ffmpeg = 'ffmpeg -i "{}" -i "{}" -c:v copy -c:a aac "{}"'

    audio = os.path.join(fps_path, os.path.splitext(os.path.basename(abs_path))[0].replace('_final', '') + ".mp4")
    video = os.path.join(fps_path, os.path.splitext(os.path.basename(abs_path))[0] + ".mp4")
    final = os.path.join(final_path, os.path.splitext(os.path.basename(abs_path))[0] + ".mp4")

    command = ffmpeg.format(video, audio, final)
    
    os.system(command)

def detect_icon_gpu(img_rgb, icon_path):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    template = cv2.imread(icon_path, 0)
 
    # Convertir la imagen de la plantilla a un objeto cv.cuda_GpuMat()
    template_gpu = cv2.cuda_GpuMat()
    template_gpu.upload(template)
 
    # Convertir la imagen de entrada a un objeto cv.cuda_GpuMat()
    img_gray_gpu = cv2.cuda_GpuMat()
    img_gray_gpu.upload(img_gray)
 
    # Escalar las imágenes
    scales = [0.7, 0.8, 0.9, 1.0]
    found_match = False
 
    for scale in scales:
        if found_match:
            break  
 
        scaled_template_gpu = cv2.cuda.resize(template_gpu, (0, 0), fx=scale, fy=scale)
        w, h = scaled_template_gpu.size()[::-1]
 
        matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_CCOEFF_NORMED)
        res_gpu = matcher.match(img_gray_gpu, scaled_template_gpu)
        res = res_gpu.download()     
 
        threshold = 0.8
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            mask_color = img_rgb[pt[1] + w - 1, pt[0]].tolist()
 
            cv2.rectangle(img_rgb, pt, (pt[0] + h, pt[1] + w), mask_color, 2)
            
            mask = np.zeros_like(img_gray)
            mask[pt[1]:pt[1]+w, pt[0]:pt[0]+h] = 255
            
            img_rgb = cv2.inpaint(img_rgb, mask, 5, cv2.INPAINT_TELEA)
 
            found_match = True
 
    return img_rgb

def detect_icon(img_rgb, icon_path):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    template = cv2.imread(icon_path, 0)
    
    scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    found_match = False

    for scale in scales:
        if found_match:
            break  

        scaled_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        w, h = scaled_template.shape[::-1]

        res = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            mask_color = img_rgb[pt[1] + h - 1, pt[0]].tolist()

            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), mask_color, 2)
            
            mask = np.zeros_like(img_gray)
            mask[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = 255
            
            img_rgb = cv2.inpaint(img_rgb, mask, 5, cv2.INPAINT_TELEA)

            found_match = True

    return img_rgb

def process_frame(frame_icon_path_tuple):
    frame, icon_path = frame_icon_path_tuple
    processed_frame = detect_icon_gpu(frame, icon_path) if USE_GPU else detect_icon(frame, icon_path)
    return processed_frame

def process_video_threads(args):
    video_path, icon_path, batch_size = args

    output_path = video_path.replace('.mp4', '_final.mp4')
    num_threads = 7
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"No se pudo abrir el archivo de video: {video_path}")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = 35
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        print("Procesando:", video_path)
        
        frames = []
        success, frame = cap.read()
        while success:
            frames.append(frame)
            success, frame = cap.read()

            if len(frames) == batch_size:
                with ThreadPoolExecutor() as executor:
                    for i in range(0, len(frames), num_threads):
                        group = frames[i:i+num_threads]
                        frame_icon_tuples = [(frame, icon_path) for frame in group]
                        processed_frames = executor.map(process_frame, frame_icon_tuples)
                        for processed_frame in processed_frames:
                            out.write(processed_frame)
                frames = []
        
        if len(frames) > 0:
            with ThreadPoolExecutor() as executor:
                for i in range(0, len(frames), num_threads):
                    group = frames[i:i+num_threads]
                    frame_icon_tuples = [(frame, icon_path) for frame in group]
                    processed_frames = executor.map(process_frame, frame_icon_tuples)
                    for processed_frame in processed_frames:
                        out.write(processed_frame)
            frames = []

        cap.release()
        out.release()
    except Exception as e:
        print(f"Error al procesar el video {video_path}: {e}")

def main(base_folder, icon_path, num_threads=4, batch_size=1000):
    fps_folder = '{}/fps'.format(base_folder)

    for video in os.listdir(base_folder):
        if video.endswith('.mp4'):
            normalize_fps(os.path.join(base_folder, video))

    videos = [
        (
            os.path.join(fps_folder, video), 
            icon_path
        ) for video in os.listdir(fps_folder) if video.endswith('.mp4') and '_fps.' in video
    ]

    pool = Pool()

    for i in range(0, len(videos), num_threads):
        group = videos[i:i+num_threads]
        pool.map(process_video_threads, [(video, icon, batch_size) for video, icon in group])

    pool.close()
    pool.join()

    for video in os.listdir(fps_folder):
        if video.endswith('.mp4') and '_fps_final' in video:
            join_video_audio(os.path.join(base_folder, video))

if __name__ == "__main__":
    base_folder = './videos'
    icon_path = './icon.png'
    num_threads = 2
    main(base_folder, icon_path, num_threads)