import sys
import os
import subprocess
import shutil
from pathlib import Path
from random import randint
from contextlib import contextmanager
from tqdm import tqdm
import cv2
import insightface
from insightface.app import FaceAnalysis
from cv2 import imread, imwrite

@contextmanager
def suppress_output():
   with open(os.devnull, 'w') as devnull:
       old_stdout, old_stderr = sys.stdout, sys.stderr
       sys.stdout, sys.stderr = devnull, devnull
       try:
           yield
       finally:
           sys.stdout, sys.stderr = old_stdout, old_stderr

def initialize_faceanalysis_and_swapper() -> tuple[FaceAnalysis, insightface.model_zoo.model_zoo.INSwapper]:
   faceanalysis: FaceAnalysis = FaceAnalysis(name="buffalo_l")
   faceanalysis.prepare(ctx_id=0, det_size=(640, 640))
   swapper: insightface.model_zoo.model_zoo.INSwapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
   return faceanalysis, swapper

def get_video_fps(video_path: str) -> float:
   cap = cv2.VideoCapture(video_path)
   fps = cap.get(cv2.CAP_PROP_FPS)
   cap.release()
   return fps

def extract_frames(video_path: str):
   """Extract all frames to unprocessed_frames/"""
   os.makedirs('unprocessed_frames', exist_ok=True)
   subprocess.run([
       'ffmpeg', '-i', video_path,
       '-q:v', '2',
       'unprocessed_frames/frame_%04d.png'
   ], check=True, capture_output=True)

def extract_audio(video_path: str) -> str:
   """Extract audio to audio.aac"""
   subprocess.run([
       'ffmpeg', '-i', video_path,
       '-map', '0:a', '-acodec', 'copy',
       'audio.aac'
   ], check=True, capture_output=True)
   return 'audio.aac'

def reconstruct_video(fps: float, audio_path: str, output_path: str):
   """Combine processed frames with audio"""
   os.makedirs('processed_frames', exist_ok=True)
   subprocess.run([
       'ffmpeg', '-framerate', str(fps),
       '-i', 'processed_frames/frame_%04d.png',
       '-i', audio_path,
       '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
       '-c:a', 'copy', '-shortest',
       output_path
   ], check=True, capture_output=True)

def cleanup(audio_path: str):
   """Remove intermediate files"""
   shutil.rmtree('unprocessed_frames', ignore_errors=True)
   shutil.rmtree('processed_frames', ignore_errors=True)
   if os.path.exists(audio_path):
       os.remove(audio_path)

def kirkify_frame(frame_path: str, output_path: str, faceanalysis: FaceAnalysis, swapper: insightface.model_zoo.model_zoo.INSwapper, kirk_face):
   """Process a single frame"""
   img = imread(frame_path)
   faces = faceanalysis.get(img)
   
   if faces:  # Only process if faces detected
       
       res = img.copy()
       for face in faces:
           res = swapper.get(res, face, kirk_face, paste_back=True)
       
       imwrite(output_path, res)
   else:
       # Copy unchanged if no faces
       imwrite(output_path, img)


def process_all_frames(faceanalysis: FaceAnalysis, swapper: insightface.model_zoo.model_zoo.INSwapper):
    """Process each frame in unprocessed_frames/ with tqdm progress bar"""
    os.makedirs('processed_frames', exist_ok=True)
    frame_files = [f for f in sorted(os.listdir('unprocessed_frames')) if f.endswith('.png')]
    kirk = imread(f'kirks/kirk_{randint(0, 2)}.jpg')
    kirk_face = faceanalysis.get(kirk)[0]
    # tqdm progress bar
    for filename in tqdm(frame_files, desc="Processing frames", unit="frame"):
        input_path = f'unprocessed_frames/{filename}'
        output_path = f'processed_frames/{filename}'
        kirkify_frame(input_path, output_path, faceanalysis, swapper, kirk_face)

def main():
   if len(sys.argv) > 1 and sys.argv[1] == "init":
       with suppress_output():
           faceanalysis, swapper = initialize_faceanalysis_and_swapper()
       print("initialized!!")
       exit()

   if len(sys.argv) != 3:
       print("Usage: python script.py <input_video> <output_video>")
       sys.exit(1)

   TARGET_PATH = sys.argv[1]
   OUTPUT_PATH = sys.argv[2]

   if not Path(TARGET_PATH).exists():
       print("ERROR: target path not real")
       sys.exit(1)

   # Initialize models
   #with suppress_output():
   faceanalysis, swapper = initialize_faceanalysis_and_swapper()

   print("Extracting frames...")
   extract_frames(TARGET_PATH)
   
   print("Extracting audio...")
   audio_path = extract_audio(TARGET_PATH)
   
   print("Processing frames...")
   process_all_frames(faceanalysis, swapper)
   
   print("Reconstructing video...")
   fps = get_video_fps(TARGET_PATH)
   reconstruct_video(fps, audio_path, OUTPUT_PATH)
   
   print("Cleaning up...")
   cleanup(audio_path)
   
   print(f"Done! Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
   main()