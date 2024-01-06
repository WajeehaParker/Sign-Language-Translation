import os
from file_helper import create_frames_from_videos

print("Extracting frames from videos")

#video_path = 'rwth-boston-104/videoBank/camera0/001_0.mpg'
video_folder = 'rwth-boston-104/videoBank/camera0'
xml_path = "rwth-boston-104/handpositions/boston-hand-data.xml"
output_root_folder = 'output/hand-detection-frames'
os.makedirs(output_root_folder, exist_ok=True)

video_files = [f for f in os.listdir(video_folder) if f.endswith('.mpg')]

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)

    # Extract video name (assuming the video file name follows the format "001_0.mpg")
    video_name = video_file.split('.')[0]

    # Create output folder for each video
    output_folder = os.path.join(output_root_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)

    # Process the video
    create_frames_from_videos(video_path, xml_path, output_folder, video_name.split('_')[0])

print("Frames successfully ectracted from videos")