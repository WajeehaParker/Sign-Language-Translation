import os
import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from file_helper import read_csv
from file_helper import create_frames_from_videos

# Function to extract frames and predict using the trained model
def predict_video(video_path, model, label_encoder):
    # Initialize data list for prediction
    prediction_data = []
    video_name = video_path.split('/')[-1].split('\\')[-1].split('.')[0]

    video_frames_path = os.path.join("output/hand-detection-frames", video_name)
    for frame_name in os.listdir(video_frames_path):
        if frame_name.lower().endswith('.jpg'):
            frame_path = os.path.join(video_frames_path, frame_name)
            img = cv2.imread(frame_path)
            img = cv2.resize(img, (50, 50))
            prediction_data.append(img)

    # Convert prediction data to numpy array
    prediction_data = np.array(prediction_data)

    # Make predictions using the model
    predictions = model.predict(prediction_data)

    # Decode the predictions using the label encoder
    decoded_predictions = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    return decoded_predictions

# Load the trained model and label encoder
trained_model = load_model('output/hand_action_model.h5')
loaded_label_encoder = LabelEncoder()
loaded_label_encoder.classes_ = np.load('output/label_encoder.npy')
xml_path = "rwth-boston-104/handpositions/boston-hand-data.xml"
output_root_folder = 'output/hand-detection-frames'

# Video Input from User
video_path = input("Enter video path: ")
print("Video Path: "+ str(video_path))
#video_path = 'rwth-boston-104/videoBank/camera0/065_0'

video_name = video_path.split('/')[-1].split('\\')[-1].split('.')[0]
print("Video Name: "+ str(video_name))
output_folder = os.path.join(output_root_folder, video_name)
print("Output Folder: "+ str(output_folder))

# Process the video
create_frames_from_videos(video_path, xml_path, output_folder, video_name.split('_')[0])
predictions = predict_video(video_path, trained_model, loaded_label_encoder)

# Print the predictions
print("Predictions for the video:")
print(predictions[0])
