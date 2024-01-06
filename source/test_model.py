import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from file_helper import read_csv
from keras.utils import to_categorical

csv_path = 'rwth-boston-104/corpus/test.sentences.multi.translations.csv'
csv_data = read_csv(csv_path)

# Load the trained model
model = load_model('output/hand_action_model.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('output/label_encoder.npy')

# Set the path to the folder containing test frames
test_frames_folder = 'output/hand-detection-frames'

# Initialize data and labels lists for testing
test_data = []
test_labels = []

# Iterate over frames and extract features for testing
for video_name in csv_data.keys():
    video_path = os.path.join(test_frames_folder, video_name+"_0")
    for frame_name in os.listdir(video_path):
        # Check if the file is a .jpg image
        if frame_name.lower().endswith('.jpg'):
            frame_path = os.path.join(video_path, frame_name)
            # Load the frame
            img = cv2.imread(frame_path)
            img = cv2.resize(img, (50, 50))  # Resize the image
            # Get the label for the current frame
            label = csv_data[video_name]['translation']
            # Check if the label is present in the training set
            if label in label_encoder.classes_:
                # Append the image data and label for testing
                test_data.append(img)
                test_labels.append(label)

# Convert test data and labels to numpy arrays
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Encode test labels
encoded_test_labels = label_encoder.transform(test_labels)
encoded_test_labels = to_categorical(encoded_test_labels, num_classes=len(label_encoder.classes_), dtype='int')

# Evaluate the model on the test set
print("Testing the model")
loss, accuracy = model.evaluate(test_data, encoded_test_labels)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy * 100:.2f}%')
