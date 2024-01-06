import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from file_helper import read_csv

csv_path = 'rwth-boston-104/corpus/train.sentences.pronunciations.multi.translations.csv'
csv_data = read_csv(csv_path)

# Set the path to the folder containing extracted frames
frames_folder = 'output/hand-detection-frames'

# Define the input shape based on your bounding box size
input_shape = (50, 50, 3)

# Initialize data and labels lists
data = []
labels = []

# Iterate over frames and extract features
for video_name in csv_data.keys():
    video_path = os.path.join(frames_folder, video_name+"_0")
    for frame_name in os.listdir(video_path):
        if frame_name.lower().endswith('.jpg'):
            frame_path = os.path.join(video_path, frame_name)
            # Load the frame
            img = cv2.imread(frame_path)
            img = cv2.resize(img, (input_shape[0], input_shape[1]))
            # Append the image data and label
            data.append(img)
            labels.append(csv_data[video_name]['translation'])

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels = to_categorical(encoded_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the trained model
print("Saving model")
model.save('output/hand_action_model.h5')

# Save label encoder for later use
print("Saving label encoder")
np.save('output/label_encoder.npy', label_encoder.classes_)
