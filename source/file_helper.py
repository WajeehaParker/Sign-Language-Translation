import xml.etree.ElementTree as ET
import numpy as np
import csv
import cv2
import os

def extract_features_from_bbox(frame, coord, box_size=25):
    features_list = []

    # Extract features for each point in the frame
    for point in coord:
        x, y = point['x'], point['y']

        # Extract a box of size 25 around the point
        x_start = max(0, x - box_size // 2)
        x_end = min(frame.shape[1], x + box_size // 2)
        y_start = max(0, y - box_size // 2)
        y_end = min(frame.shape[0], y + box_size // 2)

        # Extract features from the bounding box
        frame_roi = frame[y_start:y_end, x_start:x_end]

        # Example: mean and standard deviation of pixel values
        mean_pixel_value = np.mean(frame_roi)
        std_pixel_value = np.std(frame_roi)

        # Append the features to the list
        features_list.extend([mean_pixel_value, std_pixel_value])

    return features_list

def mark_hands_and_face(frame, coord):
    # Convert the BGR image to RGB
    #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    box_size = 25
    features_list = []

    # Extract features for each bounding box
    for point in coord:
        x, y = point['x'], point['y']

        # Extract a box around the point
        x_start = max(0, x - box_size // 2)
        x_end = min(frame.shape[1], x + box_size // 2)
        y_start = max(0, y - box_size // 2)
        y_end = min(frame.shape[0], y + box_size // 2)

        # Draw bounding box
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

        # Extract features from the bounding box
        features = extract_features_from_bbox(frame, [point], box_size)

        # Append features to the overall list
        features_list.extend(features)

    return frame, features_list

def create_frames_from_videos(video_path, xml_path, output_folder, video_name):
    # Read the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    # Fetch coordinates
    coordinates_dict = getCoordinatesFromXML(xml_path, video_name)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Detect hands and draw bounding boxes
        frame_with_boxes, features = mark_hands_and_face(frame, coordinates_dict[frame_count])

        # Save the frame with bounding boxes
        frame_filename = f"frame_{frame_count}.jpg"
        frame_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(frame_path, frame_with_boxes)

        features_filename = f"features_{frame_count}.txt"
        features_path = os.path.join(output_folder, features_filename)
        with open(features_path, 'w') as file:
            file.write(','.join(map(str, features)))

        frame_count += 1

    # Release the video capture object
    cap.release()
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

def getCoordinatesFromXML(xml_path, video_name):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    coordinates_dict = {}

    # Find the video element with the specified name
    video_element = root.find(f'./video[@name="{video_name}"]')

    if video_element is not None:
        # Iterate over frames in the video
        for frame_element in video_element.iter('frame'):
            frame_number = int(frame_element.get('number'))
            
            # Extract coordinates for each point in the frame
            coordinates = [
                {
                    'n': int(point.get('n')),
                    'x': int(point.get('x')),
                    'y': int(point.get('y'))
                }
                for point in frame_element.iter('point')
            ]

            # Store the coordinates in the dictionary
            coordinates_dict[frame_number] = coordinates

    return coordinates_dict


def read_csv(csv_path):
    data_dict = {}
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            video_name = row['name']
            orth = row['orth']
            translation = row['translation']
            # Create a dictionary entry with video name as key
            data_dict[video_name] = {'orth': orth, 'translation': translation}
    return data_dict