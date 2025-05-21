# People ID

# For each newly detected pedestrian, calculate the center point (centerX, centerY) to compare with tracked positions.
# Use person_tracker to store previously detected pedestrian positions and center points.
# For each new pedestrian, calculate the distance (dist) to all tracked pedestrians.

# If dist is less than the threshold min_distance (60 pixels), consider it the same pedestrian 
# and update their ID and position; each frame updates and records the pedestrian's new position in person_tracker.

# Assign a New ID:
# If no match is found (all distances > min_distance), assign a new ID and add this pedestrian to person_tracker.

#-------------------------------------------------------------------------------------------#

# Cloest People

# Sort persons by bounding box area (width * height) in descending order,
# assuming that larger areas indicate closer proximity to the camera.
# Select the top three persons (those with the largest bounding box areas)
# as the closest to the camera.
# Draw a rectangle and label each as "Closest 1", "Closest 2", or "Closest 3"

import cv2
import numpy as np
import sys

# Task 1: Background Subtraction and Object Detection
def task_one(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Unable to open video file")
        return
    
    # Initialize background subtractor for background modeling
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        
        # Generate foreground mask
        fg_mask = bg_subtractor.apply(frame)
        background_frame = bg_subtractor.getBackgroundImage()
        
        # Use morphological operations to remove noise (only on lower-left mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Use connected component analysis to separate objects
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask_cleaned)
        
        # Create a blank image only showing foreground objects
        detected_objects = np.zeros_like(frame)
        person_count, car_count, other_count = 0, 0, 0
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 500:
                continue  # Ignore small objects
            
            # Classify based on aspect ratio
            aspect_ratio = w / float(h)
            if 0.4 < aspect_ratio < 1.5 and h > 50:
                person_count += 1
                label = "Person"
            elif aspect_ratio > 1.5:
                car_count += 1
                label = "Car"
            else:
                other_count += 1
                label = "Other"
            
            # Copy each connected component to detected_objects, retaining original color
            mask = np.zeros_like(fg_mask_cleaned)
            mask[labels == i] = 255  # Only show current connected component
            colored_component = cv2.bitwise_and(frame, frame, mask=mask)
            detected_objects = cv2.add(detected_objects, colored_component)
        
        # Combine different views into a single window
        combined_window = np.zeros((960, 1280, 3), dtype=np.uint8)
        
        # Place different images in four quadrants of a single window
        combined_window[0:480, 0:640] = frame  # Top-left: Original frame
        if background_frame is not None:
            combined_window[0:480, 640:1280] = cv2.resize(background_frame, (640, 480))  # Top-right: Background frame (no morphology applied)
        combined_window[480:960, 0:640] = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)  # Bottom-left: Foreground mask with noise removed
        combined_window[480:960, 640:1280] = detected_objects  # Bottom-right: Color foreground objects only
        
        # Display combined window
        cv2.imshow("Task1", combined_window)
        
        # Output object count in command line
        frame_count += 1
        print(f"Frame {frame_count:04d}: {num_labels - 1} objects ({person_count} persons, {car_count} cars, {other_count} others)")
        
        # Press 'q' to exit
        if cv2.waitKey(16) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Task 2: Pedestrian Detection and Tracking
def task_two(video_file):
    # Load DNN model and class labels
    model_weights = "frozen_inference_graph.pb"
    model_config = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt"
    
    # Load DNN model
    net = cv2.dnn.readNet(model_weights, model_config, framework='Tensorflow')
    
    # Set pedestrian class ID (MS COCO's ID 1 is "person")
    person_class_id = 1

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Unable to open video file")
        return

    frame_count = 0
    person_tracker = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        (h, w) = frame.shape[:2]

        # Step 1: Original frame
        original_frame = frame.copy()
        
        # Step 2: Detect pedestrians and overlay bounding boxes
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        net.setInput(blob)
        detections = net.forward()

        detected_frame = frame.copy()
        labeled_frame = frame.copy()
        closest_frame = frame.copy()

        # Store pedestrian information (label and distance estimation)
        persons = []
        boxes = []
        confidences = []
        ids = []

        # Process each detection result
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.25:  # Lower confidence threshold slightly
                class_id = int(detections[0, 0, i, 1])
                if class_id == person_class_id:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    boxes.append([startX, startY, endX - startX, endY - startY])
                    confidences.append(float(confidence))
                    ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS) to remove overlapping detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.25, nms_threshold=0.3)

        # Check for any indices to avoid flatten error
        if len(indices) > 0:
            indices = indices.flatten()
            
            # Define max width and height thresholds for bounding boxes
            MAX_WIDTH = 200
            MAX_HEIGHT = 400

            for i in indices:  # Iterate over each valid index
                startX, startY, width, height = boxes[i]
                endX, endY = startX + width, startY + height
                centerX, centerY = (startX + endX) // 2, (startY + endY) // 2  # Calculate center point
                distance_estimate = endY - startY

                # Check if pedestrian is already tracked
                found_id = None
                min_distance = 60  # Set a distance threshold

                # Ignore overly large bounding boxes
                if width > MAX_WIDTH or height > MAX_HEIGHT:
                    continue

                for pid, (prevX, prevY, _, _, prev_centerX, prev_centerY) in person_tracker.items():
                    dist = np.sqrt((centerX - prev_centerX) ** 2 + (centerY - prev_centerY) ** 2)
                    if dist < min_distance:
                        found_id = pid
                        person_tracker[pid] = (startX, startY, endX, endY, centerX, centerY)
                        break

                if found_id is None:
                    found_id = len(person_tracker) + 1
                    person_tracker[found_id] = (startX, startY, endX, endY, centerX, centerY)

                persons.append((found_id, startX, startY, endX, endY, distance_estimate))

                # Overlay bounding boxes (top-right area) without displaying ID label
                cv2.rectangle(detected_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Step 3: Label and track pedestrians
        labeled_frame = frame.copy()

        for (label_id, startX, startY, endX, endY, _) in persons:
            # Display pedestrian ID inside bounding box
            cv2.rectangle(labeled_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(labeled_frame, f"ID {label_id}", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Step 4: Select the 3 pedestrians closest to the camera
        persons.sort(key=lambda x: (x[3] - x[1]) * (x[4] - x[2]), reverse=True)
        for idx, (label_id, startX, startY, endX, endY, _) in enumerate(persons[:3]):
            cv2.rectangle(closest_frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(closest_frame, f"Closest {idx + 1}", (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Display four areas in the combined window
        combined_window = np.zeros((960, 1280, 3), dtype=np.uint8)
        combined_window[0:480, 0:640] = original_frame
        combined_window[0:480, 640:1280] = detected_frame
        combined_window[480:960, 0:640] = labeled_frame
        combined_window[480:960, 640:1280] = closest_frame

        cv2.imshow("Task2", combined_window)
        
        # Output the number of detected pedestrians per frame
        frame_count += 1
        print(f"Frame {frame_count:04d}: {len(persons)} detected persons")

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check command line arguments for mode and video file
    if len(sys.argv) != 3:
        print("Usage: python movingObj.py -b|-d video_file")
    else:
        mode = sys.argv[1]
        video_file = sys.argv[2]
        
        if mode == "-b":
            task_one(video_file)
        elif mode == "-d":
            task_two(video_file)
        else:
            print("Invalid mode. Use -b for Task One or -d for Task Two.")
