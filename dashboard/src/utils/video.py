def process_video(video_path, model):
    import cv2
    import numpy as np

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection on the frame
        detections = model.predict(frame)
        
        # Draw bounding boxes and labels on the frame
        for detection in detections:
            x1, y1, x2, y2, class_id, confidence = detection
            color = get_color_for_class(class_id)
            label = f"Class: {class_id}, Conf: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frames.append(frame)

    cap.release()
    
    # Create a video writer to save the processed video
    output_path = video_path.replace('.mp4', '_processed.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        out.write(frame)

    out.release()
    return output_path

def get_color_for_class(class_id):
    # Define a color map for different classes
    colors = {
        0: (255, 0, 0),   # Class 0 - Red
        1: (0, 255, 0),   # Class 1 - Green
        2: (0, 0, 255),   # Class 2 - Blue
        # Add more classes and colors as needed
    }
    return colors.get(class_id, (255, 255, 255))  # Default to white if class_id not found