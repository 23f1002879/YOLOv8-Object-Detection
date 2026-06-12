from ultralytics import YOLO
import cv2 

# Loading the YOLO Model 
model = YOLO('yolov8n.pt')

# Opening the Video Files 
videoPath = 'videos/test.mp4'
cap = cv2.VideoCapture(videoPath)

# Defining the video properties 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Saving the output video
out = cv2.VideoWriter(
    'detected_video.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

frame_count = 0 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 
    # Running the YOLO model on the frame
    frame_count += 1
    results = model(frame, conf=0.5)
    annotated_frame = results[0].plot()

    counts = {}

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        counts[label] = counts.get(label, 0) + 1
      

    y_position = 30

    for label, count in counts.items():
        cv2.putText(
            annotated_frame,
            f"{label}: {count}",
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        print(f"{label}: {count}")

    y_position += 30

    out.write(annotated_frame)

    print(f'Processed frame {frame_count}')



cap.release()
out.release()

print('Video processing completed. Output saved as detected_video.mp4') 