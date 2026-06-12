from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/test.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "tracked_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.5
    )

    unique_ids = set()
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()

        for track_id in ids:
            unique_ids.add(int(track_id))
    print("Unique Objects Seen:", len(unique_ids))
    print("IDs:", sorted(unique_ids))
    
    annotated_frame = results[0].plot()

    out.write(annotated_frame)

cap.release()
out.release()

print("Tracking completed")