from ultralytics import YOLO

# Loading the model 
model = YOLO('yolov8n.pt')  # Here the .pt refers to the pytorch model file

# Run the model on an image
results = model(
    source='images/test.png',
    save=True,
    project='outputs',
    name='predictions',
    conf=0.5  # If below this threshold, the detection will be ignored.
)

counts = {}

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])  # Class index
        conf = float(box.conf[0])  # Confidence score

        label = model.names[cls]
        counts[label] = counts.get(label, 0) + 1
        print(
            f'Class: {label}',
            f'Confidence: {conf:.2f}'
        )

print('Detection completed. Results saved in the runs/detection/outputs/predictions directory.')