import cv2
from ultralytics import YOLO
import time
import numpy as np
import torch

def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.cpu().numpy().astype(int)
    else:
        return arr.astype(int)

# Load YOLOv8 model
model = YOLO("my_model.pt")

# Class indices
WORKER_CLASS = 1
PALLET_CLASS = 0

# Load video
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame of the video.")
    exit()

height, width = frame.shape[:2]
output_path = "output3.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # pyright: ignore[reportAttributeAccessIssue]
out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

# Tracking data
worker_time = 0.0  # total seconds worker is present
pallet_ids = set()
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
frame_idx = 1  # already read first frame

while True:
    # Run YOLO with ByteTrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    boxes = results[0].boxes
    worker_detected = False
    if boxes is not None and boxes.id is not None:
        ids = to_numpy(boxes.id)
        clss = to_numpy(boxes.cls)
        xyxy = to_numpy(boxes.xyxy)
        for i, (box_id, cls, box) in enumerate(zip(ids, clss, xyxy)):
            label = model.names[cls]
            if cls == WORKER_CLASS:
                worker_detected = True
                color = (0,255,0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                # No ID or label for worker
            elif cls == PALLET_CLASS:
                color = (255,0,0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, f"{label} ID:{box_id}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if box_id not in pallet_ids:
                    pallet_ids.add(box_id)
    if worker_detected:
        worker_time += 1.0 / fps
    # Overlay stats
    cv2.putText(frame, f"Unique Pallets: {len(pallet_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, f"Worker: {worker_time:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    out.write(frame)
    # Read next frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Print summary
print("\n--- SUMMARY ---")
print(f"Total unique pallets: {len(pallet_ids)}")
print(f"Worker: {worker_time:.1f} seconds present") 