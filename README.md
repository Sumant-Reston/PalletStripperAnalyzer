# Pallet Stripper Analyzer

This project analyzes a video to detect and track pallets and workers using a YOLOv8 model with ByteTrack tracking. It outputs a processed video with overlays and prints a summary of unique pallets detected and the total time a worker is present.

## Features
- Detects and tracks pallets and workers in a video using a custom YOLOv8 model (`my_model.pt`).
- Tracks unique pallet IDs and the total time a worker is present in the video.
- Outputs a processed video (`output3.avi`) with bounding boxes and statistics overlays.
- Prints a summary of unique pallets and worker presence time at the end.

## Requirements
- Python 3.x
- [Ultralytics YOLO](https://docs.ultralytics.com/) (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- PyTorch (`pip install torch`)

## Usage
1. Place your YOLOv8 model weights as `my_model.pt` in the project directory.
2. Place your input video as `test.mp4` in the project directory.
3. Run the script:
   ```bash
   python modelvideochecker2.py
   ```
4. The processed video will be saved as `output3.avi`.
5. At the end, the script prints the total number of unique pallets detected and the total time a worker was present in the video.

## Notes
- The script expects class 0 to be "pallet" and class 1 to be "worker" in your YOLO model.
- You can adjust the video and model file names in the script as needed.
