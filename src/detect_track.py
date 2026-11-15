"""Extract trajectories from a road video with YOLO + ByteTrack (ultralytics).

Usage:  pip install ultralytics
        python src/detect_track.py video.mp4 --out data/processed/video_tracks.csv

Output columns: frame, t, track_id, cls, x, y (pixel centre), w, h. Convert pixels to metres with a
homography/scale (--m-per-px) before computing TTC/PET/DRAC via features.py.
STATUS: written but NOT yet run - no road video was available in this environment.
"""
import argparse
import pandas as pd

COCO = {0: "pedestrian", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--out", default="data/processed/video_tracks.csv")
ap.add_argument("--model", default="yolov8n.pt"); ap.add_argument("--fps", type=float, default=30)
a = ap.parse_args()

from ultralytics import YOLO
model = YOLO(a.model)
rows = []
for i, r in enumerate(model.track(a.video, tracker="bytetrack.yaml", stream=True, classes=list(COCO), verbose=False)):
    if r.boxes.id is None:
        continue
    for (x, y, w, h), tid, c in zip(r.boxes.xywh.tolist(), r.boxes.id.int().tolist(), r.boxes.cls.int().tolist()):
        rows.append(dict(frame=i, t=i / a.fps, track_id=tid, cls=COCO[c], x=x, y=y, w=w, h=h))
pd.DataFrame(rows).to_csv(a.out, index=False)
print(f"wrote {len(rows)} detections to {a.out}")

