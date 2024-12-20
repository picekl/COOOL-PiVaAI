import pickle
from pathlib import Path
import cv2
import numpy as np



def load_pickle(file_path: str) -> dict:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_frame_annotations(
    annotations: dict,
    video: str,
    frame: int,
    frame_image: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int]]:
    bboxes, bbox_centers, chips, track_ids = [], [], [], []
    for ann_type in ['challenge_object']:
        try:
            for i in range(len(annotations[video][frame][ann_type])):
                    x1, y1, x2, y2 = annotations[video][frame][ann_type][i]['bbox']
                    track_ids.append(annotations[video][frame][ann_type][i]['track_id'])
                    bboxes.append([x1, y1, x2, y2])
                    bbox_centers.append([x1 + (abs(x2 - x1) / 2), y1 + (abs(y2 - y1) / 2)])
                    chips.append(frame_image[int(y1):int(y2), int(x1):int(x2)])
        except Exception as e:
            print(f'KeyError: {video}_{frame}')
    bboxes, bbox_centers = np.array(bboxes), np.array(bbox_centers)
    return bboxes, bbox_centers, chips, track_ids


def load_video_frames(video_path: str) -> list:
    """Process a single video and return its frame data."""
    video_name = Path(video_path).stem
    video_data = []
    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        raise ValueError(f"Video {video_name} could not be opened.")

    frame = 0
    while video_stream.isOpened():
        ret, frame_image = video_stream.read()
        if not ret:
            break
        video_data.append((f"{video_name}_{frame}",  frame_image))
        frame += 1

    video_stream.release()
    return video_data
