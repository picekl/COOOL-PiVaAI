import numpy as np
import ruptures as rpt
import pandas as pd
import cv2
import os
from itertools import zip_longest
from pathlib import Path
from PIL import Image
from clip_interrogator import Config, Interrogator

from experiment.ioutils import load_frame_annotations, load_video_frames
from experiment.driver_state import bbox_size_drive_state_changed, calculate_bbox_size_total, baseline_driver_state_changed, optical_flow_driver_state_changed

import warnings

warnings.filterwarnings("ignore")


ci = Interrogator(
    Config(
        clip_model_name="ViT-L-14/openai",
        quiet=True,
    )
)

def baseline_get_hazard(
    frame_image: np.ndarray,
    bbox_centers: np.ndarray,
    track_ids: list[int],
    chips: list[np.ndarray],
    captioned_tracks: dict
) -> list[tuple]:
    image_center = [frame_image.shape[1]/2, frame_image.shape[0]/2]
    if bbox_centers.shape[0] == 0:
        return [(" ", " "), ]
    potential_hazard_dists = np.linalg.norm(bbox_centers - image_center, axis=1)
    probable_hazard = np.argmin(potential_hazard_dists)
    hazard_track = track_ids[probable_hazard]

    if hazard_track not in captioned_tracks:
        hazard_chip = cv2.cvtColor(chips[probable_hazard], cv2.COLOR_BGR2RGB)
        hazard_chip = Image.fromarray(hazard_chip)
        # Generate caption
        # hazard_caption = ci.interrogate(hazard_chip)
        hazard_caption = " "

        hazard_caption = hazard_caption.replace(","," ")

        captioned_tracks[hazard_track] = hazard_caption
    else:
        hazard_caption = captioned_tracks[hazard_track]#Why compute caption again? Re-use hazard desc

    return [(hazard_track, hazard_caption), ]


def get_hazards(
    frame_image: np.ndarray,
    bbox_centers: np.ndarray,
    track_ids: list[int],
    chips: list[np.ndarray],
    captioned_tracks: dict,
    num_tracks: int = 100,
    run_captions: bool = False,
) -> list[tuple]:
    hazards = []
    image_center = [frame_image.shape[1]/2, frame_image.shape[0]/2]
    if bbox_centers.shape[0] == 0:
        return [(" ", " "), ]
    potential_hazard_dists = np.linalg.norm(bbox_centers - image_center, axis=1)

    bbox_count = bbox_centers.shape[0]
    num_tracks = min(num_tracks, bbox_count)
    probable_hazard_indexes = np.argsort(potential_hazard_dists)[:num_tracks]

    for probable_hazard_idx in probable_hazard_indexes:
        hazard_track = track_ids[probable_hazard_idx]

        if hazard_track not in captioned_tracks:
            if run_captions:
                hazard_chip = cv2.cvtColor(chips[probable_hazard_idx], cv2.COLOR_BGR2RGB)
                hazard_chip = Image.fromarray(hazard_chip)
                # Generate caption
                hazard_caption = ci.interrogate(hazard_chip)
            else:
                hazard_caption = " "

            hazard_caption = hazard_caption.replace(","," ")

            captioned_tracks[hazard_track] = hazard_caption
        else:
            hazard_caption = captioned_tracks[hazard_track]#Why compute caption again? Re-use hazard desc

        hazards.append((hazard_track, hazard_caption))

    return hazards


def process_video(
    video_path: str,
    annotations: dict,
    run_tracks: bool = False,
    run_captions: bool = False,
):
    video_name = Path(video_path).stem
    video_data = load_video_frames(video_path)
    bbox_sizes, bbox_centers_all = [], []
    captioned_tracks = {}
    video_results = []

    for id_frame, frame_image in video_data:
        frame_number = int(id_frame.split("_")[-1])
        bboxes, bbox_centers, chips, track_ids = load_frame_annotations(annotations, video_name, frame_number, frame_image)

        bbox_pixel_size = calculate_bbox_size_total(bboxes)
        bbox_sizes.append(bbox_pixel_size)
        bbox_centers_all.append(bbox_centers)

        hazard_tracks, hazard_captions = [], []
        if run_tracks:
            hazards = get_hazards(frame_image, bbox_centers, track_ids, chips, captioned_tracks, run_captions=run_captions)
            for hazard_track, hazard_caption in hazards:
                hazard_tracks.append(hazard_track)
                hazard_captions.append(hazard_caption)

        video_results.append(
            {
                'id': id_frame,
                "bbox_size": bbox_pixel_size,
                "hazard_tracks": hazard_tracks,
                "hazard_captions": hazard_captions,
            }
        )

    driver_state_changed = bbox_size_drive_state_changed(bbox_sizes)
    # driver_state_changed = baseline_driver_state_changed(bbox_centers_all)
    # optical_flow = calculate_optical_flow(video_data)
    # driver_state_changed = optical_flow_driver_state_changed(optical_flow)
    assert len(video_results) == len(driver_state_changed)
    for video_result, driver_state in zip(video_results, driver_state_changed):
        video_result["driver_state"] = driver_state

    return  video_results


def video_results_to_submission_format(video_results: list[dict]) -> dict:
    submission_results = {}
    for results in video_results:
        id_frame = results["id"]
        driver_state = results["driver_state"]
        hazard_tracks = results["hazard_tracks"]
        hazard_captions = results["hazard_captions"]
        submission_results[id_frame] = {
            "Driver_State_Changed": driver_state,
        }

        for i, track, caption in zip_longest(range(23), hazard_tracks, hazard_captions, fillvalue=" "):
            if i is None:
                print("Should not happen")
                continue
            track_col = f"Hazard_Track_{i}"
            caption_col = f"Hazard_Name_{i}"
            submission_results[id_frame][track_col] = track
            submission_results[id_frame][caption_col] = caption
    return submission_results