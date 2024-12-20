import numpy as np
import ruptures as rpt
import pandas as pd
import cv2
import os
from sklearn.linear_model import LinearRegression


def calculate_bbox_size_total(bboxes: np.ndarray[float]) -> float:
    size = 0
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        size += abs(x2 - x1) * abs(y2 - y1)
    return size


def calculate_optical_flow(
        video_data: list,
        motion_threshold: float = None,
):
    """
    Computes the degree of motion between the current and previous frames.

    Args:
        motion_threshold (float): Threshold to filter out low-motion noise.

    Returns:
        float: A single value representing the degree of motion in the frame.
    """

    optical_flow = []
    previous_frame = None
    for i, (id_frame, frame_image) in enumerate(video_data):
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        # frame_image_gpu = cv2.cuda.GpuMat()
        # frame_image_gpu.upload(frame_image)
        if previous_frame is None:
            previous_frame = frame_image
        # farneback = cv2.cuda_FarnebackOpticalFlow.create(
        #     numLevels=5, pyrScale=0.5, winSize=15, numIters=3, polyN=5, polySigma=1.2, flags=0
        # )
        #
        # # Compute optical flow on the GPU
        # gpu_flow = farneback.calc(previous_frame, frame_image_gpu, None)
        #
        # # Download the flow back to the CPU
        # flow = gpu_flow.download()
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev=previous_frame,
            next=frame_image,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Compute magnitude and angle of flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        # Filter out small magnitudes (noise) using a threshold
        if motion_threshold is not None:
            significant_motion = magnitude[magnitude > motion_threshold]
        else:
            significant_motion = magnitude

        # Calculate the mean magnitude of significant motion
        motion_score = significant_motion.mean() if significant_motion.size > 0 else 0.0
        optical_flow.append(motion_score)
        previous_frame = frame_image

    optical_flow.insert(0, optical_flow[0]) # Copy to the first frame
    # assert len(optical_flow) == len(video_data)
    return optical_flow


def optical_flow_driver_state_changed(optical_flow: list[float]) -> list[bool]:
    return bbox_size_drive_state_changed(optical_flow)



def bbox_size_drive_state_changed(bbox_sizes: list) -> list:
    n_samples, n_bkps = 2000, 3  # Number of data points and breakpoints
    bbox_sizes = np.array(bbox_sizes)
    bbox_sizes = bbox_sizes / np.sum(bbox_sizes)

    # algo = rpt.Binseg(model="l2", min_size=2, jump=5).fit(bbox_sizes)
    algo = rpt.KernelCPD(kernel="rbf").fit(bbox_sizes)
    breakpoints = algo.predict(n_bkps=n_bkps)

    driver_state_changed = [i >= breakpoints[0] for i in range(bbox_sizes.shape[0])]
    assert len(driver_state_changed) == len(bbox_sizes)
    return driver_state_changed


def baseline_driver_state_changed(bbox_centers_all: list) -> np.ndarray:
    previous_bbox_centers = []
    median_dists = []

    driver_state_changed = [False for _ in range(len(bbox_centers_all))]
    driver_state_changed = np.array(driver_state_changed)

    for i, bbox_centers in enumerate(bbox_centers_all):
        if len(bbox_centers) == 0 or len(previous_bbox_centers) == 0:
            if len(bbox_centers) !=0:
                previous_bbox_centers.append(bbox_centers)
            continue #We can't make a prediction of state change w/o knowing the previous state

        dists = []
        for bbox_center in bbox_centers:
            potential_dists = np.linalg.norm(previous_bbox_centers - bbox_center, axis=1)
            min_dist = np.sort(potential_dists)[0]
            dists.append(min_dist)

        median_dist = np.median(dists) #Take the median to reduce noise
        median_dists.append(median_dist)
        if len(median_dists) == 1:
            continue

        x = np.array(range(len(median_dists))).reshape(-1, 1)
        y = np.array(median_dists)
        speed_model = LinearRegression().fit(x, y)

        if speed_model.coef_[0] < 0: #If we are slowing down, driver state has probably changed
            driver_state_changed[i:] = True
            break
    return driver_state_changed


