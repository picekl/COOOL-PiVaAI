import os

import cv2
import numpy as np
from PIL import Image
from transformers import GenerationConfig


def get_text(image, prompt, processor, model):
    """
    Generates a text response based on the given image and prompt using the specified processor and model.

    Args:
        image (np.ndarray): The input image in the form of a NumPy array.
        prompt (str): The text prompt to guide the image-to-text generation.
        processor: The processor used to process the image and prompt.
        model: The model used to generate the text response.

    Returns:
        str: The generated text response.
    """
    img = Image.fromarray(image)
    inputs = processor.process(images=[img], text=prompt)

    # Move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=400, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer,
    )

    # Only get generated tokens; decode them to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )
    return generated_text


def crop_square_with_context(image, bounding_box, context_percent=0.0, min_size=10):
    """
    Crops a square region from the image defined by a bounding box, ensuring minimum size
    and adding additional context as a percentage of box size.

    Args:
        image (np.ndarray): The input image in the form of a NumPy array.
        bounding_box (tuple): The bounding box in the form (x1, y1, x2, y2).
        context_percent (float): The percentage of additional context to add around the bounding box.
        min_size (int): The minimum size for the bounding box.

    Returns:
        np.ndarray: The cropped square image with context.
    """
    bounding_box = np.array(bounding_box).round().astype(int)
    x1, y1, x2, y2 = bounding_box

    # Ensure minimum box size
    box_width = x2 - x1
    box_height = y2 - y1

    if box_width < min_size:
        padding_x = (min_size - box_width) // 2
        x1 -= padding_x
        x2 += padding_x

    if box_height < min_size:
        padding_y = (min_size - box_height) // 2
        y1 -= padding_y
        y2 += padding_y

    # Recalculate box dimensions
    box_width = x2 - x1
    box_height = y2 - y1

    # Ensure the box is square by making both sides equal to the larger dimension
    side_length = max(box_width, box_height)

    # Center the square box around the original bounding box
    x_center = (x1 + x2) // 2
    y_center = (y1 + y2) // 2

    x1 = x_center - side_length // 2
    x2 = x_center + side_length // 2
    y1 = y_center - side_length // 2
    y2 = y_center + side_length // 2

    # Add context as a percentage of the side length
    context = int(side_length * context_percent)
    x1 -= context
    y1 -= context
    x2 += context
    y2 += context

    # Ensure the box stays within image bounds
    height, width = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # Crop the square region
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def get_img(filename, frame_id, folder_path="../data"):
    """
    Retrieves a specific frame from a video file.

    Args:
        filename (str): The name of the video file.
        frame_id (int): The frame number to retrieve.
        folder_path (str): The path to the folder containing the video file.

    Returns:
        np.ndarray: The image corresponding to the specified frame.
    """
    cap = cv2.VideoCapture(os.path.join(folder_path, filename))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the target frame
        if frame_count == frame_id:
            video = filename.split(".")[0]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            break
        frame_count += 1
    cap.release()
    return image


def get_area(bbox):
    """
    Calculates the area of a bounding box.

    Args:
        bbox (tuple): The bounding box in the form (x1, y1, x2, y2).

    Returns:
        float: The area of the bounding box.

    Raises:
        ValueError: If the bounding box is invalid (i.e., x2 <= x1 or y2 <= y1).
    """
    x1, y1, x2, y2 = bbox

    # Ensure valid bounding box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bounding box dimensions.")

    width = x2 - x1
    height = y2 - y1

    return width * height
