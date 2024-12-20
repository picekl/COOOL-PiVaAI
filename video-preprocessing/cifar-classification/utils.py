import cv2
import numpy as np

cifar100_classes = np.array(
    [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ]
)


def crop_with_context(image, bounding_box, context_percent, min_size=20):
    """
    Crops a region from the image defined by a bounding box, ensuring minimum size
    and adding additional context as a percentage of the box size.

    This function calculates the bounding box, ensures the minimum size for cropping,
    adds context around the bounding box based on the provided percentage, and returns
    the cropped image region. The function also ensures that the crop does not extend
    beyond the image bounds.

    Parameters:
        image (numpy array): The original image from which to crop the region.
        bounding_box (tuple): The bounding box coordinates (x1, y1, x2, y2).
        min_size (int, optional): Minimum size for the width and height of the box.
                                  Default is 20 pixels.
        context_percent (float): Percentage of the bounding box size to add as context
                                  around the crop. For example, 0.1 for 10%.

    Returns:
        numpy array: The cropped image region with context added, adjusted to ensure
                      the crop doesn't go out of bounds.
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

    # Compute context in pixels
    context_x = int(box_width * context_percent)
    context_y = int(box_height * context_percent)

    # Add context, ensuring we stay within image bounds
    height, width = image.shape[:2]
    x1 = max(0, x1 - context_x)
    y1 = max(0, y1 - context_y)
    x2 = min(width, x2 + context_x)
    y2 = min(height, y2 + context_y)

    # Crop the region
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def get_img(filename, frame_id, anns=None, folder_path="../data"):
    """
    Loads an image frame from a video file and optionally draws bounding boxes for
    annotated objects.

    This function opens a video file, extracts the frame specified by `frame_id`,
    and optionally overlays bounding boxes for objects defined in the annotations.
    The image is then returned in RGB format.

    Parameters:
        filename (str): The name of the video file to process.
        frame_id (int): The index of the frame to extract from the video.
        anns (dict, optional): A dictionary containing annotations for objects in the video.
                                Each entry should map video names to frame-level annotations.
        folder_path (str, optional): Path to the folder containing the video file. Default is "../data".

    Returns:
        numpy array: The processed image frame in RGB format with optional bounding boxes drawn.
    """
    cap = cv2.VideoCapture(os.path.join(folder_path, filename))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame if it matches the target frame_id
        if frame_count == frame_id:
            video = filename.split(".")[0]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # If annotations are provided, draw bounding boxes for objects
            if anns is not None:
                objects = anns[video][frame_count]["challenge_object"]
                for obj in objects:
                    x1, y1, x2, y2 = np.array(obj["bbox"]).round().astype(int)
                    image = cv2.rectangle(
                        image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2
                    )
            break
        frame_count += 1

    cap.release()
    return image
