import numpy as np
import cv2

cifar100_classes = np.array([
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
])


def crop_with_context(image, bounding_box, context_percent, min_size=20):
    """
    Crops a region from the image defined by a bounding box, ensuring minimum size
    and adding additional context as a percentage of box size.
    
    Parameters:
        image (numpy array): The original image.
        bounding_box (tuple): The bounding box (x1, y1, x2, y2).
        min_size (int): Minimum size for the width and height of the box.
        context_percent (float): Percentage of box size to add as context (e.g., 0.1 for 10%).

    Returns:
        numpy array: Cropped image region.
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



def get_img(filename, frame_id, anns=None, folder_path = "../data"):
    cap = cv2.VideoCapture(os.path.join(folder_path, filename))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        # Process every Nth frame to match the target FPS
        if frame_count  == frame_id:
            video = filename.split('.')[0]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if anns is not None:
                objects = anns[video][frame_count]['challenge_object']
                for obj in objects:
                    x1, y1, x2, y2 = np.array(obj['bbox']).round().astype(int)
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            break
        frame_count += 1    
    cap.release()
    return image
