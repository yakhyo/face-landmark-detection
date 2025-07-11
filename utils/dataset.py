import numpy as np
import cv2
import random
from torch.utils import data
from torch.utils.data import DataLoader


def flip(image, annotation):
    """Flip image and annotations horizontally."""
    image = np.fliplr(image)
    w = image.shape[1]

    annotation[0], annotation[2] = w - annotation[2], w - annotation[0]
    annotation[4::2] = w - annotation[4::2]  # Flip x-coordinates of landmarks

    return image, annotation


def channel_shuffle(image, annotation):
    """Shuffle color channels randomly."""
    if image.shape[2] == 3:
        image = image[..., np.random.permutation(3)]
    return image, annotation


def random_noise(image, annotation, limit=(0, 0.2), p=0.5):
    """Apply random noise to the image."""
    if random.random() < p:
        noise = np.random.uniform(limit[0], limit[1], image.shape[:2]) * 255
        image = np.clip(image + noise[:, :, None], 0, 255).astype(np.uint8)
    return image, annotation


def random_brightness(image, annotation, brightness=0.3):
    """Adjust brightness randomly."""
    alpha = 1 + np.random.uniform(-brightness, brightness)
    image = np.clip(image * alpha, 0, 255).astype(np.uint8)
    return image, annotation


def random_contrast(image, annotation, contrast=0.3):
    """Adjust contrast randomly."""
    coef = np.array([0.114, 0.587, 0.299])  # RGB to grayscale conversion
    gray = np.dot(image, coef)
    mean_gray = np.mean(gray)
    image = np.clip((1.0 + np.random.uniform(-contrast, contrast)) * image +
                    (1.0 - contrast) * mean_gray, 0, 255).astype(np.uint8)
    return image, annotation


def random_saturation(image, annotation, saturation=0.5):
    """Adjust saturation randomly."""
    coef = np.array([0.299, 0.587, 0.114])
    gray = np.dot(image, coef[..., None])
    alpha = np.random.uniform(-saturation, saturation)
    image = np.clip(alpha * image + (1.0 - alpha) * gray, 0, 255).astype(np.uint8)
    return image, annotation


def random_hue(image, annotation, hue=0.5):
    """Adjust hue randomly while handling uint8 overflow."""
    h = int(np.random.uniform(-hue, hue) * 180)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)  # Convert to int16 for safe addition
    hsv[:, :, 0] = (hsv[:, :, 0] + h) % 180  # Keep values within valid range
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)  # Convert back to uint8

    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image, annotation


def scale(image, annotation, scale_range=(-0.4, 0.8)):
    """Scale image and annotations."""
    f_xy = np.random.uniform(*scale_range)
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w * f_xy), int(h * f_xy)))

    annotation[:4] *= f_xy  # Scale bounding box
    annotation[4:] *= f_xy  # Scale landmarks

    return image, annotation


def rotate(image, annotation, alpha=30):
    """Rotate image and annotations."""
    h, w = image.shape[:2]
    center = ((annotation[0] + annotation[2]) / 2, (annotation[1] + annotation[3]) / 2)

    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    image = cv2.warpAffine(image, rot_mat, (w, h))

    points = np.hstack([annotation[4:].reshape(-1, 2), np.ones((len(annotation[4:]) // 2, 1))])
    rotated_points = np.dot(rot_mat, points.T).T.flatten()

    annotation[4:] = rotated_points
    return image, annotation


def apply_random_transforms(image, annotation, p=0.5):
    """Apply a random subset of augmentations."""
    transforms = [
        channel_shuffle,
        random_noise,
        random_brightness,
        random_contrast,
        random_saturation,
        random_hue,
    ]

    random.shuffle(transforms)

    for transform in transforms:
        if random.random() < p:
            image, annotation = transform(image, annotation)

    return image, annotation

from torchvision import transforms

class WLFWDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None, apply_augmentations=True):
        """
        Args:
            file_list (str): Path to the text file containing dataset information.
            transform (callable, optional): A function/transform that converts the PIL image to tensor.
            apply_augmentations (bool): Whether to apply random color augmentations.
        """
        self.transforms = transforms
        self.apply_augmentations = apply_augmentations

        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        line = self.lines[index].strip().split()
        image = cv2.imread(line[0])

        if image is None:
            raise ValueError(f"Error loading image: {line[0]}")

        landmark = np.array(line[1:197], dtype=np.float32)
        attribute = np.array(line[197:203], dtype=np.int32)
        euler_angle = np.array(line[203:206], dtype=np.float32)

        # Apply random color augmentations **before converting to tensor**
        if self.apply_augmentations:
            image, _ = apply_random_transforms(image, landmark)

        # Convert BGR (OpenCV) to RGB (Torch uses RGB format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image to PIL format before applying Torchvision transforms
        image = transforms.ToPILImage()(image)

        # Apply Torchvision transforms (e.g., resizing, normalization)
        if self.transforms:
            image = self.transforms(image)

        return image, landmark, attribute, euler_angle

    def __len__(self):
        return len(self.lines)


if __name__ == '__main__':
    file_list = './data/test_data/list.txt'
    dataset = WLFWDatasets(file_list)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=False)

    for image, landmark, attribute, euler_angle in dataloader:
        print("Image shape:", image.shape)
        print("Landmark size:", landmark.shape)
        print("Attribute size:", attribute.shape)
        print("Euler angle size:", euler_angle.shape)
