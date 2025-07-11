import argparse
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils.dataset import WLFWDatasets
from models.pfld import PFLDInference

# Enable cuDNN optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_nme(preds, target):
    """
    Compute Normalized Mean Error (NME) between predicted and ground truth landmarks.

    Args:
        preds (numpy.ndarray): Predicted landmarks, shape (N, L, 2)
        target (numpy.ndarray): Ground truth landmarks, shape (N, L, 2)

    Returns:
        np.ndarray: NME values for each sample
    """
    L = preds.shape[1]

    if L == 68:
        interocular = np.linalg.norm(target[:, 36] - target[:, 45], axis=1)
    elif L == 98:
        interocular = np.linalg.norm(target[:, 60] - target[:, 72], axis=1)
    elif L == 19:
        interocular = 34  # Fixed value
    elif L == 29:
        interocular = np.linalg.norm(target[:, 8] - target[:, 9], axis=1)
    else:
        raise ValueError("Invalid number of landmarks.")

    rmse = np.sum(np.linalg.norm(preds - target, axis=2), axis=1) / (interocular * L)
    return rmse


def compute_auc(errors, failure_threshold=0.1, step=0.0001, show_curve=True):
    """
    Compute the Area Under Curve (AUC) and failure rate for NME.

    Args:
        errors (list): List of NME errors.
        failure_threshold (float): Threshold for failure rate calculation.
        step (float): Step size for curve computation.
        show_curve (bool): Whether to display the curve.

    Returns:
        float: AUC value
        float: Failure rate
    """
    x_axis = np.arange(0., failure_threshold + step, step)

    # Compute Cumulative Error Distribution (CED)
    ced = np.array([np.count_nonzero(errors <= x) / len(errors) for x in x_axis])

    auc = simpson(ced, x=x_axis) / failure_threshold
    failure_rate = 1.0 - ced[-1]

    if show_curve:
        plt.plot(x_axis, ced)
        plt.xlabel("Error Threshold")
        plt.ylabel("Proportion of Samples")
        plt.title("Cumulative Error Distribution")
        plt.show()

    return auc, failure_rate


def validate(dataloader, model, show_image=False):
    """
    Validate the model on the dataset.

    Args:
        dataloader (DataLoader): DataLoader for validation dataset.
        model (torch.nn.Module): Model for inference.
        show_image (bool): Whether to visualize predictions.

    Returns:
        None
    """
    model.eval()
    nme_list = []
    cost_times = []

    with torch.no_grad():
        for image, landmark_gt, _, _ in dataloader:
            image, landmark_gt = image.to(device), landmark_gt.to(device)

            start_time = time.time()
            _, landmarks = model(image)
            cost_times.append(time.time() - start_time)

            landmarks = landmarks.cpu().numpy().reshape(landmarks.shape[0], -1, 2)
            landmark_gt = landmark_gt.cpu().numpy().reshape(landmark_gt.shape[0], -1, 2)

            if show_image:
                visualize_landmarks(image[0].cpu().numpy(), landmarks[0])

            nme_list.extend(compute_nme(landmarks, landmark_gt))

    print(f'NME: {np.mean(nme_list):.4f}')
    auc, failure_rate = compute_auc(nme_list)
    print(f'AUC @ 0.1 Failure Threshold: {auc:.4f}')
    print(f'Failure Rate: {failure_rate:.4f}')
    print(f'Inference Time: {np.mean(cost_times):.6f} sec')


def visualize_landmarks(image, landmarks):
    """
    Visualize landmarks on the image in a resizable window.

    Args:
        image (numpy.ndarray): Image in CHW format (3, H, W) with values in range [0,1].
        landmarks (numpy.ndarray): Predicted landmarks.

    Returns:
        None
    """
    # Convert CHW (Channel, Height, Width) to HWC (Height, Width, Channel)
    image = (np.transpose(image, (1, 2, 0)) * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Ensure image is contiguous for OpenCV
    image = np.ascontiguousarray(image)

    # Scale landmarks to image size
    landmarks *= np.array([image.shape[1], image.shape[0]])

    for (x, y) in landmarks.astype(np.int32):
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # Create a resizable window
    cv2.namedWindow("Landmarks", cv2.WINDOW_NORMAL)

    # Resize the window to a reasonable default size (optional)
    cv2.resizeWindow("Landmarks", max(600, image.shape[1]), max(400, image.shape[0]))

    while True:
        cv2.imshow("Landmarks", image)
        key = cv2.waitKey(1) & 0xFF  # Wait for key press

        # Press 'q' or 'Esc' to close
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


def main(args):
    checkpoint = torch.load(args.model_path, weights_only=False, map_location=device)
    model = PFLDInference().to(device)
    model.load_state_dict(checkpoint['pfld_backbone'])

    transform = transforms.ToTensor()
    dataset = WLFWDatasets(args.test_dataset, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    validate(dataloader, model, args.show_image)


def parse_args():
    parser = argparse.ArgumentParser(description="Facial Landmark Detection Model Testing Script")

    parser.add_argument(
        "--model-path",
        type=str,
        default="./checkpoint/last_ckpt.pth",
        help="Path to the pre-trained model checkpoint file (.pth)."
    )

    parser.add_argument(
        "--test-dataset",
        type=str,
        default="./data/test_data/list.txt",
        help="Path to the test dataset file (list of image paths and landmarks)."
    )

    parser.add_argument(
        "--show-image",
        action="store_true",
        help="If set, displays the images with predicted landmarks overlayed."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
