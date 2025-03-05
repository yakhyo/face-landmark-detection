import argparse
import numpy as np
import cv2
import torch
import torchvision
import logging

from models.pfld import PFLDInference
from uniface import RetinaFace
from uniface.visualization import draw_detections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    """Load the trained PFLD model from checkpoint."""
    logging.info(f"Loading PFLD model from {model_path}")
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    pfld_backbone.eval()
    return pfld_backbone


def preprocess_face(img, box, width, height):
    """Crop, pad, and resize the face for landmark detection."""
    x1, y1, x2, y2, _ = box.astype(np.int32)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    size = int(max([w, h]) * 1.1)
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = x1 + size, y1 + size
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return None, None, None, None

    input_img = cv2.resize(cropped, (112, 112))
    return input_img, size, (x1, y1)


def draw_landmarks(img, landmarks, face_origin, size):
    """Draw detected facial landmarks on the image."""
    x1, y1 = face_origin
    for (x, y) in landmarks.astype(np.int32):
        cv2.circle(img, (x1 + x, y1 + y), 1, (0, 255, 0), -1)  # Green dots


def main(args):
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    # Load RetinaFace model for face detection
    uniface_inference = RetinaFace(
        model="retinaface_mnet_v2",
        conf_thresh=0.5,
        pre_nms_topk=5000,
        nms_thresh=0.4,
        post_nms_topk=750,
        dynamic_size=False,
        input_size=(640, 640)
    )
    logging.info("RetinaFace model loaded for face detection.")

    # Load PFLD model for landmark prediction
    pfld_backbone = load_model(args.model_path)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Unable to access the webcam.")
        exit()

    logging.info("Webcam feed started. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Failed to read frame.")
            break

        height, width = frame.shape[:2]

        # Face detection using RetinaFace
        boxes, lmarks = uniface_inference.detect(frame)

        for box in boxes:
            x1, y1, x2, y2, _= box.astype(np.int32)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box

            face_img, size, face_origin = preprocess_face(frame, box, width, height)
            if face_img is None:
                continue

            input_tensor = transform(face_img).unsqueeze(0).to(device)
            _, landmarks = pfld_backbone(input_tensor)
            pre_landmark = landmarks[0].cpu().detach().numpy().reshape(-1, 2) * [size, size]

            # Draw facial landmarks
            draw_landmarks(frame, pre_landmark, face_origin, size)

        # Draw detections from RetinaFace for visualization
        draw_detections(frame, (boxes, lmarks), vis_threshold=0.6)

        # Display the output
        cv2.imshow("Face Detection & Landmark Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam feed stopped. Resources released.")


def parse_args():
    parser = argparse.ArgumentParser(description='Face Detection and Landmark Prediction')
    parser.add_argument('--model_path', default="last_ckpt.pth", type=str, help="Path to the trained PFLD model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
