import argparse
import numpy as np
import cv2
import onnxruntime as ort
import logging
import os

from uniface.detection import RetinaFace
from uniface.visualization import draw_detections


def load_onnx_model(model_path):
    """Load ONNX model for inference."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")

    logging.info(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(model_path)
    logging.info("ONNX model loaded successfully")
    return session


def preprocess_face(img, bbox, width, height):
    """Crop, pad, and resize the face for landmark detection."""
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
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


def process_frame(frame, detector, onnx_session):
    """Process a single frame for face detection and landmark prediction."""
    height, width = frame.shape[:2]

    # Face detection using RetinaFace
    faces = detector.detect(frame)

    # Extract data for visualization
    bboxes = [f['bbox'] for f in faces]
    scores = [f['confidence'] for f in faces]
    landmarks = [f['landmarks'] for f in faces]

    # Process each detected face
    for face in faces:
        bbox = face['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue bounding box

        face_img, size, face_origin = preprocess_face(frame, bbox, width, height)
        if face_img is None:
            continue

        # Prepare input for ONNX model
        input_img = face_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))  # HWC to CHW
        input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension

        # Run ONNX inference
        input_name = onnx_session.get_inputs()[0].name
        output_names = [output.name for output in onnx_session.get_outputs()]

        outputs = onnx_session.run(output_names, {input_name: input_img})
        landmarks_output = outputs[1]  # landmarks is the second output

        pre_landmark = landmarks_output[0].reshape(-1, 2) * [size, size]

        # Draw facial landmarks
        draw_landmarks(frame, pre_landmark, face_origin, size)

    # Draw detections from RetinaFace for visualization
    draw_detections(frame, bboxes, scores, landmarks, vis_threshold=0.6)

    return frame


def run_webcam_inference(args, detector, onnx_session):
    """Run inference on webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Unable to access the webcam.")
        return

    logging.info("Webcam feed started. Press 'q' to exit.")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Failed to read frame.")
            break

        frame = process_frame(frame, detector, onnx_session)

        # Display the output
        cv2.imshow("ONNX PFLD - Face Detection & Landmark Prediction - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam feed stopped. Resources released.")


def run_image_inference(args, detector, onnx_session):
    """Run inference on a single image."""
    if not os.path.exists(args.image):
        logging.error(f"Image file not found: {args.image}")
        return

    # Load image
    frame = cv2.imread(args.image)
    if frame is None:
        logging.error(f"Failed to load image: {args.image}")
        return

    logging.info(f"Processing image: {args.image}")

    # Process the image
    result_frame = process_frame(frame.copy(), detector, onnx_session)

    # Save or display result
    if args.output:
        cv2.imwrite(args.output, result_frame)
        logging.info(f"Result saved to: {args.output}")
    else:
        # Display the result
        cv2.imshow("ONNX PFLD - Face Detection & Landmark Prediction - Press any key to exit", result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main(args):
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )

    # Load RetinaFace model for face detection
    detector = RetinaFace()
    logging.info("RetinaFace model loaded for face detection.")

    # Load ONNX model for landmark prediction
    onnx_session = load_onnx_model(args.model_path)

    # Run inference based on input type
    if args.image:
        run_image_inference(args, detector, onnx_session)
    else:
        run_webcam_inference(args, detector, onnx_session)


def parse_args():
    parser = argparse.ArgumentParser(description='ONNX PFLD Face Detection and Landmark Prediction')
    parser.add_argument('--model_path', default="weights/pfld_landmark.onnx", type=str,
                       help="Path to the ONNX model.")
    parser.add_argument('--image', type=str, default=None,
                       help="Path to input image. If not provided, webcam will be used.")
    parser.add_argument('--output', type=str, default=None,
                       help="Path to save output image (only for image input).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
