import cv2

def capture_image(output_path="captured_image.png"):
    """
    Captures an image using the webcam and saves it to the specified output path.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use cv2.CAP_DSHOW on Windows for stability
    if not cap.isOpened():
        raise Exception("Could not open the webcam. Ensure it's connected and drivers are installed.")

    print("Press 'Space' to capture an image or 'Esc' to exit without capturing.")
    captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame. Retrying...")
            continue

        cv2.imshow("Capture Image", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key to exit
            print("Exiting without capturing.")
            break
        elif key == 32:  # Space key to capture
            cv2.imwrite(output_path, frame)
            print(f"Image captured and saved as '{output_path}'.")
            captured = True
            break

    cap.release()
    cv2.destroyAllWindows()
    return output_path if captured else None