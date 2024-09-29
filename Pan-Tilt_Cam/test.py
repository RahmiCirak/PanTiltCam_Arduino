from ultralytics import YOLO
import cv2
import pyfirmata

# Load the YOLO model
model = YOLO('yolov8n-face.pt')

port = "COM6"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s') #pin 9 Arduino
servo_pinY = board.get_pin('d:10:s') #pin 10 Arduino

servo_pinX.write(0)
servo_pinY.write(0)  # initial servo position

# Open video capture from the camera
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

# Define the range of motion for the servos
servo_x_min_angle = 120
servo_x_max_angle = 0
servo_y_min_angle = 0
servo_y_max_angle = 120

while True:
    # Read frame from the video capture
    ret, frame = cap.read()

    # Perform inference using the YOLO model
    results = model.predict(frame)

    # Initialize variables to store the total center coordinates
    total_center_x = 0
    total_center_y = 0
    total_boxes = 0

    # Process detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract box coordinates
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format

            # Calculate center coordinates
            center_x = (b[0] + b[2]) / 2
            center_y = (b[1] + b[3]) / 2

            # Accumulate center coordinates
            total_center_x += center_x
            total_center_y += center_y
            total_boxes += 1

    # Calculate the average center coordinates
    if total_boxes > 0:
        average_center_x = total_center_x / total_boxes
        average_center_y = total_center_y / total_boxes

        # Calculate servo angles based on average center coordinates
        servo_x_angle = (servo_x_max_angle - servo_x_min_angle) * (average_center_x / frame.shape[1]) + servo_x_min_angle
        servo_y_angle = (servo_y_max_angle - servo_y_min_angle) * (average_center_y / frame.shape[0]) + servo_y_min_angle

        # Draw circle at the average center point
        cv2.circle(frame, (int(average_center_x), int(average_center_y)), 5, (0, 255, 0), -1)

        # Display the average center coordinates and servo angles
        center_text = f"Center (X: {average_center_x:.2f}, Y: {average_center_y:.2f})"
        servo_text = f"Servo (X: {servo_x_angle:.2f}, Servo Y: {servo_y_angle:.2f})"
        cv2.putText(frame, center_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, servo_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        servo_pinX.write(servo_x_angle)
        servo_pinY.write(servo_y_angle)

    # Display the frame
    cv2.imshow('YOLO V8 Detection', frame)

    # Break the loop if 'Space' key is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
