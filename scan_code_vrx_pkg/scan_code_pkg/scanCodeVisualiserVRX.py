import cv2
import numpy as np
import time
import torch
import rclpy
import logging
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from ultralytics import YOLO

# Configure logging to silence YOLO's output
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Check if CUDA is available
print(torch.cuda.is_available())

# Load the YOLOv8 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise fallback to CPU
model = YOLO('/home/numarine/Documents/GitHub/Marine-ROS-Vision/scan_the_code/scan_code_vrx_pkg/scan_code_pkg/lightTowerVRX.pt')
model.to(device)  # Ensure the model uses GPU or CPU as available

# Get the class names
class_names = model.names  # This returns a dictionary where the key is the index and value is the class name

# Find the index of the 'matrix' class
matrix_class_index = [idx for idx, name in class_names.items() if name == 'matrix']

print(f"Index for 'matrix' class: {matrix_class_index}")

def get_color_mask(hsv_roi, color):
    """Returns a mask for the given color in the HSV color space."""
    if color == 'red':
        lower_bound = np.array([0, 100, 100])
        upper_bound = np.array([10, 255, 255])
        lower_bound2 = np.array([170, 100, 100])
        upper_bound2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_roi, lower_bound, upper_bound)
        mask2 = cv2.inRange(hsv_roi, lower_bound2, upper_bound2)
        mask = mask1 | mask2
    elif color == 'green':
        lower_bound = np.array([45, 100, 100])  
        upper_bound = np.array([75, 255, 255])  
        mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
    elif color == 'blue':
        lower_bound = np.array([94, 80, 50])
        upper_bound = np.array([126, 255, 255])
        mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
    else:
        mask = None
    return mask

def get_predominant_color(roi):
    """Detects and returns the predominant color in the given ROI."""
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    red_mask = get_color_mask(hsv_roi, 'red')
    green_mask = get_color_mask(hsv_roi, 'green')
    blue_mask = get_color_mask(hsv_roi, 'blue')
    
    red_count = cv2.countNonZero(red_mask)
    green_count = cv2.countNonZero(green_mask)
    blue_count = cv2.countNonZero(blue_mask)
    
    pixel_threshold = 500  # Adjust this value as needed
    
    if red_count > green_count and red_count > blue_count and red_count >= pixel_threshold:
        return 'Red', red_count
    elif green_count > red_count and green_count > blue_count and green_count >= pixel_threshold:
        return 'Green', green_count
    elif blue_count > red_count and blue_count > green_count and blue_count >= pixel_threshold:
        return 'Blue', blue_count
    else:
        return 'None', 0

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/wamv/sensors/cameras/front_left_camera_sensor/image_raw',
            self.image_callback,
            10)
        self.subscription

        self.br = CvBridge()
        self.color_publisher = self.create_publisher(String, 'code_sequence', 10)
        
        # Sequence tracking variables
        self.detected_colors = []
        self.last_detected_color = None
        self.sequence_history = []
        self.required_matches = 3
        
        # Timing variables
        self.last_color_time = time.time()
        self.display_timeout = 2.0  # Reset current sequence after 3 seconds
        self.history_timeout = 10.0  # Reset history after 10 seconds
        
        # Color mapping for output
        self.color_map = {
            'Red': 'R',
            'Green': 'G',
            'Blue': 'B'
        }

    def get_sequence_string(self, colors):
        """Convert list of colors to sequence string"""
        return ''.join([self.color_map.get(color, '') for color in colors])

    def check_and_publish_sequence(self):
        """Check if we have seen the same sequence three times consecutively"""
        if len(self.sequence_history) >= self.required_matches:
            # Get the last three sequences
            last_sequences = self.sequence_history[-self.required_matches:]
            
            # Check if they're all the same
            if all(seq == last_sequences[0] for seq in last_sequences):
                sequence_str = self.get_sequence_string(last_sequences[0])
                
                # Publish the verified sequence
                if len(sequence_str) == 3:
                    # Get current date and time
                    from datetime import datetime
                    now = datetime.now()
                    date_str = now.strftime("%d%m%y")  # ddmmyy
                    time_str = now.strftime("%H%M%S")  # hhmmss
                    
                    # Format the message according to specification
                    message = (
                        "$RXCOD,"      # Protocol Header
                        f"{date_str},"  # EST Date
                        f"{time_str},"  # EST Time
                        "TEAM_ID,"       # Team ID
                        f"{sequence_str}"  # Light Pattern
                        # "\r\n"         # End of message
                    )
                    
                    # Create and publish message
                    color_msg = String()
                    color_msg.data = message
                    self.color_publisher.publish(color_msg)
                    
                    # Print debug information
                    print(f"\n!!! VERIFIED SEQUENCE: {sequence_str} !!!")
                    print(f"Sequence verified after seeing it {self.required_matches} times consecutively")
                    print(f"Publishing message: {message}")
                    
                    # Clear sequence history after successful publication
                    print("Clearing sequence history after successful publication")
                    self.sequence_history = []
                    return True  # Indicate successful publication
            return False

    def handle_color_detection(self, detected_color):
        """Handle color detection and sequence building"""
        current_time = time.time()
        time_since_last_color = current_time - self.last_color_time
        
        if detected_color == 'None':
            # Reset current sequence after display timeout (3 seconds)
            if self.detected_colors and time_since_last_color >= self.display_timeout:
                print(f"No color for {time_since_last_color:.1f}s - resetting current sequence")
                self.detected_colors = []
                self.last_detected_color = None
            
            # Reset history after longer timeout (10 seconds)
            if time_since_last_color >= self.history_timeout and self.sequence_history:
                print(f"\nNo color detected for {time_since_last_color:.1f}s - resetting sequence history")
                self.sequence_history = []
                print("Sequence history cleared")
        else:
            # Update timing
            self.last_color_time = current_time
            
            # If color has changed from last detection
            if detected_color != self.last_detected_color:
                print(f"\nDetected new color: {detected_color}")
                
                # If we have a complete sequence from before, clear it now
                if len(self.detected_colors) == 3:
                    self.detected_colors = []
                
                # Add to current sequence
                self.detected_colors.append(detected_color)
                print(f"Current sequence build: {[c for c in self.detected_colors]}")
                
                # If we have a complete sequence of 3 colors
                if len(self.detected_colors) == 3:
                    sequence_str = self.get_sequence_string(self.detected_colors)
                    print(f"\nComplete sequence detected: {sequence_str}")
                    print(f"Raw sequence: {self.detected_colors}")
                    
                    # Check if this sequence is different from the last one in history
                    if self.sequence_history and self.detected_colors != self.sequence_history[-1]:
                        print("New sequence differs from previous - clearing history")
                        self.sequence_history = []
                    
                    # Add to history
                    self.sequence_history.append(self.detected_colors.copy())
                    
                    # Print current history
                    print("\nSequence History:")
                    for idx, seq in enumerate(self.sequence_history):
                        print(f"  {idx + 1}: {self.get_sequence_string(seq)} ({seq})")
                    
                    # Check for three consecutive matches
                    self.check_and_publish_sequence()
                
                self.last_detected_color = detected_color

    def create_visualization(self, width, height):
        visualization = np.ones((height, width, 3), dtype=np.uint8) * 255

        title_text = "Scan the Code"
        title_font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 1.5
        title_thickness = 3
        text_size = cv2.getTextSize(title_text, title_font, title_scale, title_thickness)[0]

        text_x = (width - text_size[0]) // 2
        text_y = int(height * 0.1)

        cv2.putText(visualization, title_text, (text_x, text_y), title_font, title_scale, (0, 0, 0), title_thickness)

        color_map = {
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            'None': (255, 255, 255)
        }

        rect_width = int(width * 0.2)
        rect_height = int(height * 0.3)
        rect_y_start = int(height * 0.5)
        spacing = int(width * 0.1)

        rect_positions = [
            ((width - 3*rect_width - 2*spacing) // 2, rect_y_start),
            ((width - rect_width) // 2, rect_y_start),
            ((width + rect_width + 2*spacing) // 2, rect_y_start)
        ]

        def draw_boxed_text(img, text, pos, font, font_scale, text_color, border_color, thickness, box_width):
            x, y = pos
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_w, text_h = text_size
            box_height = text_h * 4
            
            cv2.rectangle(img, (x, y-box_height), (x+box_width, y), border_color, 2)
            cv2.rectangle(img, (x+1, y-box_height+1), (x+box_width-1, y-1), (255,255,255), -1)
            
            text_x = x + (box_width - text_w) // 2
            text_y = y - (box_height - text_h) // 2
            cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)

        # Get current time for timeout display
        current_time = time.time()
        time_since_last_color = current_time - self.last_color_time

        # Draw the visualization boxes
        for idx, (x_start, y_start) in enumerate(rect_positions):
            # Add position number
            position_number = str(idx + 1)
            cv2.putText(visualization, f"Position {position_number}", 
                    (x_start, y_start - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 0, 0), 2)

            # Draw rectangle border
            cv2.rectangle(visualization, (x_start, y_start), 
                        (x_start + rect_width, y_start + rect_height), 
                        (0, 0, 0), 2)
            
            # Only show colors if we're actively detecting colors
            if len(self.detected_colors) > 0 and idx < len(self.detected_colors):
                color = self.detected_colors[idx]
                text = f"{color}"
                text_color = (0, 0, 0)
                # Fill with detected color
                cv2.rectangle(visualization, (x_start+2, y_start+2), 
                            (x_start + rect_width-2, y_start + rect_height-2), 
                            color_map.get(color, (255, 255, 255)), -1)
            else:
                # Show empty state
                text = "None"
                text_color = (200, 200, 200)
                # Fill with white
                cv2.rectangle(visualization, (x_start+2, y_start+2), 
                            (x_start + rect_width-2, y_start + rect_height-2), 
                            color_map['None'], -1)
            
            # Draw text box
            text_y = y_start - 10
            draw_boxed_text(visualization, text, (x_start, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 
                        (0, 0, 0), 2, rect_width)

        # Status text positions from bottom
        bottom_margin = 10
        line_spacing = 30
        status_y_base = visualization.shape[0] - bottom_margin

        # Add history reset countdown if in timeout period
        if time_since_last_color >= self.display_timeout:
            time_to_reset = self.history_timeout - time_since_last_color
            if time_to_reset > 0:
                reset_text = f"History reset in: {time_to_reset:.1f}s"
                cv2.putText(visualization, reset_text,
                        (10, status_y_base - 3*line_spacing),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Add timing information
        timer_text = f"Time since last color: {time_since_last_color:.1f}s"
        cv2.putText(visualization, timer_text, 
                (10, status_y_base - line_spacing),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Show sequence count if we have sequences
        if self.sequence_history:
            last_sequence = self.sequence_history[-1]
            sequence_str = self.get_sequence_string(last_sequence)
            
            # Count consecutive matches of the last sequence
            consecutive_matches = 1
            for seq in reversed(self.sequence_history[:-1]):
                if seq == last_sequence:
                    consecutive_matches += 1
                else:
                    break
            
            if consecutive_matches >= self.required_matches:
                status = "VERIFIED!"
            else:
                status = f"Matches: {consecutive_matches}/{self.required_matches}"
                
            sequence_text = f"Current sequence: {sequence_str} | {status}"
            cv2.putText(visualization, sequence_text,
                    (10, status_y_base - 2*line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return visualization
    
    def image_callback(self, msg):
        current_frame = self.br.imgmsg_to_cv2(msg)
        current_frame_bgr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

        results = model.predict(source=current_frame_bgr, device='cuda', save=False, classes=matrix_class_index, conf=0.2)
        time.sleep(0.5)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        detected_color = 'None'
        pixel_count = 0
        
        for box, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box[:4])
            box_center_y = int((y1 + y2) / 2)
            roi = current_frame_bgr[y1:y2, x1:x2]
            detected_color, pixel_count = get_predominant_color(roi)
            
            # Draw bounding box and display information
            cv2.rectangle(current_frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text_x = x2 + 10
            text_y = box_center_y

            confidence_label = f"Confidence: {confidence:.2f}"
            cv2.putText(current_frame_bgr, confidence_label, (text_x, text_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if detected_color == 'None':
                color_label = "No color detected"
            else:
                color_label = f"{detected_color}: {pixel_count} pixels"
            cv2.putText(current_frame_bgr, color_label, (text_x, text_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Process the detected color
        self.handle_color_detection(detected_color)

        # Create visualization
        visualization = self.create_visualization(current_frame_bgr.shape[1], current_frame_bgr.shape[0])
        
        # Create the combined window
        combined_window = cv2.hconcat([current_frame_bgr, visualization])
        cv2.namedWindow('Combined Window', cv2.WINDOW_NORMAL)
        cv2.imshow("Combined Window", combined_window)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()