import cv2
import os

# Settings
#res_width = 96                          # Resolution of camera (width)
#res_height = 96                         # Resolution of camera (height)
rotation = 0                            # Camera rotation (0, 90, 180, or 270)
draw_fps = False                        # Draw FPS on screen
save_path = "./"                        # Save images to current directory
file_num = 0                            # Starting point for filename
file_suffix = ".png"                    # Extension for image file
precountdown = 2                        # Seconds before starting countdown
countdown = 5                           # Seconds to count down from

# Initialize the camera
camera = cv2.VideoCapture(0)

# Set camera properties
res_width =int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
res_height =int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get the next available file path
def get_filepath():
    global file_num
    filepath = save_path + str(file_num) + file_suffix
    while file_exists(filepath):
        file_num += 1
        filepath = save_path + str(file_num) + file_suffix
    return filepath

# Check if file exists
def file_exists(filepath):
    return os.path.exists(filepath)

# Figure out the name of the output image filename
filepath = get_filepath()

# Capture frames until countdown reaches 0
while countdown > 0:
    # Read frame from camera
    ret, frame = camera.read()

    # Draw countdown on frame
    cv2.putText(frame,
                str(countdown),
                (int(round(res_width / 2) - 5),
                 int(round(res_height / 2))),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255))

    # Draw framerate on frame
    if draw_fps:
        cv2.putText(frame,
                    "FPS: " + str(round(fps, 2)),
                    (0, 12),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255))

    # Show the frame
    cv2.imshow("Frame", frame)

    # Decrease countdown
    countdown -= 1

    # Wait for 1 second
    cv2.waitKey(1000)

# Save the image
cv2.imwrite(filepath, frame)
print("Image saved to:", filepath)

# Release the camera
camera.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
