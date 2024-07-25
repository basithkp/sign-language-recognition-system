
 
import cv2
import numpy as np
 
# Settings
file_num = 0

save_path = "./images/No/"      # Save images to current directory
file_suffix = ".jpg"  # Extension for image file
SECONDS_TO_COUNTDOWN = 0
 
 
def file_exists(filepath):
    """
    Returns true if file exists, false otherwise.
    """
    try:
        f = open(filepath, 'r')
        exists = True
        f.close()
    except:
        exists = False
    return exists
 
def get_filepath():
    """
    Returns the next available full path to image file
    """
    global file_num
    # Loop through possible file numbers to see if that file already exists
    filepath = save_path + str(file_num) + file_suffix
    while file_exists(filepath):
        file_num += 1
        filepath = save_path + str(file_num) + file_suffix
 
    return filepath
 
def main():
    countdown = SECONDS_TO_COUNTDOWN
 
    # Figure out the name of the output image filename
    filepath = get_filepath()
 
    cam = cv2.VideoCapture(0)
 
    # Set smaller resolution
    #cam.set(cv2.CAP_PROP_FRAME_WIDTH, 160) # 640
    #cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 120) # 480
 
    # Initial countdown timestamp
    countdown_timestamp = cv2.getTickCount()
 
    while cam.isOpened():
        # Read camera
        ret, frame = cam.read()
 
        # Get timestamp for calculating actual framerate
        timestamp = cv2.getTickCount()
 
        
 
        # Frame resolution
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
 
        # Crop center of image
        
 
        # Rezise to 96*96
        # Rezise to frame size
        resized = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

 
        # Put text only on copied image
        copy = resized.copy()
        # Each second, decrement countdown
        if (timestamp - countdown_timestamp) / cv2.getTickFrequency() > 1.0:
            countdown_timestamp = cv2.getTickCount()
            countdown -= 1
             
            # When countdown reaches 0, break out of loop to save image
            if countdown <= 0:
                # Get new image file name
                filepath = get_filepath()
                # Save image
                cv2.imwrite(filepath, resized)
                # Start new count down
                countdown = SECONDS_TO_COUNTDOWN
                #break
        # Draw countdown on image
        cv2.putText(copy, 
                    str(countdown),
                    (round(resized.shape[1] / 2) - 15, round(resized.shape[0] / 2)+10),
                    cv2.FONT_HERSHEY_PLAIN,
                    4,
                    (255, 255, 255))
 
        # Display raw camera image
        cv2.imshow('Kaamera', copy)
 
        # Press 'q' to exit
        if cv2.waitKey(10) == ord('q'):
            break
 
    # Clean up
    cam.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    print('To exit press "q"')
    main()