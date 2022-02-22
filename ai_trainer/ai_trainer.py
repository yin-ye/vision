import cv2
import numpy as np
import time
import pose_tracking_module as pt

# scales the image according to a given percentage
def resizeImage(img, scale=20):
    width = int(img.shape[1] * scale/100)
    height = int(img.shape[0] * scale / 100)
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized_img


if __name__ == "__main__":
    capture = cv2.VideoCapture('videos/squats.mp4')

    pose_tracking = pt.poseTracking()

    exercise_count = 0
    direction = 0
    previous_time = 0

    # only perform action while video is available
    while capture.isOpened():
        success, img = capture.read()
        # resize image due to large size
        img = resizeImage(img, 20)

        # detect pose landmarks and return indexed list
        img = pose_tracking.findPose(img)
        landmark_list = pose_tracking.getLandmarkCoordinates(img)

        if len(landmark_list) > 0:
            # Check if a squat is performed by calculating the angle of the knees
            angle = pose_tracking.calculateAngle(img, 24, 26, 28)
            percent = np.interp(angle, (180, 62), (100, 0))
            bar = np.interp(angle, (220, 310), (650, 100))

            # Check that the body completes a full squat and increment the count
            if percent == 100:
                color = (0, 255, 0)
                if direction == 0:
                    exercise_count += 0.5
                    direction = 1
            if percent == 0:
                if direction == 1:
                    exercise_count += 0.5
                    direction = 0

            # Display squat count
            cv2.putText(img, f'Squats: {int(exercise_count)}', (30, 575), cv2.FONT_HERSHEY_PLAIN, 2,
                        (18, 255, 255), 3)

        # # display FPS
        current_time = time.time()
        FPS = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, f'FPS: {int(FPS)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (18, 255, 255), 3)

        # Show resultant image
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    capture.release()