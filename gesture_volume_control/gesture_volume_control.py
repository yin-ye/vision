import hand_tracking_module as htm
import numpy as np
import math
import cv2
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# using the landmark values, calculate the difference in length
# between the index finger and thumb
def getTipsDifference(landmark_list, circle_color=(0, 0, 255), radius=15, draw=True):
    length = 0
    if len(landmark_list) > 0:
        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), radius-5, circle_color, cv2.FILLED)
            cv2.circle(img, (x1, y1), radius, circle_color, 2)
            cv2.circle(img, (x2, y2), radius-5, circle_color, cv2.FILLED)
            cv2.circle(img, (x2, y2), radius, circle_color, 2)
        # cv2.line(img, (x1, y1), (x2, y2), circle_color, line_thickness)
        length = math.hypot(x2 - x1, y2 - y1)
    return length

# adjust the system's volume using the distance between the index and thumb
# the closer the two fingers, the lesser the volume
def adjustVolume(length, minVol, maxVol):
    system_volume = np.interp(length, [50, 300], [minVol, maxVol])
    volume_bar = np.interp(length, [50, 300], [400, 150])
    volume_percent = np.interp(length, [50, 300], [0, 100])
    volume.SetMasterVolumeLevel(system_volume, None)
    return system_volume, volume_bar, volume_percent

##########################################################################
## MAIN FUNCTION
##########################################################################
if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(3, 640) # set webcam width
    capture.set(4, 480)  # set webcam height
    # define system volume setting parameters
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume_range = volume.GetVolumeRange()
    minVol = volume_range[0]
    maxVol = volume_range[1]

    previous_time = 0

    while True:
        success, img = capture.read()

        # use hand_tracking_module to detect shown hands
        hand_tracking = htm.HandTracking()
        if img is None:
            print("No frame is being displayed, ensure webcam is connected.")
            break
        img = hand_tracking.detectHands(img)

        # store the list of all hand landmarks to later extract the tips of the index
        # finger and thumb (landmarks 4 and 8)
        landmark_list = hand_tracking.findHandsPosition(img, draw=False)
        length = getTipsDifference(landmark_list)
        system_volume, volume_bar, volume_percent = adjustVolume(length, minVol, maxVol)

        # # show volume bar on the image frame
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volume_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volume_percent)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # # display FPS
        current_time = time.time()
        FPS = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, f'FPS: {int(FPS)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # show resultant image
        cv2.imshow("Frame", img)
        cv2.waitKey(1)
