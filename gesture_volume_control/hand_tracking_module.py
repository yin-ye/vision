# import libraries
import cv2
import mediapipe as mp


class HandTracking():
    def __init__(self, mode=False, max_hands=2, complexity=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.complexity = complexity
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity, self.detection_confidence,
                                         self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def detectHands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def findHandsPosition(self, img, num_of_hands=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            detected_hand = self.results.multi_hand_landmarks[num_of_hands]
            for id, landmark in enumerate(detected_hand.landmark):
                h, w, _ = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return landmark_list
