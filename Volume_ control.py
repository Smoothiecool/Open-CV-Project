import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Setup webcam
cap = cv2.VideoCapture(0)

# Mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Pycaw volume control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume_ctrl.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if len(lm_list) >= 9:
                thumb_tip = lm_list[4]
                index_tip = lm_list[8]

                # Draw markers
                cv2.circle(frame, thumb_tip, 8, (0, 255, 0), -1)
                cv2.circle(frame, index_tip, 8, (0, 255, 0), -1)
                cv2.line(frame, thumb_tip, index_tip, (255, 0, 255), 2)

                # Calculate distance and set volume
                dist = get_distance(thumb_tip, index_tip)
                vol = np.interp(dist, [15, 200], [min_vol, max_vol])
                volume_ctrl.SetMasterVolumeLevel(vol, None)

                # Display volume level
                vol_percent = int(np.interp(dist, [15, 200], [0, 100]))
                cv2.putText(frame, f'Volume: {vol_percent} %', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Volume Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
