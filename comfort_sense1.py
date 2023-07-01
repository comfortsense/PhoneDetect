import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from ultralytics import YOLO
import onnxruntime as ort
from collections import deque
import torch
import PyQt5

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence = 0.15, min_tracking_confidence = 0.15)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)

expression_detector = ort.InferenceSession("./emotion_detector.onnx", providers = ["CPUExecutionProvider"])
idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}

drowsiness_detector = YOLO("./SDV_Drowsiness.pt")


distracted_emoji = cv2.imread("emoji/distracted.png", -1)
phone_detected_emoji = cv2.imread("emoji/phone_detected.png", -1)
cigarette_detected_emoji = cv2.imread("emoji/cigarette_detected.png", -1)
drowsy_emoji = cv2.imread("emoji/drowsiness.png", -1)
try:
    emotion_emojis = {idx: cv2.imread(f"emoji/{emotion.lower().strip()}.png", -1) for idx, emotion in idx_to_class.items()}
except Exception as e:
    print(f"Error loading emoji: {e}")

# Function to add emoji
def add_emoji(frame, emoji, x, y, size=(25, 25)):
    emoji = cv2.resize(emoji, size)
    for c in range(0, 3):
        frame[y:y+size[1], x:x+size[0], c] = emoji[:, :, c] * (emoji[:, :, 3] / 255.0) + frame[y:y+size[1], x:x+size[0], c] * (1.0 - emoji[:, :, 3] / 255.0)
    return frame

def draw_label_and_emoji(frame, text, emoji, x, y, text_size=0.7, emoji_size=(25, 25), padding=7):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_size, 1)
    emoji = cv2.resize(emoji, emoji_size)

    # Draw a black background rectangle with transparency
    overlay = frame.copy()
    start_x, start_y, end_x, end_y = x - padding, y - padding, x + text_width + emoji_size[0] + padding * 3 - 5 , y + max(text_height, emoji_size[1]) + padding - 5
    cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)

    # Blend the original frame with the overlay based on the alpha
    alpha = 0.6  # Higher alpha gives more transparency
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Render the text on the frame
    cv2.putText(frame, text, (x, y + text_height), cv2.FONT_HERSHEY_SIMPLEX, text_size, (230, 230, 230), 1, cv2.LINE_AA)
    
    # Render the emoji at the center of the text on the frame
    frame = add_emoji(frame, emoji, x + text_width + padding, y - 1)

    return frame

def draw_fps(frame, text, x, y, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, thickness=1, color=(170,170,170), padding=10):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    start_x, start_y = x, y - text_height - padding  # subtracting text_height and padding from y to adjust the position
    end_x = x + text_width + padding * 2  # padding on both sides of the text
    end_y = y + padding

    # rectangle background
    frame = cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0,0,0), -1)

    # text
    cv2.putText(frame, text, (x + padding, y), font, scale, color, thickness)

    return frame

def drowsiness_check(preds):
    for pred in preds[0].boxes.data:
        if pred[-1] == 3.0:
            return True
    return False

def phone_check(preds):
    for pred in preds[0].boxes.data:
        if pred[-1] == 1.0 and pred[-2] > 0.42:
            return True
    return False
    
def cigarette_check(preds):
    for pred in preds[0].boxes.data:
        if pred[-1] == 2.0:
            return True
    return False


def bbx_extractor(results, frame_width, frame_height):
    xmin = int((min(results.multi_face_landmarks[0].landmark[234].x, results.multi_face_landmarks[0].landmark[1].x, 
   results.multi_face_landmarks[0].landmark[10].x, results.multi_face_landmarks[0].landmark[454].x, 
   results.multi_face_landmarks[0].landmark[152].x) - 0.02)*frame_width)
    
    ymin = int((results.multi_face_landmarks[0].landmark[10].y - 0.02)*frame_height)
    
    xmax = int((max(results.multi_face_landmarks[0].landmark[234].x, results.multi_face_landmarks[0].landmark[1].x, 
   results.multi_face_landmarks[0].landmark[10].x, results.multi_face_landmarks[0].landmark[454].x, 
   results.multi_face_landmarks[0].landmark[152].x) + 0.02)*frame_width)
    ymax = int((results.multi_face_landmarks[0].landmark[152].y + 0.02)*frame_height)
    return xmin, ymin, xmax, ymax

def draw_landmarks(frame, face_landmarks, video_width, video_height):
    depiction_list = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 
                     378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 
                     21, 54, 103, 67, 109, 33, 159, 145, 243, 362, 385, 380, 263, 168, 6, 197, 
                     195, 5, 4, 220, 440, 48, 278, 61, 39, 181, 0, 17, 269, 405, 291]
    for i in depiction_list:
        x = int(face_landmarks.landmark[i].x * video_width)
        y = int(face_landmarks.landmark[i].y * video_height)
        #cv2.circle(frame, (x, y), 2, (100, 100, 0), -1)
        cv2.rectangle(frame, (x, y), (x + 1, y + 1), (225, 225, 225), -1)
    
    
cap = cv2.VideoCapture(0)
img_w = cap.get(3)
img_h = cap.get(4)
img_c = 3
frame_cnt = 0
direction_history = deque(maxlen = 3)

drowsiness_frame_cnt = 0
drowsiness_stack = deque(maxlen = 2)

fps_list = deque(maxlen = 30)
frame_counter = 0

cv2.namedWindow("ComfortSense Prototype", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ComfortSense Prototype", 1920, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    start = time.time()
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    preds = drowsiness_detector(frame)
    frame.flags.writeable = False
    results = face_mesh.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    face_3d = []
    face_2d = []
    
    if frame_cnt % 30 == 0:
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x*img_w, lm.y*img_h)
                            nose_3d = (lm.x*img_w, lm.y*img_h, lm.z*3000)

                        x, y = int(lm.x*img_w), int(lm.y*img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                    if idx >= 292:
                        break

                face_2d = np.array(face_2d, dtype = np.float64)
                face_3d = np.array(face_3d, dtype = np.float64)

                focal_length = 1*img_w
                cam_matrix = np.array([[focal_length, 0, img_h/2], 
                                       [0, focal_length, img_w/2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype = np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x = angles[0]*360
                y = angles[1]*360
                z = angles[2]*360

                if y < -7.25:
                    direction_history.append("Looking Left")
                elif y > 7.25:
                    direction_history.append("Looking Right")
                elif x < -7:
                    direction_history.append("Looking Down")
                elif x > 7.25:
                    direction_history.append("Looking Up")
                else:
                    direction_history.append("Forward")
                    
                frame_cnt = 0
    elif frame_cnt % 30 != 0 and len(direction_history) == 3:
        if (len(set(direction_history)) == 1 and next(iter(set(direction_history))) != "Forward") == True:
            distracted_text = "Distracted"
            frame = draw_label_and_emoji(frame, distracted_text, distracted_emoji, 20, 30)
            
    frame_cnt += 1
    if results.multi_face_landmarks:
        
        xmin, ymin, xmax, ymax = bbx_extractor(results, img_w, img_h)
        if (xmin > 0 and ymin > 0 and xmax > 0 and ymax > 0):
            patch = cv2.cvtColor(frame[ymin:ymax, xmin:xmax, :], cv2.COLOR_BGR2RGB)
            patch = np.expand_dims(cv2.resize(patch, (224, 224)), 0).astype(np.float32)
            emotion = idx_to_class[expression_detector.run(None, {"input_1": patch})[0].argmax()]
            #nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
            emotion_text = idx_to_class[expression_detector.run(None, {"input_1": patch})[0].argmax()]
            emotion_emoji = emotion_emojis[expression_detector.run(None, {"input_1": patch})[0].argmax()]
            frame = draw_label_and_emoji(frame, emotion_text, emotion_emoji, 20, 190)

    if phone_check(preds):
        phone_text = "Phone detected"
        frame = draw_label_and_emoji(frame, phone_text, phone_detected_emoji, 20, 70)

    if cigarette_check(preds):
        cigarette_text = "Cigarette detected"
        frame = draw_label_and_emoji(frame, cigarette_text, cigarette_detected_emoji, 20, 110)
    
    if drowsiness_frame_cnt % 15 == 0:
        drowsiness_stack.append(drowsiness_check(preds))
        drowsiness_frame_cnt = 1
    drowsiness_frame_cnt +=1
    if len(drowsiness_stack) == 2 and (drowsiness_stack.count(True) == 2):
        drowsiness_text = "Drowsiness"
        frame = draw_label_and_emoji(frame, drowsiness_text, drowsy_emoji, 20, 150)
        
#     emotion_text = idx_to_class[expression_detector.run(None, {"input_1": patch})[0].argmax()]
#     emotion_emoji = emotion_emojis[expression_detector.run(None, {"input_1": patch})[0].argmax()]
#     frame = draw_label_and_emoji(frame, emotion_text, emotion_emoji, 20, 190)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    fps_list.append(fps)

    if frame_counter >= 20: # start calculating average FPS after 20 frames
        avg_fps = sum(fps_list)/len(fps_list)
        frame = draw_fps(frame, f'FPS: {avg_fps:.0f}', 500, 50)
    else:
        frame = draw_fps(frame, f'FPS: {int(fps)}', 500, 50)

    frame_counter += 1

    if results.multi_face_landmarks:
        draw_landmarks(frame, results.multi_face_landmarks[0], img_w, img_h)

    cv2.imshow('ComfortSense Prototype', frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

