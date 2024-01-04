import cv2
import mediapipe as mp
import pickle

model=pickle.load(open('model.pk1','rb'))
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cam = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5, model_complexity=0)as hands:
    while cam.isOpened():
        success, image = cam.read()
        imageWidth, imageHeight = image.shape[:2]
        if not success:
            continue
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                data=[]
                for point in mp_hands.HandLandmark:
                    normalizedLandmark = hand_landmarks.landmark[point]
                    data.append(normalizedLandmark.x)
                    data.append(normalizedLandmark.y)
                    data.append(normalizedLandmark.z)
                print(len(data))
                result=model.predict([data])
                print(result)

                font=cv2.FONT_HERSHEY_SIMPLEX
                fontScale=1
                org=(50,50)
                color=(255,0,0)
                thickness=2
                img=cv2.putText(image, result[0],org,font, fontScale, color, thickness,cv2.LINE_AA)
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF ==27:
            break
cam.release()


