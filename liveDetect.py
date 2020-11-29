import os, time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from imutils import resize



def detectExpression(roi):
    img = resize(roi, width=48, height=48) / 255
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    try:
        probs = model.predict(img)
    except ValueError:
        return ''
    predict = np.argmax(probs[0])
    prob = probs[0][predict]

    caption = f'{classes[predict]} {str(round(prob * 100))} %'
    return caption


haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0)

model_file = os.path.join('models', 'conv_model.h5')
model = load_model(model_file)
classes = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear']

start_time = time.time()
frame_count = 0
fps = 0
while True:
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
    for (x, y, w, h) in faces: 
        roi = gray[y:y+h, x:x+w]
        caption = detectExpression(roi)
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(im, caption,(x - 10, y - 10),
		        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(im, f'{fps} FPS', (20, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    
    frame_count += 1
    fps = int(frame_count / (time.time() - start_time))

    cv2.imshow('OpenCV', im) 
    key = cv2.waitKey(10) 
    if key == 27: 
        break