import os, time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from imutils import resize



def detectExpression(roi):
    '''
    Uses model to detect expression in region of interest

    arguments:
        roi - 'region of interest' snipped from larger image. 2-d array of pixels.

    return:
        caption - string containing predicted label and probability
    '''
    # Preprocess image data and reshape to (1, h, w, 1)
    img = resize(roi, width=48, height=48) / 255
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    # Feed forward through model. Sometimes the image is not resized
    # properly, so handle exception if occurs
    try:
        probs = model.predict(img)
    except ValueError:
        return ''

    # Get label and associated probability
    predict = np.argmax(probs[0])
    prob = probs[0][predict]

    caption = f'{classes[predict]} {str(round(prob * 100))} %'
    return caption



# Setup face detection and video stream
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file) 
webcam = cv2.VideoCapture(0)

# Load model
model_file = os.path.join('models', 'best_model2.h5')
model = load_model(model_file)
classes = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear']

# Setup for running average of FPS
start_time = time.time()
frame_count = 0
fps = 0
while True:
    # Get a frame from video stream and convert to gray scale
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 

    # Predict expression in each face
    for (x, y, w, h) in faces: 
        roi = gray[y:y+h, x:x+w]
        caption = detectExpression(roi)

        # Annotate original image 
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(im, caption,(x - 10, y - 10),
		        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(im, f'{fps} FPS', (20, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    
    # Update FPS
    frame_count += 1
    fps = int(frame_count / (time.time() - start_time))

    # Display image. Escape key to end loop
    cv2.imshow('OpenCV', im) 
    key = cv2.waitKey(10) 
    if key == 27: 
        break