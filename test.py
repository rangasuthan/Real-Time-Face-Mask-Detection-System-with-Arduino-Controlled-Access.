import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from keras.models import load_model

# Load the face detection cascade classifier
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

threshold = 0.90

# Open the video capture device
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

font = cv2.FONT_HERSHEY_COMPLEX

# Load the trained model
model_path = 'C:/Users/91917/Face_Mask_Detection-main/Face_Mask_Detection-main/MyTrainingModel.h5'
try:
    model = load_model(model_path)
except Exception as e:
    print("Error loading the model:", e)
    exit()

def preprocessing(img):
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def get_className(classNo):
    if classNo == 0:
        return "Mask"
    elif classNo == 1:
        return "No Mask"

while True:
    success, imgOrignal = cap.read()
    if not success:
        print("Failed to read from camera.")
        break
    
    # Detect faces in the image
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    
    # Process each detected face
    for x, y, w, h in faces:
        crop_img = imgOrignal[y:y+h, x:x+w]
        img = cv2.resize(crop_img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)
        
        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.max(prediction)
        
        if probabilityValue > threshold:
            if classIndex == 0:
                cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 255, 0), -2)
                cv2.putText(imgOrignal, "Mask", (x, y-10), font, 0.75, (0, 255, 0), 2)
            elif classIndex == 1:
                cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (50, 50, 255), 2)
                cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (50, 50, 255), -2)
                cv2.putText(imgOrignal, "No Mask", (x, y-10), font, 0.75, (50, 50, 255), 2)
    
    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
