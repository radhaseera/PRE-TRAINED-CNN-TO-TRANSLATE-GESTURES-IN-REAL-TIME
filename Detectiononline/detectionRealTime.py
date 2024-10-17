import cv2
import numpy as np
from keras.models import load_model
import time

# Variable declarations
prediction = ''
score = 0
bgModel = None

Labels = ["Bad", "Deaf", "Fine", "Good", "Goodbye", "Hearing", "Hello", "How are you", "Nice to meet you", "Please", "See you later", "See you tomorrow", "Sorry", "Thank you", "What is your name"]

# Load pre-trained model from file
model = load_model('models/resnet_data_word.hdf5')

# Function to predict the hand gesture
def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    result = Labels[np.argmax(pred_array)]
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    return result, score

# Function to remove background from the frame
def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# Region of interest dimensions
cap_region_x_begin = 0.6
cap_region_y_end = 0.6

# Threshold parameters
threshold = 60
blurValue = 41
bgSubThreshold = 50
learningRate = 0

# Prediction threshold
predThreshold = 95

isBgCaptured = 0  # Flag to indicate if the background is captured

# Camera setup
camera = cv2.VideoCapture(0)
camera.set(10, 200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)

sign_predict = ""
count = 0
s = 0

msg = ""
word = ""
old_sign_predict = ''

# Main loop for capturing and processing frames
while camera.isOpened():
    # Read frame from webcam
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    frame = cv2.flip(frame, 1)

    # Draw rectangle for detection region
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 60),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 0, 255), 5)
    
    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    # If background is captured
    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[60:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if (np.count_nonzero(thresh)/(thresh.shape[0]*thresh.shape[0]) > 0.2):
            if (thresh is not None):
                target = np.stack((thresh,) * 3, axis=-1)
                target = cv2.resize(target, (50, 50))
                target = target.reshape(1, 50, 50, 3)
            
                prediction, score = predict_rgb_image_vgg(target)
                sign_predict = prediction

                if (old_sign_predict == sign_predict):
                    count += 1
                    s += 1

                if (score >= predThreshold and count > 10):
                    msg ='Sign:' +prediction +', Conf: ' +str(score)+'%'
                    if(s > 30):
                        word = prediction
                        s = 0
    
                    count = 0
                old_sign_predict = sign_predict
                
        else:
            msg = ""
            s = 0

    cv2.putText(frame, msg, (280, 30), font, fontScale, color, thickness)

    thresh = None

    # Handle keyboard inputs
    k = cv2.waitKey(10)
    if k == ord('q'):  # Press 'q' to quit
        break
    elif k == ord('b'):  # Press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

        isBgCaptured = 1
        cv2.putText(frame, "Background captured", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255), 10, lineType=cv2.LINE_AA)
        time.sleep(2)
        print('Background captured')

    elif k == ord('r'):  # Press 'r' to reset the background
        bgModel = None
        isBgCaptured = 0
        cv2.putText(frame, "Background reset", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255), 10, lineType=cv2.LINE_AA)
        print('Background reset')
        time.sleep(1)

    elif k == ord('w'):  # Press 'w' to delete the predicted word
        word = ''
        print('Word Delete')

    # Display predicted word
    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blackboard, "Word Predict", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0,0))
    cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255))
    ress = np.hstack((frame, blackboard))
    cv2.imshow('original', cv2.resize(ress, dsize=None, fx=1, fy=1))

cv2.destroyAllWindows()
camera.release()

