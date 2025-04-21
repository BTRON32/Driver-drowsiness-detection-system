from scipy.spatial import distance# scipy to calculate distance between landmarks
from imutils import face_utils#to get landmarks of left and right eye
from pygame import mixer#to play music(warning sound)
import imutils#basic image processing function like translation, rotation, resizing, detecting edges,etc
import dlib
import cv2


mixer.init()#initialize mixer module here 
mixer.music.load("music.wav")#load our music file, pass path name and extension of file, as we have already stored music file in project folder, we don't need to mention the path

#function to calc eye aspect ratio
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])# to store euclidean dist between points 1 and 5
	B = distance.euclidean(eye[2], eye[4])#to store euclidean dist between points 2 and 4
	C = distance.euclidean(eye[0], eye[3])#to store euclidean dist between points 0 and 3
	ear = (A + B) / (2.0 * C)#to calculate aspect ratio(ear)=ear=(sum of vertical dist)/(2*horizontal dist of eye)
	return ear# it is needed to calc ear as value of ear remains constant whenever eye opens but it suddenly drops when eye closes, because when eye closes, value of A+B decreases, most effective way to detect way to detect eye blink, calc ear 2 times for left and right eye individually and calc avg of same to calc final ear
	# check if ear value is less than a threshold value for a certain frames(certain period of time), then we will play an alarm, all depends of fps of video webcam
thresh = 0.25
frame_check = 20#as we do not want alarm to sound on blinking the eye, take frame_check variable and set its value to 20, if value of flag is greater than or = frame_check value, then only we will display warning message and play a warning tune
detect = dlib.get_frontal_face_detector()#initializing frontal face detector, get_frontal_face_detector is a built in function of dlib-it is more fast and effective, it does not accept any parameters and when we call this it returns sum of pretrained hog and linear hvm face detector included in dlib library, hog=histogram of oriented gradients
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")#initialise share predictor, pass name of .dat file- it will detect 68 landmarks from our face-landmarks are points on our face which includes eyes,eyebrows, jawline, lips,nose, etc, snapchat filters use landmarks as ref point, it detects 6 points(landmarks) in eyes, sidepoints are long(horizontal points) and remaining are short(vertical) points which are used to calculate aspect ratio(ear)=ear=(sum of vertical dist)/(2*horizontal dist of eye)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]#take 2 points, these are eye landmarks of left eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]#take 2 points, these are eye landmarks of right eye
cap=cv2.VideoCapture(0)#Video capture is a built in function, it will return frames that it will detect from camera, 0 here means I am using primary camera here
flag=0#flag is frame count
while True:#infinite loop
	ret, frame=cap.read()#read is build in function, it returns 2 values, 1st one is a boolean variable which will return true or false depending on if frame is available or not which is being stored in ret, 2nd is image array vector being stored in frame variable
	frame = imutils.resize(frame, width=450)#resize frame using imutild resize function and set its width to 450
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convert color of frame, from bgr to gray scale
	subjects = detect(gray, 0)#detect faces from gray scale frames, here detect is detector, gray is gray scale frames and 0 is index, initialize this in a variable name subjects
	for subject in subjects:
		shape = predict(gray, subject)#detect landmarks on face, here predict is predictor, gray is gray scale frames and subject is detector output, store this in variable named shape
		shape = face_utils.shape_to_np(shape)#converting shapes to a list of x,y coordinates
		leftEye = shape[lStart:lEnd]#pass landmarks to extract left and right eye
		rightEye = shape[rStart:rEnd]#once we have detected left and right eye
		leftEAR = eye_aspect_ratio(leftEye)#let us calc individual Ear using ear function and for that we will call function with parameter as leftEye and ear value returned will be stored in variable name leftear
		rightEAR = eye_aspect_ratio(rightEye)#let us calc individual Ear using ear function and for that we will call function with parameter as rightEye and ear value returned will be stored in variable name leftear
		ear = (leftEAR + rightEAR) / 2.0#actual ear will be avg of left and right ear
		leftEyeHull = cv2.convexHull(leftEye)#convex hull is minimum boundary of an object that can completely enclose/wrap the , convexHull is built in method and we will add lefteye
		rightEyeHull = cv2.convexHull(rightEye) #convexHull is built in method and we will add lefteye
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)#contours-is a curve which will join all points lying on boundary of the object, red line is convex hull and green line are contours, drawContours function is a built in function, accepts 5 arguments where 1st one is image on which we need to draw contour, 2nd one is contour =indicated contour, instead we are using hull here, 3rd is contour index that represents pixel coordinates that are listed in contours, using this we can indicate index of exact points that we need to draw, as we need to print all contour points we will pass any negative value(-1), 4th is color which will indicate color of line, 5th one is thickness of line
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		if ear < thresh:#if ear is less than thresh(certain value), then we consider it as close eye
			flag += 1#incremented, that is for every frame, value of flag will be incremented and now we will print it
			print (flag)#as we do not want alarm to sound on blinking the eye, take frame_check variable and set its value to 20, if value of flag is greater than or = 
			if flag >= frame_check:#we will only warn user when his ear is continuosly is less than threshold value for 20 frames, now it will warn for eye close and not for eye blink
				cv2.putText(frame, "****************ALERT!****************", (10, 30),#warning message, we want our text on frame, and 3rd parameter are coordinates of x and y axes, 4th is font on which we have to display message
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)#0.7 is thichness of font and 6th is color of font and 7th is thickness of line
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				mixer.music.play()#play the music
		else:#if ear is larger than threshold
			flag = 0#set flag to 0
	cv2.imshow("Frame", frame)#imShow is built in method to display image on window, it accepts 2 parameters, 1st one is window name=name of window on which we need to display image and 2nd is image we have to display, in our case image is stored in frame variable
	key = cv2.waitKey(1) & 0xFF#waitkey is a builtin function which allows us to display a window for a given ms or until any key is pressed, it waits for given time to destroy the window, if 0 is passed it will wait for any key is pressed
	if key == ord("q"):#to terminate this infinite loop
		break
cv2.destroyAllWindows()#we will destroy all windows
cap.release() #release cap
