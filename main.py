import cv2
from deepface import DeepFace
from log_path import log_path_csv
from datetime import datetime
import csv
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1680)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

#Default Counter
counter = 0


def check_face(frame):
    global face_match
    try:
        #Convert the captured frame from BGR (OpenCV default) to RGB (DeepFace expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Reset match status before checking
        face_match = False

        best_distance = float('inf')

        #Loop through each reference image loaded from your folder
        for ref_img in reference_images:
            #Skip this image if it failed to load
            if ref_img is None:
                continue

            
            #Convert the reference image from BGR to RGB
            ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            
            #Use DeepFace to compare the current frame to the reference image
            result = DeepFace.verify(frame_rgb, ref_rgb, enforce_detection=False)
            
            distance = result['distance']
            if distance < best_distance:
                best_distance = distance
            #If a match is found, set face_match to True and stop checking further images
            if result['verified']:
                face_match = True   
                break
        
        # Print the type and shape of the last reference image for debugging
        print(type(ref_img), ref_img.shape if ref_img is not None else "None")
        # #Print shapes for debug        
        return face_match, best_distance

    #Error handeling
    except Exception as e:
        print("Error in DeepFace verification:", e)
        face_match = False

def log_attempt(match_status, log_path=log_path_csv):
    timestamp = datetime.now()
    file_exists = os.path.isfile(log_path)

    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "match", "distance"])
        writer.writerow([timestamp, match_status, dist])

def load_all_images(folder_path):
    images = [] #Empty list

    #loop through every file in the specified folder
    for filename in os.listdir(folder_path):
        #only process files that end with .jpg or .png (image files)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            #Build the full path to the image file
            img_path = os.path.join(folder_path, filename)
            #Load the image using OpenCV
            img = cv2.imread(img_path)
            #If the image loads successfully, add it to the list
            if img is not None:
                images.append(img)
            else:
                #If the image fails to load, print a warning
                print(f"Warning: Could not load {img_path}")

    #Return the list of all successfully loaded images
    return images
       

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

reference_images = load_all_images("images")

if not reference_images:
    print("Error: No reference images loaded from 'images' folder.")
    exit()

while True:
    #reads one frame from your video source
    ret, frame = cap.read() #Ret is a Bool. Frame is the frefrence image data as a numpy array
    #Check if frame was grabbed correctyl
    if ret:
        #check frame number 30th frame
        if counter % 30 == 0:
            try:
                #face verification function with a copy of the current frame.
                match, dist = check_face(frame.copy())
                log_attempt(match)


            except ValueError:
                pass

        counter += 1
        #inside while loop, after you get frame:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            #Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                #Draw rectangle around eyes
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        if match:
            cv2.putText(frame, f"MATCH! Dist: {dist:.2f}", (0,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255,0),3)
        else:
            cv2.putText(frame, f"NO MATCH! Dist: {dist:.2f}", (0,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0,255),3)


        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q", "Q"):
        break
    
cap.release()
cv2.destroyAllWindows()
