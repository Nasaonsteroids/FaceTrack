import cv2
from deepface import DeepFace
from path import file_path #My own filepath file with my exact filepath

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#Window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Default Counter
counter = 0

#Default face_match status
face_match = False

#My own refrence image
reference_img = cv2.imread(file_path)

#If the script cannot detected/load the refence image 
if reference_img is None:
    print(f"Error: Could not load reference image from {file_path}")
    exit()


def check_face(frame):
    global face_match
    try:
        #Set the RGB colors of the frame and refrence image since its BGR and not RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        #Print shapes for debug
        print("frame_rgb shape:", frame_rgb.shape)
        print("ref_rgb shape:", ref_rgb.shape)

        #skip the exception and instead process the entire input image
        result = DeepFace.verify(frame_rgb, ref_rgb, enforce_detection=False)
        
        print(result) #Debug info
        face_match = result['verified']
    #Error handeling
    except Exception as e:
        print("Error in DeepFace verification:", e)
        face_match = False

        


while True:
    #reads one frame from your video source
    ret, frame = cap.read() #Ret is a Bool. Frame is the frefrence image data as a numpy array
    #Check if frame was grabbed correctyl
    if ret:
        #check frame number 30th frame
        if counter % 30 == 0:
            try:
                #face verification function with a copy of the current frame.
                check_face(frame.copy())
            except ValueError:
                pass

        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255,0),3)
        else:
            cv2.putText(frame, "NO MATCH!", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0,255),3)

        cv2.imshow("video", frame)


    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()