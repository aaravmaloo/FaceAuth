import cv2
import os

cam = cv2.VideoCapture(0)
home_dir = os.path.expanduser("~")

if os.path.exists(f"{home_dir}/FaceAuth_imgs"):
    pass
else:
    os.mkdir(f"{home_dir}/FaceAuth_imgs")

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    ret, frame = cam.read()
    clean_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img_path = f"{home_dir}/FaceAuth_imgs/face_detected.png"
        cv2.imwrite(img_path, clean_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Face detected! Image saved as {img_path}")
        cv2.imshow('Camera', frame)
        cv2.waitKey(1000)
        cam.release()
        cv2.destroyAllWindows()
        exit()
    
    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()