import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 480) 


i = 1
while cap.isOpened():
    ret, frame = cap.read()

    i += 1


    if not ret:
        break

    try:
        if i % 50 == 0: 
            results = model.predict(frame)

        for r in results:
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[0][:4])
            conf = float(r.boxes.conf)
            cls = int(r.boxes.cls)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            roi = frame[y1:y2, x1:x2]

    except Exception as e:
        continue

    cv2.imshow("frame", frame)
    key = cv2.waitKey(50)  # Store the key pressed

    if key == 27:  # Check for the 'Esc' key
        break


cap.release()
cv2.destroyAllWindows()
