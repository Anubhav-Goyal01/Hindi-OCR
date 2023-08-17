import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 240) 


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    cv2.imshow('webcam feed', frame)

    # try:
    #     results = model.predict(frame)
    #     for r in results:
    #         x1, y1, x2, y2 = map(int, r.boxes.xyxy[0][:4])
    #         conf = float(r.boxes.conf)
    #         cls = int(r.boxes.cls)
            
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #         cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # except Exception as e:
    #     print(e)
    #     continue

    if cv2.waitKey(50) & 0xFF == ord('q'):  # Delay of 50 milliseconds
        break

cap.release()
cv2.destroyAllWindows()
