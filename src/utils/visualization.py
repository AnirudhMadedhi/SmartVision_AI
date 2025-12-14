
import cv2

def draw_boxes(image, boxes, labels):
    for box, label in zip(boxes, labels):
        x1,y1,x2,y2 = box
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(image,label,(x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    return image
