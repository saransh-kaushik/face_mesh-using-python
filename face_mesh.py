import mediapipe as mp
import numpy as np
import cv2






cap = cv2.VideoCapture(0)  #Using webcam -->0

fmesh = mp.solutions.face_mesh
face = fmesh.FaceMesh(static_image_mode= True, min_tracking_confidence= 0.6, min_detection_confidence= 0.6 )

draw = mp.solutions.drawing_utils


running = True

while running:
    _ , frame = cap.read()    # _ --> return, frame --> sets up frame
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # changing format from BGR(opencv) to RGB(Face mesh)

    op = face.process(rgb)
    if op.multi_face_landmarks:
        for i in op.multi_face_landmarks:
            draw.draw_landmarks(frame, i, fmesh.FACEMESH_CONTOURS, landmark_drawing_spec= draw.DrawingSpec(color=(121,213,222), circle_radius= 1) )  #drawing landmarks : i-> list of landmrks (x,y,z),  
            #^^^ FACE_CONNECTIONS IS NOW FACEMESH_CONTOURS



    cv2.imshow("window", frame) #displayying the screen 

    if cv2.waitKey(1) == 27:           #escape key to exit the window
        cap.release() #relesing the webcam
        cv2.destroyAllWindows()
        break


