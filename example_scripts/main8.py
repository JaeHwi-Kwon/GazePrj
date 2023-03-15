import mediapipe as mp
import numpy as np
import cv2 as cv

LEFT_EYE = [ 362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398 ]
RIGHT_EYE = [ 33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246 ]

LEFT_IRIS = [ 474,475,476,477 ]
RIGHT_IRIS = [ 469,470,471,472 ]

FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 
                176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                162, 21, 54, 103, 67, 109]

mp_face_mesh = mp.solutions.face_mesh
capture = cv.VideoCapture('sample_vid.mp4')

with mp_face_mesh.FaceMesh(max_num_faces = 1,
                           refine_landmarks=True,
                           min_detection_confidence = 0.5,
                           min_tracking_confidence = 0.5
) as face_mesh:

    while True:
        if capture.get(cv.CAP_PROP_POS_FRAMES) == capture.get(cv.CAP_PROP_FRAME_COUNT):
            capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        ret, frame = capture.read()
        if not ret:
            break
        
        img_h, img_w = frame.shape[:2]
        image = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])
            

            cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0,255,0), 2, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0,255,0),2, cv.LINE_AA)
            
            cv.polylines(frame, [mesh_points[FACE_OUTLINE]], True, (255,255,255),2,cv.LINE_AA)
            
            (l_cx, l_cy), l_rad = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_rad = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            l_center = np.array([l_cx, l_cy], dtype=np.int32)
            r_center = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, l_center, int(l_rad), (0,0,255), 2, cv.LINE_AA)
            cv.circle(frame, r_center, int(r_rad), (0,0,255), 2, cv.LINE_AA)

            for pt in mesh_points:
                (cx,cy) = pt[0],pt[1]
                cv.circle(frame, [cx, cy], 1, (255,255,255), -1, cv.LINE_AA)
        cv.imshow('Main',frame)
        key = cv.waitKey(1)
        if( key == ord('q')):
            break

capture.release()
cv.destroyAllWindows()