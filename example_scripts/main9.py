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

FACE_HEAD_POSE_LANDMARKS =[ 1, 33, 61, 199, 291, 263 ]

face_2d = []
face_3d = []

compensated_angle = [0,0,0]

mp_face_mesh = mp.solutions.face_mesh
#capture = cv.VideoCapture('sample_vid.mp4')
capture = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(max_num_faces = 1,
                           refine_landmarks=True,
                           min_detection_confidence = 0.5,
                           min_tracking_confidence = 0.5
) as face_mesh:

    while True:
        #if capture.get(cv.CAP_PROP_POS_FRAMES) == capture.get(cv.CAP_PROP_FRAME_COUNT):
        #    capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        ret, frame = capture.read()
        if not ret:
            break
        img_h, img_w = frame.shape[:2]
        frame = cv.flip(frame,1)
        image = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])
            
            #drawing left/right eye
            cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0,255,0), 2, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0,255,0),2, cv.LINE_AA)
            
            #drawing face outline
            cv.polylines(frame, [mesh_points[FACE_OUTLINE]], True, (255,255,255),2,cv.LINE_AA)
            #cv.polylines(frame, [mesh_points[FACE_HEAD_POSE_LANDMARKS ]], True, (255,255,255),2,cv.LINE_AA)
            
            #drawing left/right iris
            (l_cx, l_cy), l_rad = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_rad = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            l_center = np.array([l_cx, l_cy], dtype=np.int32)
            r_center = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, l_center, int(l_rad), (0,0,255), 2, cv.LINE_AA)
            cv.circle(frame, r_center, int(r_rad), (0,0,255), 2, cv.LINE_AA)
            #drawing all face mesh points as dots
            for pt in mesh_points:
                (cx,cy) = pt[0],pt[1]
                cv.circle(frame, [cx, cy], 1, (255,255,255), -1, cv.LINE_AA)


            for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
                if idx in FACE_HEAD_POSE_LANDMARKS:
                    if idx == 1:
                        nose_2d = (lm.x*img_w, lm.y*img_h)
                        nose_3d = (lm.x*img_w, lm.y*img_h, lm.z*3000)
                    
                    x,y = int(lm.x*img_w), int(lm.y * img_h)

                    face_2d.append([x,y])
                    face_3d.append([x,y,lm.z])


            face_2d = np.array(face_2d, dtype=np.float64)

            face_3d = np.array(face_3d, dtype=np.float64)

            focal_len = 1 * img_w

            camera_mat = np.array([[focal_len, 0, img_h/2], 
                                   [0, focal_len, img_w/2], 
                                   [0,0,1]])
                                       
            dist_mat = np.zeros((4,1), dtype = np.float64)

            success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, camera_mat, dist_mat)

            rot_mat, jac = cv.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rot_mat)
            x = angles[0]*360
            y = angles[1]*360
            z = angles[2]*360


            #nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rot_vec, trans_vec, camera_mat, dist_mat)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0]+y*10), int(nose_2d[1] -x * 10))
    
            cv.line(frame,p1,p2,(255,255,0),3)

            face_2d = []
            face_3d = []
        cv.imshow('Main',frame)
        key = cv.waitKey(1)

        if( key == ord('q')):
            break
        if( key == ord('c')):
            pass

capture.release()
cv.destroyAllWindows()


