import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import itertools

size = [1280, 960]

def get_gradient_from_points(P, Q, R):  
    x1, y1, z1 = P
    x2, y2, z2 = Q
    x3, y3, z3 = R
    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    norm = np.sqrt(np.square(a) + np.square(b) + np.square(c))
    return a/norm, b/norm, c/norm

def get_plane_line_intercept(planeNormal, planePoint, lineGradient, linePoint, epsilon=1e-6):
    ndotu = planeNormal.dot(lineGradient)
    if ndotu < epsilon: 
        print("parallel")
        return (0.0, 0.0, 0.0)

    w = linePoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * lineGradient + planePoint
    return Psi

def estimate_pose(detection_result, frame_rgb):
    frame = np.copy(frame_rgb)
    
    for i in range(len(detection_result.face_landmarks)):
        # World points
        world_points = [(l.x, l.y, l.z) for l in detection_result.face_landmarks[i]] # normalize so that origin is at center of image
        left_eye_center = world_points[468]
        left_eye_peripheral = np.array(world_points[469:472]) 

        # Get gradient of left eye plane
        a, b, c = get_gradient_from_points(left_eye_peripheral[0], left_eye_peripheral[1], left_eye_peripheral[2])

        # Get the gradient vector passing through left eye center coordinates
        lineGradient = np.array([a, b, c])
        linePoint = np.array([left_eye_center[0], left_eye_center[1], left_eye_center[2]])

        # Get image plane
        planeNormal = np.array([0.0, 0.0, -1.0]) # image plane
        planePoint = np.array([0.0, 0.0, 0.0]) # point on image plane

        # Get intersection of gradient vector and image plane
        intercept = get_plane_line_intercept(planeNormal, planePoint, lineGradient, linePoint)
        # print(intercept)
        x = (intercept[0]) * size[0]
        y = (intercept[1]) * size[1]

        # Display the intersection
        cv.circle(frame, (int(x), int(y)), 6, (255,0,0), -1)

    # Display image
    cv.imshow("output", cv.cvtColor(frame, cv.COLOR_BGR2RGB))

base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0)
    cap.set(3,size[0]) # set Width
    cap.set(4,size[1]) # set Height

    while(True):
        ret, frame = cap.read()
        # Flip camera vertically
        frame = cv.flip(frame, 1)
        # Convert the frame to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = landmarker.detect(mp_image)
        estimate_pose(detection_result, frame_rgb)

        k = cv.waitKey(30) & 0xff
        if k == 27: # Press ESC to quit
            break
    cap.release()
    cv.destroyAllWindows()
