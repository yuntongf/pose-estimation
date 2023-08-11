import cv2 as cv
import numpy as np
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

## Uses projection with a mag_factor of 10, with the midpoint of eyes (below center of forehead and above nose)
## as the reference point

size = [640, 480]

def estimate_pose(detection_result, frame_rgb):
    frame = np.copy(frame_rgb)
    
    for i in range(len(detection_result.face_landmarks)):
        # World points
        world_points = [(l.x - 0.5, l.y - 0.5, l.z) for l in detection_result.face_landmarks[i]] # normalize so that origin is at center of image
        left_eye_center = world_points[468]
        right_eye_center = world_points[473]
        
        mag_factor = 10
        # reference point is the average of centers of left and right eye
        mid_x = ((left_eye_center[0] * mag_factor + 0.5) * size[0] + (right_eye_center[0] * mag_factor + 0.5) * size[0]) / 2
        mid_y = ((left_eye_center[1] * mag_factor + 0.5) * size[1] + (right_eye_center[1] * mag_factor + 0.5) * size[1]) / 2
        cv.circle(frame, (int(mid_x), int(mid_y)), 10, (255,0,0), -1)

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
