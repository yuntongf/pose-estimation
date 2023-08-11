import cv2 as cv
import numpy as np
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

size = [640, 480]

def estimate_pose(detection_result, frame_rgb):
    frame = np.copy(frame_rgb)
    
    for i in range(len(detection_result.face_landmarks)):
        # 3D model points: (U, V, W)
        model_points = np.array([
            (0.0, 0.0, 0.0), # Center of left eye
            (45.0, 0.0, -5.0), # East of left eye
            (0.0, -45.0, -5.0), # South of left eye
            (-45.0, 0.0, -5.0), # West of left eye
            (0.0, 45.0, -5.0), # North of left eye
            (540.0, 0.0, 0.0), # Center of right eye
            (585.0, 0.0, -5.0), # East of right eye
            (0.0, 495.0, -5.0), # South of right eye
            (495.0, 0.0, -5.0), # West of right eye
            (0.0, 585.0, -5.0), # North of right eye
        ])

        # Image points
        world_points = [(l.x, l.y, l.z) for l in detection_result.face_landmarks[i]]
        selected_world_points = np.array(world_points[468:478]) 
        image_points = np.array([[coord[0] * size[0], coord[1] * size[1]] for coord in selected_world_points])

        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype = "double"
        )
        
        # Rotation and translation vectors:
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
        
        # Project a 3D point onto the image plane.
        (nose_end_point2D, jacobian) = cv.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # Draw line to connect 3D and 2D reference point
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        # cv.line(frame, p1, p2, (255,0,0), 2)

        cv.circle(frame, (int(p2[0]), int(p2[1])), 6, (255,0,0), -1)
 
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
