# Pose Estimation

## Perspective-n-Point
Knowing location of n 3D points on an object and the corresponding 2D projections on the image, we find the rotation + translation and hence the pose/orientation of the object.

Head pose
<img src="https://github.com/yuntongf/pose-estimation/blob/master/assets/head-pose.gif" width="100" height="100" />

Eye pose
<img src="https://github.com/yuntongf/pose-estimation/blob/master/assets/eye-pose.gif" width="100" height="100" />


## Line & plane intercept
Imagine a light ray shooting out from the center of left iris, the ray must intercept the image plane at a point. The point should be the target of the gaze. The problem involves several steps:

1. Get the plane of iris
2. Find the vector orthogonal to the plane and crosses the center point of the iris. This will be our "light ray"
3. Find the image plane 
4. Find the intercept of image plane and the "light ray"

I have implemented the math but there are some bugs that I have not been able to figure out.

<img src="https://github.com/yuntongf/pose-estimation/blob/master/assets/intercept.gif" width="100" height="100" />

## Eye/forehead projection
This seems like a cheap hack, but: shift the origin of the image plane from top left to the center of the image, and multiply the coordinates of eyes/forehead by a given factor (mag_factor) in this case 10, the effect will be similar to expanding the coordinates "radially outward" and the resulting coordinates can be used to approximate the target of gaze. 

<img src="https://github.com/yuntongf/pose-estimation/blob/master/assets/proj_eyes.gif" width="100" height="100" />

<img src="https://github.com/yuntongf/pose-estimation/blob/master/assets/proj_forehead.gif" width="100" height="100" />

