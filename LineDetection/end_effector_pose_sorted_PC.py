#for a structured pointcloud (x*y, 1650*1700 or something)
import numpy as np
import pandas as pd

SUBSAMPLE_SIZE = 10 # number of pixels that wil be chosen for the subsampeled Pointcloud, a higher value means a bigger search-grid
GRID_SPACING = 0.1

pointcloud_size= (1944,1200)

Pointcloud_input:list[list]
points_input:list[list] #chosen points
vectors_input:list[list] #vector representing the direction of the path taken by the end effector (point to point)


#make shure the Poincloud and points are in the same frame

#ALGORITHM:
"""
for each point in points
extract a subset from points based on the location of the current point (this can be done with the indexes rather than the locations of the points)
Get the vector corresponding to the current point and use it to create a plane orthogonaly to the vector (which acts as the normal vector)

In the extracted subset, find all points_ that are on the or close to the plane (either by aprox solution or by creating two planes)

Choose a set distance from the current point and select all points_ on the plane that matches that distance (by some aproximation)
For all remaining points__, choose two points__ that have a distance between each other greater than the distance beteween the current point and the chosen points__

The two chosen points__2 along with the current point are used to find the pose and angle required by the end effector, and a offset is added to make shure the weld is pushed

do this for all points in the selected subsample

NOTE: the corner is not handeled actively in this implementation, but by adding a second search or by adding enough "push-angle", it might be solved "by a lucky mistake"

Returns a list containing the position (xyz) and pose (xyzw, or ijkw (quaternions))
"""

def mean(a,b): 
    return [(a[0]+ b[0])/2, (a[1] +b[1])/2, (a[2], b[2])/2]
def restructure_pointcloud(pointcloud:list[list], pc_shape:tuple[int])-> list[list[list]]:
    # pointcloud = np.array(pointcloud)
    structured_pointcloud = np.reshape(pointcloud, (pc_shape[0], pc_shape[1], 3))
    return structured_pointcloud


SUBSAMPLE_SIZE = SUBSAMPLE_SIZE//2 # this way the input declares the whole area

def get_index_of_point(point:list, Pointcloud:list[list[list]]) -> list:
    assert len(np.shape(Pointcloud)) ==3, f"shape of poincloud is not 3-dimensional, it is {np.shape(Pointcloud)}"
    for i in range(len(np.shape(Pointcloud)[0])):
        for j in range(len(np.shape(Pointcloud)[1])):
            if point == Pointcloud[i,j]:
                return [i,j]
    return []

def create_plane(point:list, vector:list, grid_size:int, GRID_SPACING:float):
    """creates a set of evenly spaced points in a plane defined by the point and vector, with size = 2*grid_size and spacing = GRID_SPACING

    Args:
        point (list): origin point
        vector (list): normal vector
        grid_size (int): gives the limit of the appointed points
        GRID_SPACING (float): gives the spacing between the points
    """
    x,y,z = vector
    x0, y0, z0 = point
    plane_grid = []
    for i in range(-grid_size, grid_size, GRID_SPACING):
        for j in range(-grid_size, grid_size, GRID_SPACING):
            for k in range(-grid_size, grid_size, GRID_SPACING):
                plane= [i*x+x0, j*y +y0, k*z+z0]
                plane_grid.append(plane)

    return plane_grid

def get_3D_min_max(subsampeled_pointcloud:list[list[list]])->list:
    #choosing the first instance to avoid NaN problems
    min=subsampeled_pointcloud[0,0]
    max=subsampeled_pointcloud[0,0]
    for row in subsampeled_pointcloud:
        for values in row:
            if np.NaN in values:#skips instances with NaN-values
                continue
            min = np.minimum(min, values)
            max = np.maximum(max, values)
    return min, max

def drop_NaN(Pointcloud:list[list]): #this might only work on np.NaN objects, testing needed for the Zivid/Pandas NaN values
    """custom function to drop NaN values, as the whole point needs to be droped, not just the NaN-value
        bad way of doing it, but it works,
        volatile as it only checks the first value, could potentially be a problem based on the data

    Args:
        pointcloud (list[list]): subsampeled pointcloud
    """
    clean_list = []
    for el in Pointcloud:
        # print(f"{el[0] = }")
        if ~np.isnan(el[0]):
            clean_list.append(el)
    return clean_list


def get_most_similar_points(grid:list[list], pointcloud:list[list[list]], threshold= 0.5)->list[list]: #realy expensive calculation O(M*N), threshold is not set well yet
    """finds the points with the least euclidian distance from any of the points in the projected plane

    Args:
        grid (list[list]): plane-grid
        pointcloud (list[list[list]]): subsampeled pointcloud
        threshold (float, optional): distance threshold between points, should be about the same distance as between 2-3 points in the pointcloud,
                    Defaults to 0.5, should be calculated or found with testing

    Returns:
        list[list]: _description_
    """
    #flatten pointcloud if its still a 2D list containing the points
    chosen_points = []
    if len(np.shape(pointcloud))==3:
        pointcloud = np.reshape(pointcloud, (-1,3)) #flattens the subsampeled pointcloud as we dont need the shape anymore
    pointcloud = drop_NaN(pointcloud)
    for pc_point in pointcloud:
        for grid_point in grid:
            if np.linalg.norm(grid_point-pc_point) <= threshold:
                chosen_points.append(pc_point) #important that the poincloud_point is chosen for accuracy/consistency
    return chosen_points



def get_mapping_points(origin_point:list, points:list[list], input_distance:float):
    assert np.shape(points)[0] >= 2, "Error, to few points were extracted from the previous steps"
    most_similar= []
    selected_points = []

    min_distance = 999.99 #arbitrary high value, will be overvwritten
    #check points to see the minimum distance between them:
    #innefective way of doing it, but fast to code, should be relatively few points to calculate this for
    for point_a in points:
        for point_b in points:
            if point_a== point_b:
                continue #skips when on itsself
            dist =np.linalg.norm(point_a-point_b)
            if dist < min_distance and dist != 0: #this is a double check for the same point, should be redundant.
                min_distance = dist
    #check in case the previous part has a problem
    if min_distance >= 999.99:
        min_distance = 0.5
    for point in points:
        dist =np.linalg.norm(point-origin_point) #calculates the distance between the current point and the chosen point
        if dist <= input_distance + 2*min_distance  and dist >= input_distance - 2*min_distance: #check if between the chosen points, the n* distance is somewhat arbitrary
            most_similar.append(point)

    #choose last two points,
    for point_a in most_similar:
        for point_b in most_similar:
            if point_a == point_b:
                continue
            dist =np.linalg.norm(point_a-point_b)
            if dist >= input_distance: #make shure the points are on different planes
                selected_points = [point_a, point_b] #choose the first pair odf points that satisfies the conditions
                break
    assert len(np.shape(selected_points)), "Error, no two points were possible to select satisfying the requirements"
    return selected_points


#rotations:
def skew(axis): #returns skew representation of the 3x1 vector
    assert(len(axis) == 3)
    return np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0],0]])

def rotation(axis, theta = 0.0): #Rotation metrix from axis and magnitude w, theta
    return np.eye(3) + np.sin(theta) * skew(axis) + (1 - np.cos(theta)) * skew(axis) @ skew(axis)

def deg_to_rad(deg):
    return (deg/360)*2*np.pi


#first iteration: get the pose-vector based on the method above, for the next ones, change its orientation based on the calculated input-vector
def get_pose(point:list, PointCloud:list[list[list]], vector:list, angle_offset:int=0, chosen_point_distance:float = 10) -> list:
    """calculates the end-effector pose based on input point, vector and pointcloud

    Args:
        point (list): current point for the welding path
        PointCloud (list[list[list]]): input pointcloud (structured and full)
        vector (list): current vector indicating direction of travel
        angle_offset (int): osset added to the pose "away" from direction of travel
        chosen_point_distance (float, optional): distance in mm away from the current point the points used to calculate the pose is chosen by. Defaults to 10.

    Returns:
        list: _description_
    """
    #subsample the pointcloud:
    index_point = get_index_of_point(point=point, Pointcloud=PointCloud)
    assert len(index_point), "no index found for the current point"
    sub_cloud = PointCloud[index_point[0]-SUBSAMPLE_SIZE:index_point[0]+SUBSAMPLE_SIZE][index_point[1]-SUBSAMPLE_SIZE:index_point[1]+SUBSAMPLE_SIZE]
    #get the maximum size of the sampeled area, makes sure all points within this area is taken into account
    min, max = get_3D_min_max
    grid_size = np.max(max - min)
    plane_grid = create_plane(point=point, vector=vector, grid_size=grid_size)
    intersection_points= get_most_similar_points(grid=plane_grid, pointcloud=sub_cloud, threshold=0.5)
    chosen_points = get_mapping_points(point, intersection_points, chosen_point_distance)

    #calculate the pose


    # current_vec = vectors[0] #get the first vector from the
    middle_pos= mean(chosen_points[0], chosen_points[1]) #works with both array an numpy-array (for numpy this is overkill tho)
    first_pose_vec = middle_pos - point
    #rotate the extracted pose around the axis
    rotation_axis= np.cross(vector, first_pose_vec )
    angle_offset_rad = deg_to_rad(angle_offset)
    pose_inv = rotation(rotation_axis, angle_offset_rad) * pose_inv #NOTE make sure this rotates the correct way (pre/post multiplication and sign of rotation)
    #reversing the pose as it is pointing out of the workpiece
    pose= -pose_inv

    return pose

def change_pose(pose, current_vec, next_vec):
    current_vec = np.array(current_vec)
    next_vec = np.array(next_vec)
    pose = np.array(pose)
    assert (np.abs(current_vec)* np.abs(next_vec)) != 0, f"Error, one of the input-vectors: {current_vec =}, {next_vec = } is a null-vector"
    angle_rad= np.arcsin(np.abs(np.cross(current_vec, next_vec))/ (np.abs(current_vec)* np.abs(next_vec)))
    angle_deg = angle_rad/(2*np.pi) * 360
    # axis = np.cross(current_vec, pose)/np.abs(np.cross(pose, current_vec)) #normalizing the axis vector
    axis = np.cross(current_vec, next_vec)/np.abs(np.cross(current_vec, next_vec))
    new_pose= rotation(axis, angle_deg) * pose # NOTE the negative sign is there based on the sequence in which
    #maby normalize?
    return new_pose




def estimate_poses(points:list[list], PointCloud_in:list[list], vectors:list[list], angle_offset:int, chosen_point_distance:float = 10):
    poses = []
    """ Calculates the end-effector pose for a welding path based on the surounding walls and end-effectors movement-vector

    Args:
        points (list[list]): The current point corrolating to the current vector.
        vectors (list[list]): The current vector, giving the dirction of the next points, meaning it shows where the end-effector is going to move.
        PointCloud (list[list[list]]): The structured pointcloud, containing x,y,z-coordinates and NaN-values
        angle_offset (int): Given in degrees, gives the angle the end-effctor is going to "push" the weld with, with 0 being perpendicular to the surface, and 90 would
                            send the end effector "straight into" the part it is going to weld, alligning the end-effector with the direction of travel.
        chosen_point_distance (float): distance (in mm?) to where the points used to calualte the pose are chosen

    Returns:
        list: End-effector pose for the input-point
    """
    #extract the first pose
    # NOTE: due to edge-variations we want to extract the 2nd or 3rd point, not the first, then superimpose this on the first (and second/third) point
    poses = []
    point_num = 2 #
    
    #restructure the pointcloud
    PointCloud= restructure_pointcloud(PointCloud_in, pointcloud_size)
    
    pose= get_pose(points[point_num], PointCloud, vectors[point_num], 30, 10)

    for i in range(len(point_num+1)):
        poses.appen(pose)

    #for all remaining points, calculate the offset-vector and rotate the pose based on this rotation
    for i in range(np.shape(points)[0]):
        if i <= np.shape(poses)[0]-1: #skip all poses already inserted 2 or 3, ... -1 due to indexing from 0
            continue
        poses.append(change_pose(poses[i-1], vectors[i-1], vectors[i]))

    if np.shape(poses)[0] ==  np.shape(vectors)[0]:
        return poses
    #if there is a logic-error in the previous steps, this should fix it, might be overkill
    elif np.shape(poses)[0] >  np.shape(vectors)[0]:
        while np.shape(poses)[0] >  np.shape(vectors)[0]:
            poses.pop(0)
    elif np.shape(poses)[0] <  np.shape(vectors)[0]:
        while np.shape(poses)[0] <  np.shape(vectors)[0]:
            vectors.pop(0)
    return poses


