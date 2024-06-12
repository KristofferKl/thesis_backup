import rclpy
import math
from rclpy.node import Node
import sys
import numpy as np
import time 

from std_msgs.msg import String
from moveit_group_planner_interfaces.msg import Waypoints
from moveit_group_planner_interfaces.srv import Plan
from geometry_msgs.msg import Pose

import pandas as pd
from scipy.spatial.transform import Rotation as R

# from geometry_msgs.msg import Pose
# import numpy as np
# from scipy.spatial.transform import Rotation as R
 
# pose_goal = Pose()
# roll = 180 * np.pi / 180
# pitch = 0 * np.pi / 180
# yaw = -90 * np.pi / 180

# # Create a rotation object from Euler angles
# r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
# q = r.as_quat()  # Get quaternion [x, y, z, w]

# pose_goal.orientation.x = q[0]
# pose_goal.orientation.y = q[1]
# pose_goal.orientation.z = q[2]
# pose_goal.orientation.w = q[3]

# pose_goal.position.x = -0.45
# pose_goal.position.y = 0.7
# pose_goal.position.z = 1.4




DEBUG = True

# point_pose_input = [
# [1404.9008,318.1502,-738.1728,0.9362,-0.0793,-0.3424,0.0],
# [1403.6824,299.134,-738.0828,0.9362,-0.0793,-0.3424,0.0],
# [1401.6535,280.3287,-737.963,0.9362,-0.0793,-0.3424,0.0],
# [1399.5942,261.1582,-737.8514,0.9362,-0.0793,-0.3424,0.0],
# [1397.6347,241.182,-737.9442,0.9362,-0.0793,-0.3424,0.0],
# [1395.4201,220.8562,-737.8583,0.9362,-0.0793,-0.3424,0.0],
# [1393.64,200.3873,-737.9326,0.9362,-0.0793,-0.3424,0.0],
# [1391.5335,179.6207,-737.8311,0.9362,-0.0793,-0.3424,0.0],
# [1389.0904,158.5628,-737.5624,0.9362,-0.0793,-0.3424,0.0],
# [1387.0082,136.8783,-737.4965,0.9362,-0.0793,-0.3424,0.0],
# [1384.9629,114.1762,-737.5927,0.9362,-0.0793,-0.3424,0.0],
# [1382.571,91.1286,-737.4743,0.9362,-0.0793,-0.3424,0.0],
# [1378.2929,69.1974,-736.1708,0.9362,-0.0793,-0.3424,0.0],
# [1376.3571,45.284,-736.2238,0.9362,-0.0793,-0.3424,0.0],
# [1374.0671,20.975,-736.0766,0.9362,-0.0793,-0.3424,0.0],
# [1374.571,-6.096,-737.5502,0.9362,-0.0793,-0.3424,0.0],
# [1371.7321,-32.1,-737.2782,0.9362,-0.0793,-0.3424,0.0],
# [1369.0388,-58.8697,-737.0621,0.9362,-0.0793,-0.3424,0.0],
# [1366.5775,-86.0737,-736.8889,0.9362,-0.0793,-0.3424,0.0],
# [1361.3299,-110.8734,-735.0172,-0.712,0.4577,0.5325,0.0],
# [1362.0833,-110.8635,-735.2859,-0.712,0.4577,0.5325,0.0],
# [1362.6048,-110.7791,-735.4788,-0.712,0.4577,0.5325,0.0],
# [1361.3299,-110.8734,-735.0172,-0.712,0.4577,0.5325,0.0],
# [1360.7421,-111.0471,-735.3497,-0.712,0.4577,0.5325,0.0],
# [1360.0946,-111.0393,-735.1967,-0.712,0.4577,0.5325,0.0],
# [1359.5513,-111.2238,-735.4403,-0.712,0.4577,0.5325,0.0],
# [1356.5165,-111.244,-735.8598,-0.712,0.4577,0.5325,0.0],
# [1362.0833,-110.8635,-735.2859,-0.712,0.4577,0.5325,0.0],
# [1355.0003,-111.1007,-735.7568,-0.712,0.4577,0.5325,0.0],
# [1354.0915,-110.9874,-736.0071,-0.712,0.4577,0.5325,0.0],
# [1353.393,-111.0127,-735.6524,-0.712,0.4577,0.5325,0.0],
# [1325.597,-108.3266,-736.0625,-0.712,0.4577,0.5325,0.0],
# [1289.7116,-104.9207,-736.4541,-0.712,0.4577,0.5325,0.0],
# [1255.5129,-101.3094,-736.3566,-0.712,0.4577,0.5325,0.0],
# [1245.0658,-99.9359,-736.3118,-0.712,0.4577,0.5325,0.0],
# [1245.1693,-100.8325,-737.1428,-0.712,0.4577,0.5325,0.0],
# [1245.2521,-101.6776,-737.3147,-0.712,0.4577,0.5325,0.0],
# [1244.0632,-101.0525,-737.124,-0.712,0.4577,0.5325,0.0],
# [1229.2892,-95.756,-734.5739,-0.828,0.3363,0.4487,0.0],
# [1228.5174,-95.59,-734.6655,-0.828,0.3363,0.4487,0.0],
# [1228.1521,-95.7086,-734.5174,-0.828,0.3363,0.4487,0.0],
# [1227.3723,-95.5273,-734.6753,-0.828,0.3363,0.4487,0.0],
# [1214.6662,-94.5165,-734.8266,-0.828,0.3363,0.4487,0.0],
# [1185.1147,-91.5417,-734.9423,-0.828,0.3363,0.4487,0.0],
# [1169.6949,-92.9223,-737.2055,-0.828,0.3363,0.4487,0.0],
# [1169.2545,-93.0597,-736.9918,-0.828,0.3363,0.4487,0.0],
# [1168.5786,-92.9236,-737.0863,-0.828,0.3363,0.4487,0.0],
# [1157.8228,-91.9309,-737.0523,-0.828,0.3363,0.4487,0.0],
# [1131.3581,-89.6121,-737.6629,-0.828,0.3363,0.4487,0.0],
# [1106.4814,-87.7531,-738.0636,-0.828,0.3363,0.4487,0.0],
# [1082.3668,-85.8447,-738.5638,-0.828,0.3363,0.4487,0.0],
# [1077.0009,-86.9022,-739.4278,-0.828,0.3363,0.4487,0.0],
# [1076.5286,-86.9585,-739.6255,-0.828,0.3363,0.4487,0.0],
# [1071.6156,-86.136,-739.2309,-0.828,0.3363,0.4487,0.0],
# [1071.024,-85.9649,-739.2765,-0.828,0.3363,0.4487,0.0],
# [1059.2031,-82.9932,-738.1105,-0.828,0.3363,0.4487,0.0]


point_pose_input = np.array(pd.read_csv("/home/zivid/pytorch_env/OUTPUT.csv", sep = ',', header= None)) # read the data shown above from the file at the given location, allows for updating the path without rebuilding the workspace
print(f"{point_pose_input= }")

#position and pose for our coordinate frame relative to to world:
# position_baseframe = [-1.1072, 2.2256, 0.82]
position_baseframe = [-1.1072, 2.2256, 1.315]

orientation_baseframe = [0, 0, -0.26303, 0.96479]

world = [0,0,0,1]

robot2_capture_pos_1 = [-0.51684, 2.2927, 1.0029]
# robot2_capture_quat_1 = [0.13725, 0.32965, -0.39001, 0.84875]

robot2_capture_quat_1= [0.6972, -0.042663, -0.50888, 0.50311]
robot2_capture_point_pose_1 = [-0.51684, 2.2927, 1.0029, 0.6972, -0.042663, -0.50888, 0.50311]



end_effector_pos_pose = [-0.33861, 2.0932, 0.72185 ,0.85907, 0.26881, 0.06429, 0.43081]




#the translation from worldframe to workpiece center.
cx = 0.0
cy = 1.532
cz = 0.575

MAX_SPEED = 0.1 #zero is limitless
GROUP_NAME = "group_2"
JOB_NR     = 5

offset = 25.e-3/2 + 10.e-3 #half of the diameter of the end effector + 10mm # -approach 
offset_tig = (50.e-3)/2


def quaternion_mult(q,r):
    # Extract individual components of the quaternions
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = r
    
    # Perform quaternion multiplication
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return [x,y,z,w]
    
def quaternion_qunj(q):
    x, y, z, w = q
    
    # Compute the conjugate
    conjugate = [-x, -y, -z, w]
    
    return conjugate


# def point_rotation_by_quaternion(point,q):
#     #q = xyzw
#     r = point + [0] #adds w = 0 to xyz point for such that r represent a quaternion
#     q_conj = quaternion_qunj(q)
#     return quaternion_mult(quaternion_mult(q,r),q_conj)[:3]#returns xyz rotated

def point_rotation_by_quaternion(point,q): #KK edit
    """
    Takes either a point or a quaternion (length 3 or 4)
    Returns the rotated element (of length 3 or 4 respectively)
    """
    #q = xyzw
    if len(point)==3: # checks if the input is represented as a quaternion already
        r = point + [0] #adds w = 0 to xyz point for such that r represent a quaternion
        q_conj = quaternion_qunj(q)
        return quaternion_mult(quaternion_mult(q,r),q_conj)[:3]#returns xyz rotated
    else:#the input is a quaternion
        r = point
        q_conj = quaternion_qunj(q)
        return quaternion_mult(quaternion_mult(q,r),q_conj)#returns xyzw rotated
    
#KK
def create_image_pose(current_orientation_euler):

    # Example current orientation in Euler angles (roll, pitch, yaw) in degrees

    # Convert current orientation to rotation matrix
    current_rotation = R.from_euler('xyz', current_orientation_euler, degrees=True)
    current_rotation_matrix = current_rotation.as_matrix()

    # Define the "up" direction (z-axis pointing up in Cartesian space)
    up_direction = np.array([0, 0, 1])

    # Align the z-axis of the current rotation to the "up" direction
    # This is done by ensuring the third column of the rotation matrix is the up direction
    # while keeping the orthogonality and right-handedness of the matrix
    z_axis = current_rotation_matrix[:, 2]

    # Compute the rotation needed to align z_axis to up_direction
    # This can be done using the cross product and angle between the vectors
    axis = np.cross(z_axis, up_direction)
    angle = np.arccos(np.dot(z_axis, up_direction) / (np.linalg.norm(z_axis) * np.linalg.norm(up_direction)))

    # Rotation axis needs to be normalized
    axis = axis / np.linalg.norm(axis)

    # Create the rotation matrix to align z-axis to the up direction
    alignment_rotation = R.from_rotvec(angle * axis)
    alignment_rotation_matrix = alignment_rotation.as_matrix()

    # Combine the rotations
    new_rotation_matrix = alignment_rotation_matrix @ current_rotation_matrix

    # Convert the new rotation matrix to a quaternion
    new_rotation = R.from_matrix(new_rotation_matrix)
    new_orientation_quaternion = new_rotation.as_quat()
    return new_orientation_quaternion



#KK
def input_vec_to_rotation_quat(vec):
    x_vec = np.array([0,0,1]) #original vector that is transformed as given by q=[0,0,0,1] , w = 1, imaginary is 0
    # assert vec[3]==0, "Error, the assumed 'pure vector' contains a real part"
    if len(vec) > 3:
        vec = np.array(vec[:3]) #this is now of length 3, removed  w=0
    axis= np.cross(x_vec, vec)
    axis = axis/np.linalg.norm(axis)
    angle_rad= 2*np.arctan(np.linalg.norm(np.linalg.norm(vec)*x_vec - np.linalg.norm(x_vec)*vec)/
                       np.linalg.norm(np.linalg.norm(vec)*x_vec + np.linalg.norm(x_vec)*vec))
    axis *= angle_rad
    r = R.from_rotvec(axis)
    q = r.as_quat() #get the rotation quaternion
    # r = R.from_rotvec(vec, angle_rad)
    # q = r.as_quat() #get the rotation quaternion
    return q

def rotate_by_R_from_quat(vec, quat):
    r=R.from_quat(quat)
    rot = r.as_matrix()
    if len(vec) >3: # in case it has a 0 added as quat-padding
        vec = vec[:3]

    return np.array(rot) @ np.array(vec) 

def transform_input_by_quaternionPose_and_pos(point_pose_in:list[list], rotation_quaternion, position_offset): #this currently has ofset_local that moves the job up in the y-position
    """ Takes a list representing the position (index 0,1,2) and pose (index 3,4,5,6), a rotation_quaternion and a position_offset, 
        Transforms the input point_pose to a new frame.
        NOTE: the position is represented in meters!! 

        Output: The transformed point_pose list in the same format as the input list
    """
    # offset_local = 0.52 #this is aproximately the height of the lower part of the workpiece


    # offset_local = 0.0 # 0.70 is good for testing
    offset_local = 0.015
    # offset_local = 0.02

    # def input_rot_to_pose(vec):

    point, pose = [], []
    point_pose_out = []
    for pp in point_pose_in:
        point = [pp[0], pp[1], pp[2]]
        pose = [pp[3], pp[4], pp[5], pp[6]]
        
        #rotate the point by the quaternion
        x0,y0,z0 = point_rotation_by_quaternion(point, rotation_quaternion) # note this outputs a quaternion representation
        x1,y1,z1 = x0+position_offset[0], y0+ position_offset[1], z0+position_offset[2]

        
        # pose = point_rotation_by_quaternion(pose, rotation_quaternion)
        pose = rotate_by_R_from_quat(pose, rotation_quaternion)
        # pose = point_rotation_by_quaternion([0,0,0,1], pose)
        pose = input_vec_to_rotation_quat(pose)
        # rot_quat = set_pose_orientation(pose, np.pi/8)
        # print(f"{rot_quat = }")
        # pose = point_rotation_by_quaternion(pose, rot_quat)
        
        # pose = quaternion_mult(rot_quat, pose)
        # pose = quaternion_mult(pose, rotation_quaternion)

                # quat= [1,0,0,0]
        # quat /= np.linalg.norm(quat)
        # pose = point_rotation_by_quaternion(pose, quat) #THIS IS ADDED AS AN EXPERIMENT, NOT THE OG IMPLEMENTATION!!!! 

        # pose = point_rotation_by_quaternion(pose, np.array(pose)/4) #THIS IS ADDED AS AN EXPERIMENT, NOT THE OG IMPLEMENTATION!!!! 

        i,j,k,w = pose/np.abs(np.linalg.norm(pose))#alltid normaliser etter bruk
        offset_extra= 0.01
        wirefeeder = 0.010
        point_pose_out.append([x1-offset_tig+offset_extra+wirefeeder,
                               y1 + offset_tig/3, 
                               z1+offset_local+ offset_tig-offset_extra-0.002, 
                               i,j,k,w]) ### offset_tig added for testing
    assert np.shape(point_pose_in) == np.shape(point_pose_out), f"Error, the resulting point_pose with shape: {np.shape(point_pose_out)}, does not match the input point_pose with shape: {np.shape(point_pose_in)}"
    return point_pose_out


def set_pose_orientation(pose ,angle_radians):
    i, j, k, w = pose
    half_angle_rad = np.arccos(w) # this is the half angle, but no need to times and divide it by 2

    # i /= np.sin(half_angle_rad)
    # j /= np.sin(half_angle_rad)
    # k /= np.sin(half_angle_rad)

    s = np.sin(angle_radians/2)
    c = np.cos(angle_radians/2)
    new_pose = [i*s, j*s, k*s, w*c]

    return new_pose/np.linalg.norm(new_pose)


def point_pose_scale_from_mm_to_m(point_pose_list:list[list]):
    max = 0
    #adding debugging:
    for el in point_pose_list:
        max = np.max([max, np.max(el)])
    if max <= 10:
        print(f"The input in 'point_pose_scale_from_mm_to_m' seems to be in meters already, double check the input!!")        
    for i in range(np.shape(point_pose_list)[0]): #converting the x,y,z positions from mm to m, the psoe is left unaltered
        point_pose_list[i][0] *= 1e-3
        point_pose_list[i][1] *= 1e-3
        point_pose_list[i][2] *= 1e-3
    return point_pose_list



def create_pose(x, y, z, r = None , p = None , q = None, w= None):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    if (r != None):
        pose.orientation.x = float(r)
    if (p != None):
        pose.orientation.y = float(p)
    if(q != None):
        pose.orientation.z = float(q)
    if(w != None):
        pose.orientation.w = float(w)
    return pose


def circle(r, x, y, z, n_points = 100):
    points = []
    for i in range(n_points+1):  #resolution, + 1 for full circle
        
        #step = 2*3.1415/n_points
        
        xx = x + r*math.cos(i*2*3.1415/n_points)
        yy = y + r*math.sin(i*2*3.1415/n_points)
        points.append(create_pose(xx, yy, z ,1.0, 0.0, 0.0, 0.0)) #xyz + orientering
    return points


def normalize_pose(pose:Pose):
    i = pose.orientation.x
    j = pose.orientation.y
    k = pose.orientation.z
    w = pose.orientation.w
    ijkw = np.array([i,j,k,w])
    ijkw /= np.linalg.norm(ijkw)
    i,j,k,w = ijkw
    pose.orientation.x = i
    pose.orientation.y = j
    pose.orientation.z = k
    pose.orientation.w = w



def zivid_job(point_pose_list, rotation_quaternion, position_offset):
    result = []

    # rot_quat = point_rotation_by_quaternion(orientation_baseframe, world) #this one is stil in testing

    point_pose_list= point_pose_scale_from_mm_to_m(point_pose_list)
    point_pose = transform_input_by_quaternionPose_and_pos(point_pose_list, rotation_quaternion, position_offset)

    # sup_pose = np.array([1,-1,-1,1])
    # sup_pose =sup_pose/np.abs(np.linalg.norm(sup_pose))
    # sup_pose /= 0.5
    for pp in point_pose:
        # print(pp)

        result.append(create_pose(pp[0], pp[1], pp[2],
                                    pp[3], pp[4], pp[5] ,pp[6])) 
                                  
                                #   sup_pose[0], sup_pose[1], sup_pose[2], sup_pose[3]))
        # 0,0,0,0))
        #1,0,0,0)) this orientation works
        # 0,1,0,0))
        # 0,0,1,0)) #straight down
        # 0,0,0,1)) # straight down


    # end_pose = point_rotation_by_quaternion(robot2_capture_point_pose_1[3:], orientation_baseframe)

    
    # xi, yi, zi, ii, ji, ki, wi = robot2_capture_point_pose_2
    # # placeholder_pose = [ii, ji, ki, wi] 
    # placeholder_pose = end_pose
    # placeholder_pose /= np.linalg.norm(np.array(placeholder_pose))
    # ii, ji, ki, wi = placeholder_pose
    # ii, ji, ki, wi = set_pose_orientation([ii,ji,ki,wi], 0)
    # xi, yi, zi = robot2_capture_pos_1
    # cam_pose = create_image_pose([30, 45, 60])
    # ii, ji, ki, wi = cam_pose

##############################################################################
    ########### this kinda works, fix orientation?
    ############### adding positions after the main job
    # xi, yi, zi = robot2_capture_pos_1
    xi, yi, zi = end_effector_pos_pose[:3]

    # xi += +0.20
    # yi += -0.20 
    # zi += -0.25
    #first pos: creates a middle position between the end point and the image capture point to go to before starting to re-orient the end effector
    first_pose = Pose()
    first_pose.position.x= (result[-1].position.x + xi)/2
    first_pose.position.y=(result[-1].position.y + yi)/2
    first_pose.position.z= (result[-1].position.z + zi)/2
    first_pose.orientation.x = result[-1].orientation.x
    first_pose.orientation.y = result[-1].orientation.y
    first_pose.orientation.z = result[-1].orientation.z
    first_pose.orientation.w = result[-1].orientation.w
    normalize_pose(first_pose)
    result.append(first_pose)

    ######## second constant: orienting back to initial welding orientation

    second_pose = Pose()
    original_weld_pose = result[0]
    second_pose.position.x = result[-1].position.x
    second_pose.position.y = result[-1].position.y
    second_pose.position.z = result[-1].position.z
    second_pose.orientation.x = original_weld_pose.orientation.x
    second_pose.orientation.y = original_weld_pose.orientation.y
    second_pose.orientation.z = original_weld_pose.orientation.z
    second_pose.orientation.w = original_weld_pose.orientation.w
    normalize_pose(second_pose)
    result.append(second_pose)



    ###### last pose, the orientation to capture the image from
    # xi, yi, zi = robot2_capture_pos_1
    # third_pose = Pose()
    # third_pose.position.x= xi 
    # third_pose.position.y= yi
    # third_pose.position.z= zi
    # third_pose.orientation.x = original_weld_pose.orientation.x
    # third_pose.orientation.y = original_weld_pose.orientation.y
    # third_pose.orientation.z = original_weld_pose.orientation.z
    # third_pose.orientation.w = original_weld_pose.orientation.w
    # normalize_pose(third_pose)
    # result.append(third_pose)

    

    xi, yi, zi,  = end_effector_pos_pose[:3]
    ii, ji, ki, wi = end_effector_pos_pose[3:] / np.linalg.norm(end_effector_pos_pose[3:])

    # input_vec = np.array([0.3, -0.3, -0.7])
    # input_vec /= np.linalg.norm(input_vec)

    # quat = input_vec_to_rotation_quat(input_vec)
    # quat = set_pose_orientation(quat, np.pi * (1/2 + 1/4)) #sets the orientation in space, not around its own axis
    # ii, ji, ki, wi = quat

    fourth_pose = create_pose(xi, yi, zi, ii, ji, ki, wi)
    normalize_pose(fourth_pose)
    result.append(fourth_pose)
###############################################################################################

    # result.append(create_pose(xi+0.20, yi-0.20, zi-0.25, ii, ji, ki, wi)) 

    return result


def getWaypoints(job_nr):
    msg = Waypoints()

        
    if job_nr == 0:
        diff = point_rotation_by_quaternion([0,0,  -offset], [-0.8535533905932737, 0.3535533905932736, -0.14644660940672624, 0.353553390593274]) # convert z in body frame to world

        #approach = point_rotation_by_quaternion([0,0,1], [-0.8535533905932737, 0.3535533905932736, -0.14644660940672624, 0.353553390593274]) #the end effector apporach dir

        msg.waypoints.append(create_pose(cx+0.5, cy, 1.4, -0.5720614, 0.5720614, 0, 0.5877852522924731))
        msg.waypoints.append(create_pose(cx+0.5, cy, 1.4, -0.8535533905932737, 0.3535533905932736, -0.14644660940672624, 0.353553390593274))
        msg.waypoints.append(create_pose(cx+0.5 + diff[0], cy + diff[1], 1.4 + diff[2], -0.8535533905932737, 0.3535533905932736, -0.14644660940672624, 0.353553390593274))
        
    if job_nr == 1:
        diff = point_rotation_by_quaternion([0,0,  -offset], [0.8535533905932738, -0.35355339059327373, -0.14644660940672619, 0.3535533905932738]) # convert z in body frame to world
        approach = point_rotation_by_quaternion([0,0,1], [0.8535533905932738, -0.35355339059327373, -0.14644660940672619, 0.3535533905932738])
        #gruppe_1 sveise langs kant
        msg.waypoints.append(create_pose(cx+0.5, cy, 1.4, -0.5720614, 0.5720614, 0, 0.5877852522924731)) #<- needed if robot is at home +-w if error
        msg.is_job.append(False) #død-bevegelse
        msg.waypoints.append(create_pose(cx + 0.065 + diff[0],
                                         cy + 0.065 + diff[1],
                                              0.73  + diff[2],
                                          1,0,0,0))#denne orienteringen er ca straight down group_1
        msg.is_job.append(False) #bevege ee til ca start
        
        msg.waypoints.append(create_pose(cx + 0.065 + diff[0],
                                         cy + 0.065 + diff[1],
                                               0.73 + diff[2],
                                         0.8535533905932738, -0.35355339059327373, -0.14644660940672619, 0.3535533905932738)) #q
        msg.is_job.append(False)
        
        #weld job
        msg.waypoints.append(create_pose(cx + 0.065 + diff[0],
                                         cy + 0.065 + diff[1],
                                         cz + 0.01  + diff[2],
                                         0.8535533905932738, -0.35355339059327373, -0.14644660940672619, 0.3535533905932738)) #q
        msg.is_job.append(True)  #begynne sveis langs kortside av profil
        msg.waypoints.append(create_pose(cx + 0.065   + diff[0], 
                                         cy +  0.400  + diff[1],
                                         cz + 0.01    + diff[2],
                                         0.8535533905932738, -0.35355339059327373, -0.14644660940672619, 0.3535533905932738)) #q
        msg.is_job.append(False) #stopp sveis
        msg.waypoints.append(create_pose(cx + 0.065   + diff[0], 
                                         cy +  0.400  + diff[1],
                                         cz + 0.13    + diff[2],
                                         0.8535533905932738, -0.35355339059327373, -0.14644660940672619, 0.3535533905932738)) #q
        msg.is_job.append(False) #løfte ee
        
    elif job_nr == 2:
        diff = point_rotation_by_quaternion([0,0,  -offset], [-0.8535533905932737, 0.3535533905932736, -0.14644660940672624, 0.353553390593274]) # convert z in body frame to
        #gruppe_1 sveise langs kant
        msg.waypoints.append(create_pose(cx+0.5, cy, 1.4, -0.5720614, 0.5720614, 0, 0.5877852522924731)) #<- needed if robot is at home +-w if error
        msg.is_job.append(False) #død-bevegelse
        msg.waypoints.append(create_pose(cx-0.065, cy - 0.4 , 0.73, 1,0,0,0))#denne orienteringen er ca rett ned for gruppe 1
        msg.is_job.append(False) #bevege ee til ca start
        
        #weld job
        msg.waypoints.append(create_pose(cx - 0.065 + diff[0],
                                         cy - 0.4   + diff[1],
                                         cz + 0.13  + diff[2],
                                         -0.8535533905932737, 0.3535533905932736, -0.14644660940672624, 0.353553390593274)) #q
        msg.is_job.append(True)  #begynne sveis langs kortside av profil
        
        msg.waypoints.append(create_pose(cx - 0.065 + diff[0],
                                         cy - 0.4   + diff[1],
                                         cz + 0.01  + diff[2],
                                         -0.8535533905932737, 0.3535533905932736, -0.14644660940672624, 0.353553390593274)) #q
        msg.is_job.append(False)
        msg.waypoints.append(create_pose(cx - 0.065 + diff[0], 
                                         cy -  0.065 + diff[1],
                                         cz + 0.01  + diff[2],
                                         -0.8535533905932737, 0.3535533905932736, -0.14644660940672624, 0.353553390593274)) #q
        msg.is_job.append(False) #stopp sveis
        msg.waypoints.append(create_pose(cx - 0.065 + diff[0], 
                                         cy -  0.065 + diff[1],
                                         cz +  0.13 + diff[2],
                                         -0.8535533905932737, 0.3535533905932736, -0.14644660940672624, 0.353553390593274)) #q
        msg.is_job.append(False) #løfte ee
        
    elif job_nr == 3:
        #gruppe_1 sveise invendig
        msg.is_job.append(False) #movement
        msg.waypoints.append(create_pose(cx+0.5, cy, 1.4 + 0.5, -0.5720614, 0.5720614, 0, -0.5877852522924731)) #<needed if robot at home +-w if error
        
        msg.waypoints.append(create_pose(cx+0.6, cy , 0.75, 0.856925, -0.514625, 0.0216612, -0.0192533))#denne orienteringen er ca rett ned for gruppe 1
        msg.is_job.append(False)#to position
        msg.waypoints.append(create_pose(cx + 0.50+0.05 , cy, 0.51 + 0.04 + 0.01 + 0.05, 0, -0.7071067811865476, 0.0, 0.7071067811865476))
        msg.is_job.append(False)#to position
        msg.waypoints.append(create_pose(cx + 0.50 , cy, 0.51 + 0.04 + 0.01 + 0.05, 0, -0.7071067811865476, 0.0, 0.7071067811865476))
        msg.is_job.append(True)
        msg.waypoints.append(create_pose(cx + 0.50 - 0.15 , cy, 0.51 + 0.04 + 0.01 + 0.05, 0, -0.7071067811865476, 0.0, 0.7071067811865476))
        msg.is_job.append(False) #"weld" along edge
        msg.waypoints.append(create_pose(cx + 0.50+0.05 , cy, 0.51 + 0.04 + 0.01 + 0.05, 0, -0.7071067811865476, 0.0, 0.7071067811865476))
        msg.is_job.append(False)#move ee out
        msg.waypoints.append(create_pose(cx+0.6, cy , 0.73, 0.856925, -0.514625, 0.0216612, -0.0192533))#denne orienteringen er ca rett ned for gruppe 1
        msg.is_job.append(False) #move up

    elif job_nr == 4:
        msg.waypoints.append(create_pose(cx, cy, 0.76, 1, 0, 0, 0))
        msg.waypoints.append(create_pose(cx+0.5, cy, 0.76, 1, 0, 0, 0))
        msg.waypoints.append(create_pose(cx-0.5, cy, 0.76, 1, 0, 0, 0))
        

    elif job_nr == 5:
        print("starting job 5")
        msg.waypoints = zivid_job(point_pose_input,orientation_baseframe, position_baseframe)
        # msg.waypoints.insert(0,create_pose(cx+0.5, cy, 1.4, -0.5720614, 0.5720614, 0, 0.5877852522924731)) #<- needed if robot is at home +-w if error
    if (len(msg.is_job) != len(msg.waypoints)):
            for w in msg.waypoints:
                print(w)
                msg.is_job.append(True)
                
    return msg.waypoints, msg.is_job

    # elif job_nr == 5:
    #     msg.waypoints = circle(0.5, cx, cy, 0.73, 100)
    # if (len(msg.is_job) != len(msg.waypoints)):
    #         for w in msg.waypoints:
    #             msg.is_job.append(True)
                
    # return msg.waypoints, msg.is_job

class MinimalPublisher(Node):
    #dette burde vært service
    def __init__(self):
        super().__init__('minimal_publisher')
        self.client_    = self.create_client(Plan, "plan_group")
        
        
        
    def call_service(self):
        #build the request
        req = Plan.Request()
        req.waypoints.groupname = GROUP_NAME
        req.waypoints.speed      = float(MAX_SPEED) 
        print("creating plan...")
        req.waypoints.waypoints, req.waypoints.is_job = getWaypoints(JOB_NR)
        print("calling service...")

        #request request
        future = self.client_.call_async(req)

        # print(self.get_logger().info(str(future.result().trajectory_fraction)))

        rclpy.spin_until_future_complete(self, future, timeout_sec=1.3)
        # rclpy.spin_once(self, timeout_sec = 2)

        self.get_logger().info(str(future.result().trajectory_fraction))

#### attempt at making it work with just 1 call
        # req = Plan.Request()
        # req.waypoints.groupname = GROUP_NAME
        # req.waypoints.speed      = float(MAX_SPEED) 
        # print("creating plan...")
        # req.waypoints.waypoints, req.waypoints.is_job = getWaypoints(JOB_NR)
        # future = self.client_.call_async(req)
        # print(type(self.get_logger().info(str(future.result()))))

        # while type(self.get_logger().info(str(future.result()))) is not None:
        #     req = Plan.Request()
        #     req.waypoints.groupname = GROUP_NAME
        #     req.waypoints.speed      = float(MAX_SPEED) 
        #     print("creating plan...")
        #     req.waypoints.waypoints, req.waypoints.is_job = getWaypoints(JOB_NR)
        #     future = self.client_.call_async(req)
        #     print("future done")



        #     rclpy.spin_until_future_complete(self, future, timeout_sec=1)
        #     print("rcply done")
        #     print("logger done")
        #     # time.sleep(1)
        #     try:
        #         self.get_logger().info(str(future.result().trajectory_fraction))
        #         break
        #     except:
        #         continue
        
        
        
        
 
    	


def main(args=None):
    args=sys.argv
    try:
        JOB_NR = int(args)
    except Exception:
        pass #could not convert args to int
    
    rclpy.init()

    minimal_publisher = MinimalPublisher()
    minimal_publisher.call_service()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
