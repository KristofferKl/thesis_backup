import rclpy
import math
from rclpy.node import Node
import sys
import numpy as np

from std_msgs.msg import String
from moveit_group_planner_interfaces.msg import Waypoints
from moveit_group_planner_interfaces.srv import Plan
from geometry_msgs.msg import Pose

# import pandas as pd

DEBUG = True

point_pose_input = [
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

[1405.0378,315.728,-738.0498,0.8559,0.4707,-0.2139,0.0],
[1401.3544,273.4491,-737.9873,0.8559,0.4707,-0.2139,0.0],
[1397.7296,232.9701,-737.944,0.8559,0.4707,-0.2139,0.0],
[1392.6167,186.8011,-737.9045,0.8559,0.4707,-0.2139,0.0],
[1388.3708,143.3329,-737.3956,0.8559,0.4707,-0.2139,0.0],
[1383.5903,93.0395,-737.4446,0.8559,0.4707,-0.2139,0.0],
[1376.8381,46.0201,-735.7811,0.8559,0.4707,-0.2139,0.0],
[1375.3088,-0.4538,-737.1496,0.8559,0.4707,-0.2139,0.0],
[1374.605,-11.8161,-737.374,0.8559,0.4707,-0.2139,0.0],
[1369.9189,-65.5627,-736.8203,0.8559,0.4707,-0.2139,0.0],
[1362.4392,-110.5003,-734.9353,-0.3997,-0.8291,0.391,0.0],
[1363.504,-110.9007,-735.3253,-0.3997,-0.8291,0.391,0.0],
[1364.6336,-111.259,-735.9184,-0.3997,-0.8291,0.391,0.0],
[1362.4392,-110.5003,-734.9353,-0.3997,-0.8291,0.391,0.0],
[1362.0833,-110.8635,-735.2859,-0.3997,-0.8291,0.391,0.0],
[1361.3299,-110.8734,-735.0172,-0.3997,-0.8291,0.391,0.0],
[1358.0401,-111.2443,-735.9068,-0.3997,-0.8291,0.391,0.0],
[1357.358,-111.3192,-735.6777,-0.3997,-0.8291,0.391,0.0],
[1356.5165,-111.244,-735.8598,-0.3997,-0.8291,0.391,0.0],
[1355.817,-111.1815,-735.6726,-0.3997,-0.8291,0.391,0.0],
[1355.0003,-111.1007,-735.7568,-0.3997,-0.8291,0.391,0.0],
[1354.0915,-110.9874,-736.0071,-0.3997,-0.8291,0.391,0.0],
[1346.8503,-110.3777,-736.102,-0.3997,-0.8291,0.391,0.0],
[1269.7523,-103.0134,-736.5529,-0.3997,-0.8291,0.391,0.0],
[1245.0658,-99.9359,-736.3118,-0.3997,-0.8291,0.391,0.0],
[1245.1693,-100.8325,-737.1428,-0.3997,-0.8291,0.391,0.0],
[1245.2521,-101.6776,-737.3147,-0.3997,-0.8291,0.391,0.0],
[1244.0632,-101.0525,-737.124,-0.3997,-0.8291,0.391,0.0],
[1229.2892,-95.756,-734.5739,-0.0992,-0.9358,0.3382,0.0],
[1228.5174,-95.59,-734.6655,-0.0992,-0.9358,0.3382,0.0],
[1228.1521,-95.7086,-734.5174,-0.0992,-0.9358,0.3382,0.0],
[1227.3723,-95.5273,-734.6753,-0.0992,-0.9358,0.3382,0.0],
[1192.2369,-91.5129,-734.5063,-0.0992,-0.9358,0.3382,0.0],
[1169.6949,-92.9223,-737.2055,-0.0992,-0.9358,0.3382,0.0],
[1169.2545,-93.0597,-736.9918,-0.0992,-0.9358,0.3382,0.0],
[1168.5786,-92.9236,-737.0863,-0.0992,-0.9358,0.3382,0.0],
[1134.5361,-89.8123,-737.5346,-0.0992,-0.9358,0.3382,0.0],
[1081.8914,-85.8242,-738.2834,-0.0992,-0.9358,0.3382,0.0],
[1077.0009,-86.9022,-739.4278,-0.0992,-0.9358,0.3382,0.0],
[1076.8688,-87.1906,-739.4677,-0.0992,-0.9358,0.3382,0.0],
[1075.5525,-86.6157,-739.4186,-0.0992,-0.9358,0.3382,0.0],
[1071.8902,-85.7545,-738.8222,-0.0992,-0.9358,0.3382,0.0],
[1071.6156,-86.136,-739.2309,-0.0992,-0.9358,0.3382,0.0],
[1071.024,-85.9649,-739.2765,-0.0992,-0.9358,0.3382,0.0],
[1036.8904,-81.076,-738.196,-0.0992,-0.9358,0.3382,0.0]



]

# point_pose_input = [
#     [1400.9666,319.4054,-738.0789,0.7715,0.6319,-0.074,0.0],
#     [1399.7158,311.4532,-737.5739,0.7715,0.6319,-0.074,0.0],
#     [1399.7641,311.7238,-737.554,0.7715,0.6319,-0.074,0.0],
#     [1400.0034,312.1432,-737.5116,0.7715,0.6319,-0.074,0.0],
#     [1400.1974,309.7475,-737.8195,0.7715,0.6319,-0.074,0.0],
#     [1400.3631,310.113,-737.828,0.7715,0.6319,-0.074,0.0],
#     [1397.6884,298.5276,-737.176,0.7715,0.6319,-0.074,0.0],
#     [1395.9406,280.9348,-736.9315,0.7715,0.6319,-0.074,0.0],
#     [1394.9714,262.7614,-737.0605,0.7715,0.6319,-0.074,0.0],
#     [1392.7133,240.3959,-737.3175,0.7715,0.6319,-0.074,0.0],
#     [1391.718,221.5022,-737.408,0.7715,0.6319,-0.074,0.0],
#     [1390.6034,202.0843,-737.5881,0.7715,0.6319,-0.074,0.0],
#     [1387.7648,185.3663,-737.0629,0.7715,0.6319,-0.074,0.0],
#     [1387.1352,178.9044,-737.0162,0.7715,0.6319,-0.074,0.0],
#     [1385.8307,158.7692,-737.0407,0.7715,0.6319,-0.074,0.0],
#     [1385.3799,137.7383,-737.5205,0.7715,0.6319,-0.074,0.0],
#     [1383.1266,112.7432,-737.6485,0.7715,0.6319,-0.074,0.0],
#     [1381.4241,90.7805,-737.5726,0.7715,0.6319,-0.074,0.0],
#     [1377.9357,69.8757,-736.4326,0.7715,0.6319,-0.074,0.0],
#     [1376.2506,47.4497,-736.1679,0.7715,0.6319,-0.074,0.0],
#     [1373.6174,19.605,-736.2304,0.7715,0.6319,-0.074,0.0],
#     [1374.6511,-1.1234,-737.4572,0.7715,0.6319,-0.074,0.0],
#     [1374.7418,-0.5792,-737.3583,0.7715,0.6319,-0.074,0.0],
#     [1374.571,-6.096,-737.5502,0.7715,0.6319,-0.074,0.0],
#     [1372.4525,-30.3105,-737.1624,0.7715,0.6319,-0.074,0.0],
#     [1369.334,-60.3552,-737.0033,0.7715,0.6319,-0.074,0.0],
#     [1367.5298,-86.2692,-736.8815,0.7715,0.6319,-0.074,0.0],
#     [1362.4392,-110.5003,-734.9353,0.7715,0.6319,-0.074,0.0],
#     [1363.504,-110.9007,-735.3253,0.7715,0.6319,-0.074,0.0]
#     ]

# point_pose_input= [
#     [1400.0415,319.5687,-738.0591,-0.6003,-0.7576,0.2563,0.0],
#     [1398.9049,311.6745,-737.5245,-0.6003,-0.7576,0.2563,0.0],
#     [1399.0155,312.0501,-737.5054,-0.6003,-0.7576,0.2563,0.0],
#     [1398.6459,309.7086,-737.5963,-0.6003,-0.7576,0.2563,0.0],
#     [1398.9944,310.1034,-737.6137,-0.6003,-0.7576,0.2563,0.0],
#     [1399.1081,310.3606,-737.631,-0.6003,-0.7576,0.2563,0.0],
#     [1394.2756,284.5873,-736.4884,-0.6003,-0.7576,0.2563,0.0],
#     [1388.8067,200.1874,-737.2987,-0.6003,-0.7576,0.2563,0.0],
#     [1388.4133,192.8276,-737.1689,-0.6003,-0.7576,0.2563,0.0],
#     [1374.6511,-1.1234,-737.4572,-0.6003,-0.7576,0.2563,0.0],
#     [1374.7418,-0.5792,-737.3583,-0.6003,-0.7576,0.2563,0.0],
#     [1362.4392,-110.5003,-734.9353,-0.4384,0.8715,0.2198,0.0],
#     [1363.504,-110.9007,-735.3253,-0.4384,0.8715,0.2198,0.0]
#     ]

# point_pose_input= [
# [1405.0777,316.9909,-738.0793,0.9912,0.062,0.117,0.0],
# [1401.8865,277.3952,-738.0327,0.9912,0.062,0.117,0.0],
# [1397.9135,236.6888,-737.9797,0.9912,0.062,0.117,0.0],
# [1394.1077,194.346,-737.9185,0.9912,0.062,0.117,0.0],
# [1389.5337,150.277,-737.6106,0.9912,0.062,0.117,0.0],
# [1385.0503,103.7319,-737.3316,0.9912,0.062,0.117,0.0],
# [1377.0038,57.5251,-735.066,0.9912,0.062,0.117,0.0],
# [1375.8083,-1.3125,-736.9458,0.9912,0.062,0.117,0.0],
# [1374.2311,4.8449,-735.9122,0.9912,0.062,0.117,0.0],
# [1371.5718,-50.731,-736.869,0.9912,0.062,0.117,0.0],
# [1366.8776,-107.5499,-736.395,0.9912,0.062,0.117,0.0]
# ]

# point_pose_input= [
# [1405.0378,315.728,-738.0498,-0.1061,-0.751,-0.6517,0.0],
# [1402.0708,276.0939,-738.0362,-0.1061,-0.751,-0.6517,0.0],
# [1398.2102,235.3553,-737.9431,-0.1061,-0.751,-0.6517,0.0],
# [1394.2984,192.9062,-737.9153,-0.1061,-0.751,-0.6517,0.0],
# [1389.8193,148.771,-737.5715,-0.1061,-0.751,-0.6517,0.0],
# [1385.1146,102.1421,-737.3531,-0.1061,-0.751,-0.6517,0.0],
# [1377.124,55.8787,-735.013,-0.1061,-0.751,-0.6517,0.0],
# [1375.8083,-1.3125,-736.9458,-0.1061,-0.751,-0.6517,0.0],
# [1375.8226,-0.7968,-736.8616,-0.1061,-0.751,-0.6517,0.0],
# [1374.6849,2.8546,-736.0709,-0.1061,-0.751,-0.6517,0.0],
# [1371.6363,-52.5116,-736.6778,-0.1061,-0.751,-0.6517,0.0],
# [1367.1293,-109.7055,-736.5306,-0.1061,-0.751,-0.6517,0.0]
# ]

# point_pose_input= [
# [1404.5365,318.7443,-738.1968,-0.9137,0.2611,0.3115,0.0],
# [1375.7213,-1.8602,-737.0446,-0.9137,0.2611,0.3115,0.0],
# [1375.8083,-1.3125,-736.9458,-0.9137,0.2611,0.3115,0.0],
# [1375.8226,-0.7968,-736.8616,-0.9137,0.2611,0.3115,0.0]]





#position and pose for our coordinate frame relative to to world:
position_baseframe = [-1.1072, 2.2256, 0.82]
orientation_baseframe = [0, 0, -0.26303, 0.96479]

#the translation from worldframe to workpiece center.
cx = 0.0
cy = 1.532
cz = 0.575

MAX_SPEED = 0.1 #zero is limitless
GROUP_NAME = "group_2"
JOB_NR     = 5

offset = 25.e-3/2 + 10.e-3 #half of the diameter of the end effector + 10mm # -approach 


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

def transform_input_by_quaternionPose_and_pos(point_pose_in:list[list], rotation_quaternion, position_offset): #this currently has ofset_local that moves the job up in the y-position
    """ Takes a list representing the position (index 0,1,2) and pose (index 3,4,5,6), a rotation_quaternion and a position_offset, 
        Transforms the input point_pose to a new frame.
        NOTE: the position is represented in meters!! 

        Output: The transformed point_pose list in the same format as the input list
    """
    # offset_local = 0.52 #this is aproximately the height of the lower part of the workpiece
    offset_local = 0.70


    point, pose = [], []
    point_pose_out = []
    for pp in point_pose_in:
        point = [pp[0], pp[1], pp[2]]
        pose = [pp[3], pp[4], pp[5], pp[6]]

        #rotate the point by the quaternion
        x0,y0,z0 = point_rotation_by_quaternion(point, rotation_quaternion) # note this outputs a quaternion representation
        x1,y1,z1 = x0+position_offset[0], y0+ position_offset[1], z0+position_offset[2]
        pose = point_rotation_by_quaternion(pose, [1,0,0,0]) #THIS IS ADDED AS AN EXPERIMENT, NOT THE OG IMPLEMENTATION!!!! 
        pose = point_rotation_by_quaternion(pose, rotation_quaternion)
        i,j,k,w = pose/np.abs(np.linalg.norm(pose))#alltid normaliser etter bruk ;)
        point_pose_out.append([x1,y1,z1+offset_local, i,j,k,w])
    assert np.shape(point_pose_in) == np.shape(point_pose_out), f"Error, the resulting point_pose with shape: {np.shape(point_pose_out)}, does not match the input point_pose with shape: {np.shape(point_pose_in)}"
    return point_pose_out


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


def zivid_job(point_pose_list, rotation_quaternion, position_offset):
    result = []
    point_pose_list= point_pose_scale_from_mm_to_m(point_pose_list)
    point_pose = transform_input_by_quaternionPose_and_pos(point_pose_list, rotation_quaternion, position_offset)
    # sup_pose = np.array([1,-1,-1,1])
    # sup_pose =sup_pose/np.abs(np.linalg.norm(sup_pose))
    # sup_pose /= 0.5
    for pp in point_pose:
        print(pp)
        result.append(create_pose(pp[0], pp[1], pp[2],
                                    pp[3], pp[4], pp[5] ,pp[6])) 
                                  
                                #   sup_pose[0], sup_pose[1], sup_pose[2], sup_pose[3]))
        # 0,0,0,0))
        #1,0,0,0)) this orientation works
        # 0,1,0,0))
        # 0,0,1,0)) #straight down
        # 0,0,0,1)) # straight down

    # sup_pose = np.array([1,1,-0.5,0.5]) # denne er ish det vi vil ha
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
        rclpy.spin_until_future_complete(self, future)
        
        self.get_logger().info(str(future.result().trajectory_fraction))
        
        
        
        
 
    	


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
