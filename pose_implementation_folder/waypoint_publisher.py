import rclpy
import math
from rclpy.node import Node
import sys

from std_msgs.msg import String
from moveit_group_planner_interfaces.msg import Waypoints
from moveit_group_planner_interfaces.srv import Plan
from geometry_msgs.msg import Pose

# import pandas as pd

DEBUG = True

point_pose_input = [[1404.9008,318.1502,-738.1728,0.9362,-0.0793,-0.3424,0.0],
[1403.6824,299.134,-738.0828,0.9362,-0.0793,-0.3424,0.0],
[1401.6535,280.3287,-737.963,0.9362,-0.0793,-0.3424,0.0],
[1399.5942,261.1582,-737.8514,0.9362,-0.0793,-0.3424,0.0],
[1397.6347,241.182,-737.9442,0.9362,-0.0793,-0.3424,0.0],
[1395.4201,220.8562,-737.8583,0.9362,-0.0793,-0.3424,0.0],
[1393.64,200.3873,-737.9326,0.9362,-0.0793,-0.3424,0.0],
[1391.5335,179.6207,-737.8311,0.9362,-0.0793,-0.3424,0.0],
[1389.0904,158.5628,-737.5624,0.9362,-0.0793,-0.3424,0.0],
[1387.0082,136.8783,-737.4965,0.9362,-0.0793,-0.3424,0.0],
[1384.9629,114.1762,-737.5927,0.9362,-0.0793,-0.3424,0.0],
[1382.571,91.1286,-737.4743,0.9362,-0.0793,-0.3424,0.0],
[1378.2929,69.1974,-736.1708,0.9362,-0.0793,-0.3424,0.0],
[1376.3571,45.284,-736.2238,0.9362,-0.0793,-0.3424,0.0],
[1374.0671,20.975,-736.0766,0.9362,-0.0793,-0.3424,0.0],
[1374.571,-6.096,-737.5502,0.9362,-0.0793,-0.3424,0.0],
[1371.7321,-32.1,-737.2782,0.9362,-0.0793,-0.3424,0.0],
[1369.0388,-58.8697,-737.0621,0.9362,-0.0793,-0.3424,0.0],
[1366.5775,-86.0737,-736.8889,0.9362,-0.0793,-0.3424,0.0],
[1361.3299,-110.8734,-735.0172,-0.712,0.4577,0.5325,0.0],
[1362.0833,-110.8635,-735.2859,-0.712,0.4577,0.5325,0.0],
[1362.6048,-110.7791,-735.4788,-0.712,0.4577,0.5325,0.0],
[1361.3299,-110.8734,-735.0172,-0.712,0.4577,0.5325,0.0],
[1360.7421,-111.0471,-735.3497,-0.712,0.4577,0.5325,0.0],
[1360.0946,-111.0393,-735.1967,-0.712,0.4577,0.5325,0.0],
[1359.5513,-111.2238,-735.4403,-0.712,0.4577,0.5325,0.0],
[1356.5165,-111.244,-735.8598,-0.712,0.4577,0.5325,0.0],
[1362.0833,-110.8635,-735.2859,-0.712,0.4577,0.5325,0.0],
[1355.0003,-111.1007,-735.7568,-0.712,0.4577,0.5325,0.0],
[1354.0915,-110.9874,-736.0071,-0.712,0.4577,0.5325,0.0],
[1353.393,-111.0127,-735.6524,-0.712,0.4577,0.5325,0.0],
[1325.597,-108.3266,-736.0625,-0.712,0.4577,0.5325,0.0],
[1289.7116,-104.9207,-736.4541,-0.712,0.4577,0.5325,0.0],
[1255.5129,-101.3094,-736.3566,-0.712,0.4577,0.5325,0.0],
[1245.0658,-99.9359,-736.3118,-0.712,0.4577,0.5325,0.0],
[1245.1693,-100.8325,-737.1428,-0.712,0.4577,0.5325,0.0],
[1245.2521,-101.6776,-737.3147,-0.712,0.4577,0.5325,0.0],
[1244.0632,-101.0525,-737.124,-0.712,0.4577,0.5325,0.0],
[1229.2892,-95.756,-734.5739,-0.828,0.3363,0.4487,0.0],
[1228.5174,-95.59,-734.6655,-0.828,0.3363,0.4487,0.0],
[1228.1521,-95.7086,-734.5174,-0.828,0.3363,0.4487,0.0],
[1227.3723,-95.5273,-734.6753,-0.828,0.3363,0.4487,0.0],
[1214.6662,-94.5165,-734.8266,-0.828,0.3363,0.4487,0.0],
[1185.1147,-91.5417,-734.9423,-0.828,0.3363,0.4487,0.0],
[1169.6949,-92.9223,-737.2055,-0.828,0.3363,0.4487,0.0],
[1169.2545,-93.0597,-736.9918,-0.828,0.3363,0.4487,0.0],
[1168.5786,-92.9236,-737.0863,-0.828,0.3363,0.4487,0.0],
[1157.8228,-91.9309,-737.0523,-0.828,0.3363,0.4487,0.0],
[1131.3581,-89.6121,-737.6629,-0.828,0.3363,0.4487,0.0],
[1106.4814,-87.7531,-738.0636,-0.828,0.3363,0.4487,0.0],
[1082.3668,-85.8447,-738.5638,-0.828,0.3363,0.4487,0.0],
[1077.0009,-86.9022,-739.4278,-0.828,0.3363,0.4487,0.0],
[1076.5286,-86.9585,-739.6255,-0.828,0.3363,0.4487,0.0],
[1071.6156,-86.136,-739.2309,-0.828,0.3363,0.4487,0.0],
[1071.024,-85.9649,-739.2765,-0.828,0.3363,0.4487,0.0],
[1059.2031,-82.9932,-738.1105,-0.828,0.3363,0.4487,0.0]]

#the translation from worldframe to workpiece center.
cy = 1.532
cx = 0.0
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


def point_rotation_by_quaternion(point,q):
    #q = xyzw
    r = point + [0] #adds w = 0 to xyz point for such that r represent a quaternion
    q_conj = quaternion_qunj(q)
    return quaternion_mult(quaternion_mult(q,r),q_conj)[:3]#returns xyz rotated

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


def zivid_job(point_pose_list):
    result = []
    for pp in point_pose_list:
        result.append(create_pose(pp[0]* 10e-3, pp[1]* 10e-3, pp[2]* 10e-3, pp[3], pp[4], pp[5] ,pp[6]))
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
        msg.waypoints = zivid_job(point_pose_input)
    if (len(msg.is_job) != len(msg.waypoints)):
            for w in msg.waypoints:
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
