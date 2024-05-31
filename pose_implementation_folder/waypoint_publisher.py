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

# [1405.0378,315.728,-738.0498,0.8559,0.4707,-0.2139,0.0],
# [1401.3544,273.4491,-737.9873,0.8559,0.4707,-0.2139,0.0],
# [1397.7296,232.9701,-737.944,0.8559,0.4707,-0.2139,0.0],
# [1392.6167,186.8011,-737.9045,0.8559,0.4707,-0.2139,0.0],
# [1388.3708,143.3329,-737.3956,0.8559,0.4707,-0.2139,0.0],
# [1383.5903,93.0395,-737.4446,0.8559,0.4707,-0.2139,0.0],
# [1376.8381,46.0201,-735.7811,0.8559,0.4707,-0.2139,0.0],
# [1375.3088,-0.4538,-737.1496,0.8559,0.4707,-0.2139,0.0],
# [1374.605,-11.8161,-737.374,0.8559,0.4707,-0.2139,0.0],
# [1369.9189,-65.5627,-736.8203,0.8559,0.4707,-0.2139,0.0],
# [1362.4392,-110.5003,-734.9353,-0.3997,-0.8291,0.391,0.0],
# [1363.504,-110.9007,-735.3253,-0.3997,-0.8291,0.391,0.0],
# [1364.6336,-111.259,-735.9184,-0.3997,-0.8291,0.391,0.0],
# [1362.4392,-110.5003,-734.9353,-0.3997,-0.8291,0.391,0.0],
# [1362.0833,-110.8635,-735.2859,-0.3997,-0.8291,0.391,0.0],
# [1361.3299,-110.8734,-735.0172,-0.3997,-0.8291,0.391,0.0],
# [1358.0401,-111.2443,-735.9068,-0.3997,-0.8291,0.391,0.0],
# [1357.358,-111.3192,-735.6777,-0.3997,-0.8291,0.391,0.0],
# [1356.5165,-111.244,-735.8598,-0.3997,-0.8291,0.391,0.0],
# [1355.817,-111.1815,-735.6726,-0.3997,-0.8291,0.391,0.0],
# [1355.0003,-111.1007,-735.7568,-0.3997,-0.8291,0.391,0.0],
# [1354.0915,-110.9874,-736.0071,-0.3997,-0.8291,0.391,0.0],
# [1346.8503,-110.3777,-736.102,-0.3997,-0.8291,0.391,0.0],
# [1269.7523,-103.0134,-736.5529,-0.3997,-0.8291,0.391,0.0],
# [1245.0658,-99.9359,-736.3118,-0.3997,-0.8291,0.391,0.0],
# [1245.1693,-100.8325,-737.1428,-0.3997,-0.8291,0.391,0.0],
# [1245.2521,-101.6776,-737.3147,-0.3997,-0.8291,0.391,0.0],
# [1244.0632,-101.0525,-737.124,-0.3997,-0.8291,0.391,0.0],
# [1229.2892,-95.756,-734.5739,-0.0992,-0.9358,0.3382,0.0],
# [1228.5174,-95.59,-734.6655,-0.0992,-0.9358,0.3382,0.0],
# [1228.1521,-95.7086,-734.5174,-0.0992,-0.9358,0.3382,0.0],
# [1227.3723,-95.5273,-734.6753,-0.0992,-0.9358,0.3382,0.0],
# [1192.2369,-91.5129,-734.5063,-0.0992,-0.9358,0.3382,0.0],
# [1169.6949,-92.9223,-737.2055,-0.0992,-0.9358,0.3382,0.0],
# [1169.2545,-93.0597,-736.9918,-0.0992,-0.9358,0.3382,0.0],
# [1168.5786,-92.9236,-737.0863,-0.0992,-0.9358,0.3382,0.0],
# [1134.5361,-89.8123,-737.5346,-0.0992,-0.9358,0.3382,0.0],
# [1081.8914,-85.8242,-738.2834,-0.0992,-0.9358,0.3382,0.0],
# [1077.0009,-86.9022,-739.4278,-0.0992,-0.9358,0.3382,0.0],
# [1076.8688,-87.1906,-739.4677,-0.0992,-0.9358,0.3382,0.0],
# [1075.5525,-86.6157,-739.4186,-0.0992,-0.9358,0.3382,0.0],
# [1071.8902,-85.7545,-738.8222,-0.0992,-0.9358,0.3382,0.0],
# [1071.6156,-86.136,-739.2309,-0.0992,-0.9358,0.3382,0.0],
# [1071.024,-85.9649,-739.2765,-0.0992,-0.9358,0.3382,0.0],
# [1036.8904,-81.076,-738.196,-0.0992,-0.9358,0.3382,0.0]


# [1402.9178,319.2152,-738.1614,-0.6808,-0.4794,-0.5538,0.0],
# [1398.0831,278.2125,-737.3649,-0.6808,-0.4794,-0.5538,0.0],
# [1396.5539,269.9888,-737.2104,-0.6808,-0.4794,-0.5538,0.0],
# [1394.2005,235.2125,-737.5504,-0.6808,-0.4794,-0.5538,0.0],
# [1389.7853,190.4211,-737.4923,-0.6808,-0.4794,-0.5538,0.0],
# [1386.4292,148.1283,-737.2187,-0.6808,-0.4794,-0.5538,0.0],
# [1382.3178,98.3151,-737.5242,-0.6808,-0.4794,-0.5538,0.0],
# [1375.9303,48.1827,-736.3446,-0.6808,-0.4794,-0.5538,0.0],
# [1374.5778,-1.5945,-737.5236,-0.6808,-0.4794,-0.5538,0.0],
# [1373.8773,-7.8637,-737.6777,-0.6808,-0.4794,-0.5538,0.0],
# [1369.0619,-60.7192,-737.0613,-0.6808,-0.4794,-0.5538,0.0],
# [1362.0833,-110.8635,-735.2859,-0.6808,-0.4794,-0.5538,0.0],
# [1362.6048,-110.7791,-735.4788,-0.6808,-0.4794,-0.5538,0.0],
# [1363.4996,-110.902,-735.8294,-0.6808,-0.4794,-0.5538,0.0],
# [1362.0833,-110.8635,-735.2859,-0.6808,-0.4794,-0.5538,0.0],
# [1361.2891,-110.8704,-735.6111,-0.6808,-0.4794,-0.5538,0.0],
# [1360.7421,-111.0471,-735.3497,-0.6808,-0.4794,-0.5538,0.0],
# [1357.1699,-111.1734,-736.1839,-0.3614,-0.7469,-0.5581,0.0],
# [1356.5165,-111.244,-735.8598,-0.3614,-0.7469,-0.5581,0.0],
# [1355.6704,-111.1658,-736.0393,-0.3614,-0.7469,-0.5581,0.0],
# [1355.0003,-111.1007,-735.7568,-0.3614,-0.7469,-0.5581,0.0],
# [1354.0915,-110.9874,-736.0071,-0.3614,-0.7469,-0.5581,0.0],
# [1352.5164,-110.8999,-735.8102,-0.3614,-0.7469,-0.5581,0.0],
# [1274.5306,-103.4417,-736.6363,-0.3614,-0.7469,-0.5581,0.0],
# [1245.0658,-99.9359,-736.3118,-0.3614,-0.7469,-0.5581,0.0],
# [1245.1693,-100.8325,-737.1428,-0.3614,-0.7469,-0.5581,0.0],
# [1243.8414,-100.0281,-736.7654,-0.3614,-0.7469,-0.5581,0.0],
# [1244.0632,-101.0525,-737.124,-0.3614,-0.7469,-0.5581,0.0],
# [1229.2892,-95.756,-734.5739,-0.3614,-0.7469,-0.5581,0.0],
# [1228.5174,-95.59,-734.6655,-0.3614,-0.7469,-0.5581,0.0],
# [1227.7468,-95.4249,-734.7569,-0.3614,-0.7469,-0.5581,0.0],
# [1227.3723,-95.5273,-734.6753,-0.3614,-0.7469,-0.5581,0.0],
# [1196.7747,-92.227,-734.4749,-0.3614,-0.7469,-0.5581,0.0],
# [1169.6949,-92.9223,-737.2055,-0.3614,-0.7469,-0.5581,0.0],
# [1169.0105,-92.7752,-737.2291,-0.3614,-0.7469,-0.5581,0.0],
# [1168.5786,-92.9236,-737.0863,-0.3614,-0.7469,-0.5581,0.0],
# [1138.7807,-90.2224,-737.5721,-0.3614,-0.7469,-0.5581,0.0],
# [1086.021,-86.5451,-738.5621,-0.3614,-0.7469,-0.5581,0.0],
# [1077.0009,-86.9022,-739.4278,-0.3614,-0.7469,-0.5581,0.0],
# [1076.5286,-86.9585,-739.6255,-0.3614,-0.7469,-0.5581,0.0],
# [1075.5525,-86.6157,-739.4186,-0.3614,-0.7469,-0.5581,0.0],
# [1071.8902,-85.7545,-738.8222,-0.3614,-0.7469,-0.5581,0.0],
# [1071.6156,-86.136,-739.2309,-0.3614,-0.7469,-0.5581,0.0],
# [1071.024,-85.9649,-739.2765,-0.3614,-0.7469,-0.5581,0.0],
# [1036.8904,-81.076,-738.196,-0.3614,-0.7469,-0.5581,0.0]



# [1405.0777,316.9909,-738.0793,671.6522,-166.0833,212.643,0.0],
# [1401.672,277.0406,-737.9951,671.6522,-166.0833,212.643,0.0],
# [1397.4369,236.014,-737.9457,671.6522,-166.0833,212.643,0.0],
# [1393.2289,192.9389,-737.8905,671.6522,-166.0833,212.643,0.0],
# [1388.0094,148.2109,-737.4726,671.6522,-166.0833,212.643,0.0],
# [1383.9153,101.3914,-737.433,671.6522,-166.0833,212.643,0.0],
# [1376.5607,54.0926,-735.7872,671.6522,-166.0833,212.643,0.0],
# [1374.7718,-0.0746,-737.269,671.6522,-166.0833,212.643,0.0],
# [1369.7091,-55.2529,-737.1368,671.6522,-166.0833,212.643,0.0],
# [1362.0833,-110.8635,-735.2859,671.6522,-166.0833,212.643,0.0],
# [1362.6048,-110.7791,-735.4788,671.6522,-166.0833,212.643,0.0],
# [1363.4996,-110.902,-735.8294,671.6522,-166.0833,212.643,0.0],
# [1362.0833,-110.8635,-735.2859,671.6522,-166.0833,212.643,0.0],
# [1361.2891,-110.8704,-735.6111,671.6522,-166.0833,212.643,0.0],
# [1360.7421,-111.0471,-735.3497,671.6522,-166.0833,212.643,0.0],
# [1357.1699,-111.1734,-736.1839,671.6522,-166.0833,212.643,0.0],
# [1356.5165,-111.244,-735.8598,671.6522,-166.0833,212.643,0.0],
# [1363.4996,-110.902,-735.8294,-0.0986,-0.9618,0.2555,0.0],
# [1354.7802,-111.043,-736.1916,-0.0986,-0.9618,0.2555,0.0],
# [1354.0915,-110.9874,-736.0071,-0.0986,-0.9618,0.2555,0.0],
# [1285.8552,-104.4573,-736.6608,-0.0986,-0.9618,0.2555,0.0],
# [1245.3378,-99.7405,-736.9421,-0.0986,-0.9618,0.2555,0.0],
# [1245.1383,-100.2597,-736.9122,-0.0986,-0.9618,0.2555,0.0],
# [1244.2714,-100.0073,-736.9559,-0.0986,-0.9618,0.2555,0.0],
# [1244.2352,-100.7241,-737.5997,-0.0986,-0.9618,0.2555,0.0],
# [1244.1508,-101.1592,-737.7439,-0.0986,-0.9618,0.2555,0.0],
# [1242.6696,-100.2206,-737.2527,-0.0986,-0.9618,0.2555,0.0],
# [1230.9835,-95.7491,-734.8599,-0.0986,-0.9618,0.2555,0.0],
# [1230.1525,-95.5128,-734.9087,-0.0986,-0.9618,0.2555,0.0],
# [1229.364,-95.3259,-734.9793,-0.0986,-0.9618,0.2555,0.0],
# [1228.9975,-95.4244,-734.8222,-0.0986,-0.9618,0.2555,0.0],
# [1228.1687,-95.1952,-734.9476,-0.0986,-0.9618,0.2555,0.0],
# [1227.4901,-95.1318,-735.0236,-0.0986,-0.9618,0.2555,0.0],
# [1209.1722,-93.5819,-735.4394,-0.0986,-0.9618,0.2555,0.0],
# [1151.3616,-90.7013,-737.6491,-0.0986,-0.9618,0.2555,0.0],
# [1098.5069,-87.5711,-739.0764,-0.0986,-0.9618,0.2555,0.0],
# [1078.0497,-86.6108,-739.9375,-0.0986,-0.9618,0.2555,0.0],
# [1077.4938,-86.5525,-740.004,-0.0986,-0.9618,0.2555,0.0],
# [1075.1815,-86.3899,-740.0095,-0.0986,-0.9618,0.2555,0.0],
# [1074.604,-86.2451,-740.0326,-0.0986,-0.9618,0.2555,0.0],
# [1061.7943,-82.585,-738.5555,-0.0986,-0.9618,0.2555,0.0],
# [1061.5614,-82.894,-738.5505,-0.0986,-0.9618,0.2555,0.0],
# [1060.9327,-82.6174,-738.532,-0.0986,-0.9618,0.2555,0.0],
# [1060.2834,-82.3005,-738.4809,-0.0986,-0.9618,0.2555,0.0],
# [1060.0837,-82.544,-738.4286,-0.0986,-0.9618,0.2555,0.0],
# [1059.4674,-82.2104,-738.4252,-0.0986,-0.9618,0.2555,0.0],
# [1035.913,-79.673,-738.3974,-0.0986,-0.9618,0.2555,0.0]

# [1405.0777,316.9909,-738.0793,-659.831,-222.8146,-196.9263,0.0],
# [1401.1185,274.7689,-737.9588,-659.831,-222.8146,-196.9263,0.0],
# [1396.5747,231.1853,-737.9571,-659.831,-222.8146,-196.9263,0.0],
# [1393.4393,189.6717,-737.8962,-659.831,-222.8146,-196.9263,0.0],
# [1387.9111,142.618,-737.3846,-659.831,-222.8146,-196.9263,0.0],
# [1384.6276,96.9631,-737.3991,-659.831,-222.8146,-196.9263,0.0],
# [1376.885,46.4953,-735.7571,-659.831,-222.8146,-196.9263,0.0],
# [1375.7213,-1.8602,-737.0446,-659.831,-222.8146,-196.9263,0.0],
# [1375.8083,-1.3125,-736.9458,-659.831,-222.8146,-196.9263,0.0],
# [1374.8621,-10.8585,-737.2561,-659.831,-222.8146,-196.9263,0.0],
# [1370.4024,-64.1851,-736.7201,-659.831,-222.8146,-196.9263,0.0],
# [1364.3125,-111.0911,-736.1866,-659.831,-222.8146,-196.9263,0.0],
# [1363.237,-110.8019,-736.2386,-659.831,-222.8146,-196.9263,0.0],
# [1362.3148,-110.658,-736.3779,-659.831,-222.8146,-196.9263,0.0],
# [1361.831,-110.7791,-736.2069,-659.831,-222.8146,-196.9263,0.0],
# [1360.9375,-110.6917,-736.4741,-659.831,-222.8146,-196.9263,0.0],
# [1360.2294,-110.7197,-736.6233,-0.4397,0.8598,-0.2596,0.0],
# [1356.6194,-110.8036,-736.9263,-0.4397,0.8598,-0.2596,0.0],
# [1356.0471,-110.9541,-736.6516,-0.4397,0.8598,-0.2596,0.0],
# [1355.1608,-110.835,-736.805,-0.4397,0.8598,-0.2596,0.0],
# [1354.2353,-110.678,-736.9341,-0.4397,0.8598,-0.2596,0.0],
# [1353.6512,-110.7226,-736.8104,-0.4397,0.8598,-0.2596,0.0],
# [1348.8198,-110.2905,-736.9286,-0.4397,0.8598,-0.2596,0.0],
# [1270.5873,-102.0931,-737.0915,-0.4397,0.8598,-0.2596,0.0],
# [1245.3378,-99.7405,-736.9421,-0.4397,0.8598,-0.2596,0.0],
# [1244.1738,-99.1226,-736.6836,-0.4397,0.8598,-0.2596,0.0],
# [1244.2652,-100.0075,-737.507,-0.4397,0.8598,-0.2596,0.0],
# [1244.2352,-100.7241,-737.5997,-0.4397,0.8598,-0.2596,0.0],
# [1243.0993,-100.152,-737.4431,-0.4397,0.8598,-0.2596,0.0],
# [1231.4683,-95.5829,-735.0168,-0.4397,0.8598,-0.2596,0.0],
# [1230.5897,-95.2994,-735.1154,-0.4397,0.8598,-0.2596,0.0],
# [1229.8574,-95.1609,-735.1477,-0.4397,0.8598,-0.2596,0.0],
# [1229.364,-95.3259,-734.9793,-0.4397,0.8598,-0.2596,0.0],
# [1228.575,-95.1376,-735.0516,-0.4397,0.8598,-0.2596,0.0],
# [1195.3051,-91.362,-734.9695,-0.4397,0.8598,-0.2596,0.0],
# [1139.2186,-90.0894,-737.7871,-0.4397,0.8598,-0.2596,0.0],
# [1084.8425,-86.2511,-738.6949,-0.4397,0.8598,-0.2596,0.0],
# [1078.0794,-86.6024,-739.4933,-0.4397,0.8598,-0.2596,0.0],
# [1077.479,-86.9341,-739.7188,-0.4397,0.8598,-0.2596,0.0],
# [1077.0009,-86.9022,-739.4278,-0.4397,0.8598,-0.2596,0.0],
# [1076.5286,-86.9585,-739.6255,-0.4397,0.8598,-0.2596,0.0],
# [1075.5525,-86.6157,-739.4186,-0.4397,0.8598,-0.2596,0.0],
# [1071.7915,-85.7396,-739.1726,-0.4397,0.8598,-0.2596,0.0],
# [1071.3982,-85.8303,-739.4569,-0.4397,0.8598,-0.2596,0.0],
# [1071.024,-85.9649,-739.2765,-0.4397,0.8598,-0.2596,0.0],
# [1037.7594,-81.0714,-738.2835,-0.4397,0.8598,-0.2596,0.0]

# [1405.0777,316.9909,-738.0793,0.6829,-0.09,-0.725,0.0],
# [1401.6192,275.4242,-738.0079,0.6829,-0.09,-0.725,0.0],
# [1397.0518,231.8598,-737.9903,0.6829,-0.09,-0.725,0.0],
# [1393.8518,190.3878,-737.8911,0.6829,-0.09,-0.725,0.0],
# [1388.624,143.6776,-737.4048,0.6829,-0.09,-0.725,0.0],
# [1384.1777,94.2191,-737.3871,0.6829,-0.09,-0.725,0.0],
# [1377.2001,58.4499,-734.9262,0.6829,-0.09,-0.725,0.0],
# [1377.1829,48.4301,-735.4243,0.6829,-0.09,-0.725,0.0],
# [1375.5717,13.3359,-736.0076,0.6829,-0.09,-0.725,0.0],
# [1375.8226,-0.7968,-736.8616,0.6829,-0.09,-0.725,0.0],
# [1375.8509,-0.2081,-736.7416,0.6829,-0.09,-0.725,0.0],
# [1375.4694,-8.4479,-737.0358,0.6829,-0.09,-0.725,0.0],
# [1370.2607,-66.4621,-736.7357,0.6829,-0.09,-0.725,0.0],
# [1363.504,-110.9007,-735.3253,0.6829,-0.09,-0.725,0.0],
# [1362.4392,-110.5003,-734.9353,0.6829,-0.09,-0.725,0.0],
# [1359.3636,-111.1971,-735.8955,0.6829,-0.09,-0.725,0.0],
# [1358.7447,-111.2153,-735.7576,-0.0514,-0.6929,-0.7192,0.0],
# [1356.5165,-111.244,-735.8598,-0.0514,-0.6929,-0.7192,0.0],
# [1355.6704,-111.1658,-736.0393,-0.0514,-0.6929,-0.7192,0.0],
# [1349.2316,-110.5271,-736.1119,-0.0514,-0.6929,-0.7192,0.0],
# [1271.8843,-103.3042,-736.473,-0.0514,-0.6929,-0.7192,0.0],
# [1244.8833,-100.4926,-736.3792,-0.0514,-0.6929,-0.7192,0.0],
# [1245.2521,-101.6776,-737.3147,-0.0514,-0.6929,-0.7192,0.0],
# [1244.9744,-101.8926,-737.3216,-0.0514,-0.6929,-0.7192,0.0],
# [1243.4016,-100.7871,-736.7635,-0.0514,-0.6929,-0.7192,0.0],
# [1229.7151,-96.0594,-734.35,-0.0514,-0.6929,-0.7192,0.0],
# [1228.8613,-95.7943,-734.4515,-0.0514,-0.6929,-0.7192,0.0],
# [1228.1521,-95.7086,-734.5174,-0.0514,-0.6929,-0.7192,0.0],
# [1227.7193,-95.9115,-734.3933,-0.0514,-0.6929,-0.7192,0.0],
# [1195.1711,-92.2167,-734.206,-0.0514,-0.6929,-0.7192,0.0],
# [1169.5267,-93.191,-736.6746,-0.0514,-0.6929,-0.7192,0.0],
# [1168.851,-93.0569,-736.8378,-0.0514,-0.6929,-0.7192,0.0],
# [1168.3607,-93.124,-736.5807,-0.0514,-0.6929,-0.7192,0.0],
# [1136.4258,-90.1245,-736.8107,-0.0514,-0.6929,-0.7192,0.0],
# [1094.5074,-87.057,-737.6609,-0.0514,-0.6929,-0.7192,0.0],
# [1091.1141,-86.0895,-737.2767,-0.0514,-0.6929,-0.7192,0.0],
# [1088.9268,-85.4352,-736.7657,-0.0514,-0.6929,-0.7192,0.0],
# [1084.3807,-85.7943,-737.4512,-0.0514,-0.6929,-0.7192,0.0],
# [1083.6859,-85.1588,-736.8369,-0.0514,-0.6929,-0.7192,0.0],
# [1078.1674,-85.9819,-738.045,-0.0514,-0.6929,-0.7192,0.0],
# [1065.0917,-83.2455,-736.7948,-0.0514,-0.6929,-0.7192,0.0],
# [1064.9738,-84.8281,-738.1215,-0.0514,-0.6929,-0.7192,0.0],
# [1059.7354,-83.1786,-737.1913,-0.0514,-0.6929,-0.7192,0.0],
# [1034.9014,-81.2702,-737.5184,-0.0514,-0.6929,-0.7192,0.0]

# ]

point_pose_input = np.array(pd.read_csv("/home/zivid/pytorch_env/OUTPUT.csv", sep = ',', header= None)) # read the data shown above from the file at the given location, allows for updating the path without rebuilding the workspace
print(f"{point_pose_input= }")

#position and pose for our coordinate frame relative to to world:
# position_baseframe = [-1.1072, 2.2256, 0.82]
position_baseframe = [-1.1072, 2.2256, 1.325]

orientation_baseframe = [0, 0, -0.26303, 0.96479]

world = [0,0,0,1]

robot2_capture_pos_1 = [-0.51684, 2.2927, 1.0029]
# robot2_capture_quat_1 = [0.13725, 0.32965, -0.39001, 0.84875]

robot2_capture_quat_1= [0.6972, -0.042663, -0.50888, 0.50311]
robot2_capture_point_pose_1 = [-0.51684, 2.2927, 1.0029, 0.6972, -0.042663, -0.50888, 0.50311]

robot2_capture_quat_2 = [0.22, -0.48, -0,59, 0.62]
robot2_capture_point_pose_2 = [-0.51684, 2.2927, 1.0029, 0.22, -0.48, -0.59, 0.62]




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
    
def create_image_pose(current_orientation_euler):

    # Example current orientation in Euler angles (roll, pitch, yaw) in degrees

    # Convert current orientation to rotation matrix
    current_rotation = R.from_euler('xyz', current_orientation_euler, degrees=True)
    current_rotation_matrix = current_rotation.as_matrix()

    # Define the desired "up" direction (z-axis pointing up in Cartesian space)
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
    global offset
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


    offset_local = 0.0 # 0.70 is good for testing
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
        # pose = quaternion_mult(pose, rotation_quaternion)

                # quat= [1,0,0,0]
        # quat /= np.linalg.norm(quat)
        # pose = point_rotation_by_quaternion(pose, quat) #THIS IS ADDED AS AN EXPERIMENT, NOT THE OG IMPLEMENTATION!!!! 

        # pose = point_rotation_by_quaternion(pose, np.array(pose)/4) #THIS IS ADDED AS AN EXPERIMENT, NOT THE OG IMPLEMENTATION!!!! 

        i,j,k,w = pose/np.abs(np.linalg.norm(pose))#alltid normaliser etter bruk
        point_pose_out.append([x1,y1,z1+offset_local+ offset, i,j,k,w])
    assert np.shape(point_pose_in) == np.shape(point_pose_out), f"Error, the resulting point_pose with shape: {np.shape(point_pose_out)}, does not match the input point_pose with shape: {np.shape(point_pose_in)}"
    return point_pose_out


def set_pose_orientation(pose ,angle_radians):
    i, j, k, w = pose
    angle_rad = np.arccos(w)

    i /= angle_rad
    j /= angle_rad
    k /= angle_rad

    s = np.sin(angle_radians)
    c = np.cos(angle_radians)
    new_pose = [i*s, j*s, k*s, c]

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


    ########### this kinda works, fix orientation?
    xi, yi, zi = robot2_capture_pos_1
    input_vec = np.array([0.3, -0.3, -0.7])
    input_vec /= np.linalg.norm(input_vec)

    quat = input_vec_to_rotation_quat(input_vec)
    # quat = set_pose_orientation(quat, np.pi/2)
    ii, ji, ki, wi = quat



    result.append(create_pose(xi+0.20, yi-0.20, zi-0.25, ii, ji, ki, wi)) 

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
