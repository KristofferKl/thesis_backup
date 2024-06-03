#import roslib
#roslib.load_manifest("urdfdom_py")
#import rospy
import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from urdf_parser_py.urdf import URDF




def skew(axis): #returns skew representation of the 3x1 vector
    assert(len(axis) == 3)
    return np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0],0]])
                     
def exp3(axis, theta = 0): #Rotation metrix from axis and magnitude w, theta
    return np.eye(3) + np.sin(theta) * skew(axis) + (1 - np.cos(theta)) * skew(axis) @ skew(axis)

def Tmat(R, r): #Transformation matrix
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = r
    return T

def getPrefixes(jointlist) -> list: #returns the group names = prefix: group_1/joint_1 -> group_1
    #todo, what i no prefix?
    prefixes = []
    for joint in jointlist:
        prefix = joint.name.split("/")[0]
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes
    
def getTransformationInChain(chain): #returns a list with transformations from joint_i to joint_i+1
    Ts = []
    for i in range(len(chain)):
        #chain[i].origin is the transformation to given frame relative to previous frame
        dx = chain[i].origin.xyz[0] 
        dy = chain[i].origin.xyz[1] 
        dz = chain[i].origin.xyz[2] 
        r = np.array([dx, dy, dz])
        
        axis      = np.array([chain[i].origin.rpy[0], chain[i].origin.rpy[1], chain[i].origin.rpy[2]])
        #because of the way the urdf is set up, if there are no rotation between the frame, the norm of "axis" will be 0
        #if there are any rotations, the norm will be non-zero
        #Furthermore, a joint rotate about its "z" axis, so that this doesn|t really makes sense
        #However, if the slt model is defined such that each joint/link follows the same coordinate system
        #the axis will be correct with the respect of global frame, and it can be used with PoE conventions
        theta = np.linalg.norm(axis)
        if theta != 0:
            axis /= theta

        rot = exp3(axis, theta)        
        diff_rot = rot
        Ts.append(Tmat(diff_rot, r))
    return Ts
            
    
    
def getGroups(jointlist) -> dict: #returns a dict {group, [joints]} #if group_x exist as child of group_y, then group_y includes joints of group_x. but not other way
    prefixes = getPrefixes(jointlist)
    groups = {}
    for prefix in prefixes:
        groups[prefix] = []
    
    for joint in jointlist:
        for prefix in prefixes:
            if prefix in joint.name:
                groups[prefix].append(joint)
    return groups
    

def getChain(jointlist) -> list: #A group may contain additional "virtual" fixed joint/links which will not be a part of the kinematic

    """Assuming each group has a "joint_1",
    this function returns the chain defined from the first joint and out.
    This will not take links before the first dynamic joint into account
    
    for example [fixed_joint_1, fixed_joint_2, joint_1, joint_2 ...] -> [joint_1, joint_2 ...]
    where fixed_jont_1 etc is joints defining the fixed transformation between two frames/joints
    """
    chains = {}
    
    groups = getGroups(jointlist)
    for key in groups:
        chain = []
        group_joints = groups[key]
    
        for i, joint in enumerate(group_joints):
            if "joint_1" in joint.name: ##if "fixed" not in joint.type. changed because group_n+1 may be the parent of group_n
                chain.append(joint)
                child = joint.child
                for j, obj in enumerate(jointlist):
                    if obj.parent == child:
                        chain.append(obj)
                        child = obj.child
                
                chains[key] = chain
            chain = []
    return chains


def Slist(chain): #generate a np.array representation of the spatial home twist S
    #takes a chain and return the spatial twist in home position
    #e^(s*theta) = T
    #The M position (home position)
    Ts =  getTransformationInChain(chain)
    qs = []
    Rs = []
    
    M = np.eye(4) #Tsb
    for i in Ts:
        M = M @ i
        qs.append(M[:3,3])
        Rs.append(M[:3, :3])

    #S_i = rotation_axis, -(w x q)
    
    S = []
    for i, joint in enumerate(chain):
        #only supported for revolute and prismatic joints
        if joint.type == "revolute":
            #from urdf, the joint.axis is with respect of the joint in question
            #it is however defined as rotation of the geometries origin (.slt file)
            
            w_s = np.array([joint.axis[0], joint.axis[1], joint.axis[2]])
            v = -skew(w_s) @ qs[i] #qs[i]
            S.append(np.hstack((w_s, v)))

        if joint.type == "prismatic":
            w_s = np.array([0,0,0])
            v = np.array([joint.axis[0], joint.axis[1], joint.axis[2]])
            S.append(np.hstack((w_s, v)))
    #[s_1, s_2, s_3, ...]
    return np.array(S).T
    




def calculateSpatialJacobian(chain, thetalist, S = None):
    if S is None:
        S = Slist(chain)
    return mr.JacobianSpace(S, thetalist)

def sToBtwist(chain, s): #change of frame
    M = getM(chain)
    #To change the reference
    #Vb = Ad(Tbs) Vs
    return Adjoint(np.linalg.inv(M)) @ s

def sToBjac(chain, J_s): #change of frame
    M = getM(chain)
    return mr.Adjoint(np.linalg.inv(M)) @ J_s    
  
def getM(chain): #generates the home matrix M
    Ts = getTransformationInChain(chain)
    M = np.eye(4) #Tsb
    for T in Ts:
        M = M @ T
    return M  
    

def fk(chain, theta, S = None): #forward kinemtics
    #T0-n = Prod(exp6(s_i, theta_i))@M
    
    M = getM(chain)
    if S is None:
        S = Slist(chain)

    return mr.FKinSpace(M, S, theta)    

 
 
def capSpeed(chain, velocitylist, positionlist, timelist, max_speed = None, timestep = 0.1): #caps the speed to desired max speed

    """takes a trajectory, max cartesian speed and timestep
       returns a trajectory following the same paths but with max_speed velocities
       
       from moveit, timestep seems to be 0.1 but this is not nessisarily the case
    """
    
    #len position = len time = len velocity
    
    
    newTimelist     = []
    newVelocitylist = []
    newPositionlist = []


    #plan:
    #   for pos, vel:
    #       is cartesian vel > max_speed?
    #           new_vel = max_vel
    #           new_pos = pos + vel * timestep
    #       else
    #           insert old
    i = 0
    
    prev_time = 0.0
    delta = 0.0
    newTimelist.append(0) #first step at time 0
    S = Slist(chain)
    for theta, dtheta, t in zip(positionlist, velocitylist, timelist): #for all datapoints
        jac = sToBjac(chain, calculateSpatialJacobian(chain, theta, S))   #Jac for given position, changes frame
        end_effector_twist = jac @ dtheta
        end_effector_speed = (np.linalg.norm(end_effector_twist[3:]))
        end_effector_cartesian_speed = 0
        
        
        if (i != len(positionlist)-1): #if there exist a next point
            fk_current = fk(chain, theta, S)
            fk_next    = fk(chain, positionlist[i+1], S)
            euclidian_diff = fk_next[:3,3] - fk_current[:3,3]   # delta t
            #rotation_diff  = fk_next[:3,:3] - fk_current[:3,:3] # R'
            timestep = timelist[i+1] - timelist[i]
            end_effector_cartesian_speed = np.linalg.norm(euclidian_diff) / timestep #t'
            
            

            if end_effector_cartesian_speed > max_speed + 1.e-6: #plus a little tolerance because of float precision
                #need to scale v (and thus twist) = t'-R'R.T t such that t' = max_speed
                #t2-t1 is constant and therefore delta T is the variable as t' = t2-t1 / delta T = cartesian speed
                scalefactor = end_effector_cartesian_speed/max_speed
                #because v = t'-R'R = t_2 - t1 / delta_t - R/delta_t R_1 t_1 = can move delta_t outside
                #and new v (and thus new twist) = 1/scaling_factor = 1/alpha * v_old
                scaled_twist = end_effector_twist/scalefactor #desired twist
                
                #if jac.shape[0] == jac.shape[1]:  # If Jacobian matrix is square
                #    inverse_jac = np.linalg.inv(jac)
                #else:  # If Jacobian matrix is not square i.e not 6 dof robot arm, need psuedo inverse
                #    inverse_jac = np.linalg.pinv(jac)
                
                inverse_jac = np.linalg.pinv(jac) #this works for all cases
                new_velocity = inverse_jac @ scaled_twist
                newTimelist.append(np.linalg.norm(euclidian_diff)/max_speed + newTimelist[-1]) #timestep = distance/speed
                
                
            else: #max speed > current speed, do not do anything
                new_velocity = dtheta #new is the same as old
                newTimelist.append(timestep + newTimelist[-1]) #no changes, add the difference in time from old plan to the last element in the new
            newVelocitylist.append(new_velocity)
           
        else: #if i == len(positionlist)
            newVelocitylist.append(velocitylist[-1]) #should be all zeros

        i += 1

    return newVelocitylist, positionlist, newTimelist
            


def main():
    robot = URDF.from_xml_file("/home/johan/ws_test2/src/motoman/motoman_gp25sys_support/urdf/gp25sys.urdf") #read the parameters from file
    #robot = URDF.from_parameter_server()
    joints = robot.joints
    robotchains = getChain(joints)
    
    MAX_SPEED = 0.1 #the desired max speed
    GROUPNAME = "group_3" #the desired group, because group 3 is the root of group 1, group 3 for the 7-axis plan created in planning node
    fontsz = 24 #fontsize for plots
    CASE = "ItTP"
    #lists to store information of the trajectory
    timelist = []
    velocitylist = []
    positionlist = []
    
    #lists to store calculated information of velocities
    endeffectorvelocity_space = []
    endeffectorvelocity_body = []
    endeffectorabsvelocity_space = []
    endeffectorabsvelocity_body = []
    endeffectorabsangular_body = []
    
    #lists to store the forward kinematics
    cartesianPosx = []
    cartesianPosy = []
    cartesianPosz = []
    
    #start, stop-index representing the position of the desired joints in the files imported in the lines beneeth
    startidx, endidx = 7, 14 #idx 0 = time, 1-15 = joints group2: 1,7    group_3: 7-14

    with open("/home/johan/debugs/positions.txt", "r") as file: #reading positions
        lines = file.readlines()
        for line in lines:
            if line.strip():
                vals = [float(i) for i in line.split(" ") if i.strip()]  # skip empty strings
                #timelist.append(vals[0])
                positionlist.append(np.array(vals[startidx:endidx]))
    
    with open("/home/johan/debugs/positions.txt", "r") as file: #reading time
        lines = file.readlines()
        for line in lines:
            if line.strip():
                time = line.split(" ")[0]
                secs, nanos = time.split(".")
                if len(nanos) < 9: #asserts that x sec and 999 nanosec = x sec and 0.000000999 nanosec instead of 1 sec and 999000000 nanosec
                    nanos = "0"*(9-len(nanos)) + nanos
                timelist.append(float(secs + "." + nanos))
    
    
    with open("/home/johan/debugs/velocities.txt", "r") as file: #reading velocities
        lines = file.readlines()
        for line in lines:
            if line.strip():
                vals = [float(i) for i in line.split(" ") if i.strip()]  # skip empty strings
                velocitylist.append(np.array(vals[startidx:endidx]))
                
    chain = robotchains[GROUPNAME] #the desired group to calculate the kinematics for, group_3 is the base of group_2
    jointnames = []
    S = Slist(chain) #precalculate
    for joint in chain:
        if joint.type != "fixed":
            jointnames.append(joint.name)


    for theta, dtheta in zip(positionlist, velocitylist): #for each position
        #twist = [w, v] = angular velocity, linear velocity
        jac = calculateSpatialJacobian(chain, theta, S)
        endeffectorvelocity_space.append(jac @ dtheta)
        endeffectorabsvelocity_space.append(np.linalg.norm(endeffectorvelocity_space[-1][3:]))
        endeffectorvelocity_body.append(sToBjac(chain, jac) @ dtheta)
        endeffectorabsvelocity_body.append(np.linalg.norm(endeffectorvelocity_body[-1][3:]))
        endeffectorabsangular_body.append(np.linalg.norm(endeffectorvelocity_body[-1][:3]))
    
        #plotting the forward kinematics
        T = fk(chain, theta, S)
        cartesianPosx.append(T[0,-1])
        cartesianPosy.append(T[1,-1])
        cartesianPosz.append(T[2,-1])

    #fontsize

    plt.figure()
    plt.suptitle(GROUPNAME, fontsize = 32)
    plt.subplot(2,1,1) # 2 rows, 1 column, subplot 1
    plt.title(CASE, fontsize = fontsz)
    plt.plot(timelist, endeffectorabsvelocity_body, label="End effector linear velocity, v")
    plt.plot(timelist, [MAX_SPEED]*len(timelist), "--", label = f"Goal: {MAX_SPEED} [m/s]")
    plt.xlabel("time [s]", fontsize =  fontsz)
    plt.ylabel("Velocity [m/s]", fontsize = fontsz)
    plt.legend(fontsize = fontsz, loc = "upper left")

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ### 3D plot for plotting end effector trajectory ###
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #
    #
    #ax.scatter(cartesianPosx, cartesianPosy, cartesianPosz)
    #for i in range(len(timelist)):
    #    ax.text(cartesianPosx[i], cartesianPosy[i], cartesianPosz[i], str(timelist[i]), color='red')
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
 
    #plt.show()
 

    #velocitylimit with jacobian
    newVelocitylist, newpositionlist, newtimelist = capSpeed(chain, velocitylist, positionlist, timelist, max_speed = MAX_SPEED)
    endeffectorvelocity_space = []
    endeffectorvelocity_body = []
    endeffectorabsvelocity_space = []
    endeffectorabsvelocity_body = []
    cartesianPosx = []
    cartesianPosy = []
    cartesianPosz = []


    for theta, dtheta in zip(newpositionlist, newVelocitylist):
        #twist = [w, v] = angular velocity, linear velocity
        jac = calculateSpatialJacobian(chain, theta, S)
        endeffectorvelocity_space.append(jac @ dtheta)
        endeffectorabsvelocity_space.append(np.linalg.norm(endeffectorvelocity_space[-1][3:]))
        endeffectorvelocity_body.append(sToBjac(chain, jac) @ dtheta)
        endeffectorabsvelocity_body.append(np.linalg.norm(endeffectorvelocity_body[-1][3:]))
        endeffectorabsangular_body.append(np.linalg.norm(endeffectorvelocity_body[-1][:3]))

        
    
    
    plt.subplot(2,1,2)
    plt.title("Twist manipulation", fontsize = fontsz)
    plt.plot(newtimelist, endeffectorabsvelocity_body, label="End effector linear velocity v")
    plt.plot(newtimelist, [MAX_SPEED]*len(timelist), "--", label = f"Goal: {MAX_SPEED} [m/s]")
    plt.xlabel("time [s]",fontsize = fontsz)
    plt.ylabel("Velocity [m/s]",fontsize = fontsz)
    plt.legend(fontsize = fontsz, loc = "upper left")
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    
    #joint velocities plot
    plt.figure()
    plt.suptitle(GROUPNAME, fontsize = 32)
    plt.subplot(2, 1, 2) # 2 rows, 1 column, subplot 1
    plt.title("Twist manipulation",fontsize = fontsz)
    for i in range(len(newVelocitylist[0])): #for each joint
        plt.plot(newtimelist, [x[i] for x in newVelocitylist],label = jointnames[i])
    plt.legend(fontsize = fontsz, loc = "upper left")
    plt.xlabel("time [s]",fontsize = fontsz)
    plt.ylabel("Velocity [rad/s]",fontsize = fontsz)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(2, 1, 1) # 2 rows, 1 column, subplot 2
    plt.title(CASE,fontsize = fontsz)
    for i in range(len(velocitylist[0])):
        plt.plot(timelist, [x[i] for x in velocitylist], label = jointnames[i])
    
    plt.legend(fontsize = fontsz, loc = "upper left")
    plt.xlabel("time [s]",fontsize = fontsz)
    plt.ylabel("Velocity [rad/s]",fontsize = fontsz)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    
    
if __name__ == '__main__':
    main()
