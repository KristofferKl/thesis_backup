#include "utilities.h"

bool DEBUG = false;

int log(std::string data, const int verbose){
  if (verbose == 0){

    std::cout << data << std::endl;
  }
  else if (verbose > 0 and DEBUG){
    std::cout << data << std::endl;
  }
  return 1;
}

//used to find how many joints in a group
//returns the number of occurences in a vector, and the first position of where the index is found in the dataset
//for example for group_1: [group_4/joint_1, group_1/joint_1, group_1/joint_2] -> group_1, (2, 1)
std::vector<std::pair<std::string, std::pair<size_t, size_t>>> findNumberOfMatches(const std::vector<std::string>& keywords, const std::vector<std::string>& dataset) {
  std::vector<std::pair<std::string, std::pair<size_t, size_t>>> results;

  for (const auto& sequence : keywords){
    size_t counter  = 0;
    size_t idx_counter = 0;
    //for all groups 
    for (const auto& group : dataset){
      //see if sequence is found, and how many
      if (group.find(sequence) != std::string::npos){
        counter ++;
      }
      if (counter == 0){
        idx_counter ++; //vi vil bare finne posisjonen til den første matches.
      }
    }
    results.push_back(std::make_pair(sequence, std::make_pair(counter, idx_counter)));

  }
  return results;
}
//prints the joint values of a plan
int printTrajectory(moveit::planning_interface::MoveGroupInterface::Plan plan){
  for (auto it : plan.trajectory_.joint_trajectory.points){
    std::cout <<"t: " <<it.time_from_start.sec << "."<<it.time_from_start.nanosec<<" ";
    for (auto point : it.positions){
      std::cout << point << " ";
    }
    std::cout << std::endl;
  }
  return 1;
}

bool compareByIndex(const std::pair<std::string, std::pair<size_t, size_t>>& a, const std::pair<std::string, std::pair<size_t, size_t>>& b) {
    return a.second.second < b.second.second;
}

//checks if group exist in plan
bool isGroupInPlan(const std::string& group, const moveit::planning_interface::MoveGroupInterface::Plan& plan){
    if (plan.trajectory_.joint_trajectory.joint_names[0].find(group) != std::string::npos){
      return true;
    }
  return false;
}

//returns the index of which the group was found in a vector with multiple plans, -1 if not found
int8_t isGroupInPlans(const std::string& group, const std::vector<moveit::planning_interface::MoveGroupInterface::Plan>& plans){
  //returnerer indeksen gruppen finnes i planvektoren, -1 hvis den ikke finnes
  int8_t index = 0;
  for (const auto& plan : plans){
    if (isGroupInPlan(group, plan)){
      return index;
    }
    index ++;
  }
  return -1;
}

//exapnds the length of the plan to a given length with the last value in the plan
moveit::planning_interface::MoveGroupInterface::Plan expandTrajectory(moveit::planning_interface::MoveGroupInterface::Plan plan, size_t lengthOfTrajectory){
  //each plan includes  start_state_
  
  moveit::planning_interface::MoveGroupInterface::Plan newPlan = plan;

  for(size_t i = plan.trajectory_.joint_trajectory.points.size(); i < lengthOfTrajectory; i++){
    newPlan.trajectory_.joint_trajectory.points.push_back(plan.trajectory_.joint_trajectory.points.back());
  }
  return newPlan;
}

//returns a new plan where the first point in the plan is the start state of the template plan, and has the same "width" as the start state
//start state may have different number of joints than the trajectory, because a trajectory is only defined for the group while start state is for the system
moveit::planning_interface::MoveGroupInterface::Plan newPlanFromStartState(moveit::planning_interface::MoveGroupInterface::Plan templatePlan, std::string name, size_t numberOfJoints, size_t startIndex){ 


  moveit::planning_interface::MoveGroupInterface::Plan newPlan(templatePlan);

  newPlan.trajectory_.joint_trajectory.joint_names = std::vector<std::string>(templatePlan.start_state_.joint_state.name.begin() + startIndex, templatePlan.start_state_.joint_state.name.begin() + startIndex + numberOfJoints);
  newPlan.trajectory_.joint_trajectory.points[0].positions = std::vector<double>(templatePlan.start_state_.joint_state.position.begin() + startIndex, templatePlan.start_state_.joint_state.position.begin() + startIndex + numberOfJoints);
  //effort er tom slik at dette leder til segfault
  //newPlan.trajectory_.joint_trajectory.points[0].effort = std::vector<double>(templatePlan.start_state_.joint_state.effort.begin() + startIndex, templatePlan.start_state_.joint_state.effort.begin() + startIndex + numberOfJoints);
  newPlan.trajectory_.joint_trajectory.points[0].velocities = std::vector<double>(templatePlan.start_state_.joint_state.velocity.begin() + startIndex, templatePlan.start_state_.joint_state.velocity.begin() + startIndex + numberOfJoints);
  newPlan.trajectory_.joint_trajectory.points[0].time_from_start = templatePlan.trajectory_.joint_trajectory.points[0].time_from_start; //dette vil finnes første punkt [0]
  newPlan.trajectory_.joint_trajectory.points[0].accelerations = std::vector<double>(numberOfJoints, 0.1);

  //Er bare interessert i første punkt
  std::vector<trajectory_msgs::msg::JointTrajectoryPoint> firstPoint;
  firstPoint.push_back(newPlan.trajectory_.joint_trajectory.points[0]);
  newPlan.trajectory_.joint_trajectory.points = firstPoint;

  return newPlan;
}
//allows D1 + D2
builtin_interfaces::msg::Duration addDurations(builtin_interfaces::msg::Duration dur1, builtin_interfaces::msg::Duration dur2){
  builtin_interfaces::msg::Duration result;
  result.nanosec = (dur1.nanosec + dur2.nanosec)%1000000000;
  result.sec = dur1.sec + dur2.sec + std::floor((dur1.nanosec + dur2.nanosec)/1000000000);
  return result;
}
//allows D/n
builtin_interfaces::msg::Duration divideDuration(const builtin_interfaces::msg::Duration& d, float value) {
    builtin_interfaces::msg::Duration result;
    long long totalNanosec = d.sec * 1000000000 + d.nanosec; // Convert to nanoseconds
    totalNanosec = totalNanosec + totalNanosec/value; // Divide by input
    result.sec = totalNanosec / 1000000000; // Convert back to seconds and nanoseconds
    result.nanosec = totalNanosec % 1000000000;
    return result;
}

//takes a plan a and add plan b at index. Will allways insert at time = 0. (plan[time = 0] = [group_1/joint_1 pos_t=0, group_2/joint_2_pos_t=0....group_n/joint_n pos_t=0])
//addToPlan(plan, plan_for_group_2, group_2_starts_idx = 1) ->             (plan[time = 0] = [group_1/joint_1 pos_t=0, group_2/joint_2_newPos_t=0....group_n/joint_n pos_t=0]) for all t
//if planToAdd is shorter, then rest values will be filled with last planToAdd values
void addToPlan(moveit::planning_interface::MoveGroupInterface::Plan& plan,const moveit::planning_interface::MoveGroupInterface::Plan planToAdd, int startidx){ 
  //plan has a length i
  //and a width j
  //moveit::planning_interface::MoveGroupInterface::Plan results(plan);
  log("trying to add vector with length: " + std::to_string(planToAdd.trajectory_.joint_trajectory.points.size()) + " at index " + std::to_string(startidx) + " to a vector with size " + std::to_string(plan.trajectory_.joint_trajectory.points.size()), 1);

  for(int i = startidx; i < startidx + planToAdd.trajectory_.joint_trajectory.points.size(); i ++){
    int idx = findIndex(planToAdd.trajectory_.joint_trajectory.joint_names, plan.start_state_.joint_state.name); //denne sikrer at gruppen, dersom den ikke gjør det vil ikke det finnes fysiske joints å bevege
    if (idx >= 0){ //if the group exist insert, startidx is >= 0
      plan.trajectory_.joint_trajectory.points.at(i).positions.erase(plan.trajectory_.joint_trajectory.points.at(i).positions.begin() + idx, plan.trajectory_.joint_trajectory.points.at(i).positions.begin() + idx + planToAdd.trajectory_.joint_trajectory.joint_names.size());
      plan.trajectory_.joint_trajectory.points.at(i).positions.insert(plan.trajectory_.joint_trajectory.points.at(i).positions.begin() + idx, planToAdd.trajectory_.joint_trajectory.points.at(i-startidx).positions.begin(), planToAdd.trajectory_.joint_trajectory.points.at(i-startidx).positions.end());
      plan.trajectory_.joint_trajectory.points.at(i).velocities.erase(plan.trajectory_.joint_trajectory.points.at(i).velocities.begin() + idx, plan.trajectory_.joint_trajectory.points.at(i).velocities.begin() + idx + planToAdd.trajectory_.joint_trajectory.joint_names.size());
      plan.trajectory_.joint_trajectory.points.at(i).velocities.insert(plan.trajectory_.joint_trajectory.points.at(i).velocities.begin() + idx, planToAdd.trajectory_.joint_trajectory.points.at(i-startidx).velocities.begin(), planToAdd.trajectory_.joint_trajectory.points.at(i-startidx).velocities.end());
      
      //planToAdd starts at t = 0, has to start at t = t_prev_bane 
      plan.trajectory_.joint_trajectory.points.at(i).time_from_start = addDurations(planToAdd.trajectory_.joint_trajectory.points.at(i - startidx).time_from_start, plan.trajectory_.joint_trajectory.points.back().time_from_start);
    }
  }
  //uses the last point for temp storing the accumulated time
  if(startidx + planToAdd.trajectory_.joint_trajectory.points.size() != plan.trajectory_.joint_trajectory.points.size()){
    plan.trajectory_.joint_trajectory.points.back().time_from_start = addDurations(plan.trajectory_.joint_trajectory.points.back().time_from_start, planToAdd.trajectory_.joint_trajectory.points.back().time_from_start);
  }
  //return results;
}


//returns the index of which subset is found in set
int8_t findIndex(std::vector<std::string> subset, std::vector<std::string> set){
  //the order is equals such that we can iterate up
  for (uint8_t i = 0; i < set.size(); i ++){
    //finner den første matchen
    if(set.at(i) == subset.front()){ //function call - not optimal
      return i;
    }
  }
  return -1;
}


//combines states from different groups/robots and returns a new state consisting of the states from the two groups
sensor_msgs::msg::JointState concatenateStates(const std::vector<moveit::planning_interface::MoveGroupInterface::Plan>& plans){ 
  sensor_msgs::msg::JointState state(plans[0].start_state_.joint_state);
  /*
  må finne ut hvilken posisjon i joint_state de forskjellige verdiene skal
  group_names = vector<string>
  */
  for (auto const& plan : plans){
    int8_t idx = findIndex(plan.trajectory_.joint_trajectory.joint_names, plan.start_state_.joint_state.name);
    int8_t sizeOfGroup = plan.trajectory_.joint_trajectory.joint_names.size();
    if (idx >= 0){
      state.position.erase(state.position.begin() + idx, state.position.begin() + idx + sizeOfGroup);
      state.position.insert(state.position.begin() + idx, plan.trajectory_.joint_trajectory.points.back().positions.begin(), plan.trajectory_.joint_trajectory.points.back().positions.end());
    }
  }
  return state;
}

//returns a plan with minimum info
moveit::planning_interface::MoveGroupInterface::Plan stupidPlanCreator(const std::string& group, const std::shared_ptr<rclcpp::Node> node){
  moveit::planning_interface::MoveGroupInterface::Plan newPlan;
  using moveit::planning_interface::MoveGroupInterface;
  auto move_group_interface = MoveGroupInterface(node, group);

  sensor_msgs::msg::JointState tempState;
  tempState.name = move_group_interface.getJointNames();
  for(auto const& joint : tempState.name){
    tempState.position.push_back(0.0);
  }
  move_group_interface.setJointValueTarget(tempState); //plan for reaching "home (zeros)" to fill out neccessary information
  move_group_interface.plan(newPlan);
  //std::cout << newPlan.start_state_.joint_state.header.frame_id << std::endl; //is correct frame?? yes

  return newPlan;
}



std::vector<moveit::planning_interface::MoveGroupInterface::Plan> findPlans(const std::vector<geometry_msgs::msg::Pose>& points, const std::string& group, const std::shared_ptr<rclcpp::Node> node){
  std::vector<moveit::planning_interface::MoveGroupInterface::Plan> plans;
  //Hentet fra moveit cpp tutorial
  using moveit::planning_interface::MoveGroupInterface;
  auto move_group_interface = MoveGroupInterface(node, group);
  move_group_interface.setPlanningTime(20); //dette er dumt

  //log(move_group_interface.getEndEffectorLink());
  move_group_interface.setMaxVelocityScalingFactor(0.02); //setter skalering -> nærmere 0 == tregere

  //the first point has the robot current startvalues
  for (auto const& point : points){
    if (point == points.front()){ //if first plans is empty 
      log("setting target pose", 1);
      move_group_interface.setPoseTarget(point);
      log("creating plan", 1);
      auto const [success, plan] = [&move_group_interface]{
        moveit::planning_interface::MoveGroupInterface::Plan msg;
        auto const ok = static_cast<bool>(move_group_interface.plan(msg));
        return std::make_pair(ok, msg);
      }();
      if (success){
        plans.push_back(plan);
      }
      else{
        log("PLANNING FAILED! EXITING");
        exit(-1);
      } 
    }
    else{//else calculate from previous position
      moveit_msgs::msg::RobotState startState;
      sensor_msgs::msg::JointState temp = concatenateStates(std::vector<moveit::planning_interface::MoveGroupInterface::Plan>{plans.back()});
      startState.joint_state.name = temp.name;
      startState.joint_state.position = temp.position;
      startState.joint_state.velocity = temp.velocity;
      startState.joint_state.effort = temp.effort;

      log("setting start state to previous end state", 1);
      move_group_interface.setStartState(startState);
      log("setting target pose", 1);
      move_group_interface.setPoseTarget(point);
      log("creating plan", 1);
      auto const [success, plan] = [&move_group_interface]{
        moveit::planning_interface::MoveGroupInterface::Plan msg;
        auto const ok = static_cast<bool>(move_group_interface.plan(msg));
        return std::make_pair(ok, msg);
      }();
      if (success){
        plans.push_back(plan);
      }
      else{
        log("PLANNING FAILED! EXITING");
        exit(-1);
      } 
    }
  }
  return plans;
}




std::vector<geometry_msgs::msg::Pose> createStraightPathPoints(std::vector<double> xyz_start, std::vector<double> xyz_stop, std::vector<double> xyzw_orientation, int num_points){
  std::vector<geometry_msgs::msg::Pose> points;

    //std::cout << xyz_start.size();
  assert(xyzw_orientation.size() == 4); //må være verdier for alle 
  //assert((xyz_start.size() == xyz_stop.size()) == 3); //må være verdier for xyz

  auto const target_pose = [](double x, double y, double z, std::vector<double> xyzw_orientation){
    geometry_msgs::msg::Pose msg;
    msg.position.x = x;
    msg.position.y = y;
    msg.position.z = z;
    msg.orientation.x = xyzw_orientation.at(0);
    msg.orientation.y = xyzw_orientation.at(1);
    msg.orientation.z = xyzw_orientation.at(2);
    msg.orientation.w = xyzw_orientation.at(3);
    return msg;
  };

  double dx = (xyz_stop.at(0) - xyz_start.at(0))/num_points;
  double dy = (xyz_stop.at(1) - xyz_start.at(1))/num_points;
  double dz = (xyz_stop.at(2) - xyz_start.at(2))/num_points;

  for(int i = 0; i < num_points - 1; i++){
    points.push_back(target_pose(xyz_start.at(0) + dx * i, xyz_start.at(1) + dy * i, xyz_start.at(2) + dz * i, xyzw_orientation));
  }
  points.push_back(target_pose(xyz_stop.at(0), xyz_stop.at(1), xyz_stop.at(2), xyzw_orientation));
  return points;
}


moveit_msgs::msg::JointConstraint createJointConstrain(std::string joint_name, double lower_limit, double upper_limit){
    moveit_msgs::msg::JointConstraint result;
    result.joint_name = joint_name;
    result.tolerance_below = lower_limit;
    result.tolerance_above = upper_limit;
    result.weight = 0.3;
    //må potensielt ha posisjon også
    return result;

}


moveit_msgs::msg::Constraints createJointConstrains(std::vector<std::string> joint_names, std::vector<double> lower_constrains, std::vector<double> upper_constrains){
    moveit_msgs::msg::Constraints results;
    for(int i = 0; i < joint_names.size(); i ++){
        results.joint_constraints.push_back(createJointConstrain(joint_names.at(i), lower_constrains.at(i), upper_constrains.at(i)));
    }
    return results;
}



geometry_msgs::msg::Pose createPose(double x, double y, double z, double ox, double oy, double oz, double ow){
  geometry_msgs::msg::Pose result;
  result.orientation.set__w(ow);
  result.orientation.set__x(ox);
  result.orientation.set__y(oy);
  result.orientation.set__z(oz);
  result.position.set__x(x);
  result.position.set__y(y);
  result.position.set__z(z);
  
  return result;
}




bool posesEqual(const geometry_msgs::msg::Pose& pose1, const geometry_msgs::msg::Pose& pose2, double tolerance)
{
    // Compare position components within tolerance
    if (std::abs(pose1.position.x - pose2.position.x) > tolerance ||
        std::abs(pose1.position.y - pose2.position.y) > tolerance ||
        std::abs(pose1.position.z - pose2.position.z) > tolerance)
    {
        return false;
    }

    // Compare orientation components within tolerance
    
    if (std::abs(pose1.orientation.x - pose2.orientation.x) > tolerance ||
        std::abs(pose1.orientation.y - pose2.orientation.y) > tolerance ||
        std::abs(pose1.orientation.z - pose2.orientation.z) > tolerance ||
        std::abs(pose1.orientation.w - pose2.orientation.w) > tolerance)
    {
        return false;
    }
    
    
    // If we've made it this far, the poses are equal within tolerance
    return true;
}

moveit::core::RobotState getRobotStateFromMoveGroupInterface(const moveit::planning_interface::MoveGroupInterface& mgi){
    moveit::core::RobotModelConstPtr robot_model = mgi.getRobotModel();
    moveit::core::RobotStatePtr robot_stateptr(new moveit::core::RobotState(robot_model)); //is not used?
    moveit::core::RobotState robot_state(robot_model);
    return robot_state;
}



void restructure_vectors(std::vector<std::string>& a,  std::vector<std::string>& b, std::vector<double>& y) {
    // Create a mapping between elements in "a" and their indices in "x"
    std::unordered_map<std::string, size_t> index_map;
    for (size_t i = 0; i < a.size(); ++i) {
        index_map[a[i]] = i; 
    }
    
    // Create new vectors "new_b" and "new_y" with the same order as "a"
    std::vector<std::string> new_b(a.size());
    std::vector<double> new_y(a.size());
    for (size_t i = 0; i < b.size(); ++i) {
        // Find the corresponding index in "x" using the mapping
        auto it = index_map.find(b[i]);
        if (it != index_map.end()) {
            size_t index = it->second;
            // Add the element to the new vectors
            new_b[index] = b[i];
            new_y[index] = y[i];
        }
    }
    
    // Replace "b" and "y" with the new vectors
    b = std::move(new_b);
    y = std::move(new_y);
}
