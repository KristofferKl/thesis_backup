#include "utilities.h"

#include <moveit_group_planner_interfaces/msg/waypointsets.hpp>
#include <moveit_group_planner_interfaces/msg/waypoints.hpp>
#include <moveit_group_planner_interfaces/srv/execute.hpp>
#include <moveit_group_planner_interfaces/srv/plan.hpp>
#include <rclcpp/qos.hpp>


#include "fstream"

//#include <fstream> //used for storing and plotting the trajectory/velocity etc in python


using std::placeholders::_1; //used by service / subscription
using std::placeholders::_2; //used by service / subscription
using std::placeholders::_3; //used by service / subscription
using std::placeholders::_4; //used by service / subscription

//Class in main and not in .h because legacy from development.
class WaypointListener : public rclcpp::Node
{
  public:
    WaypointListener()
    : Node("waypoint_listener"), logger_(rclcpp::get_logger("waypoint_listener"))
    {
      joint_state_callback_group_     = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      ready_publisher_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

      // static const rclcpp::QoS & qos = this->rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT); //added by KK

      auto joint_state_callback_group_options_ = rclcpp::SubscriptionOptions();
      auto execute_callback_group_options_     = rclcpp::SubscriptionOptions();


      joint_state_callback_group_options_.callback_group = joint_state_callback_group_;
      joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
        // "joint_states", 10, std::bind(&WaypointListener::joint_state_callback, this, _1), joint_state_callback_group_options_); // Johans code
        "ctrl_groups/r1/joint_states",rclcpp::QoS(10).reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT), std::bind(&WaypointListener::joint_state_callback, this, _1), joint_state_callback_group_options_); //KK changed this to get position from non-standard topic (only one out of 4 chanels)

        //KK changed this to get position from non-standard topic (only one out of 4 chanels)


      plan_service_ = this->create_service<moveit_group_planner_interfaces::srv::Plan>(
        "plan_group", std::bind(&WaypointListener::plan_callback, this, std::placeholders::_1, std::placeholders::_2));

      execute_service_= this->create_service<moveit_group_planner_interfaces::srv::Execute>(
        "execute_plan",std::bind(&WaypointListener::execute_callback, this,
                                  std::placeholders::_2)); 

      ready_publisher_ = this->create_publisher<std_msgs::msg::Bool>("ready", 10);

      //alternative for inline like in execute_service and plan_service, because that did not work.
      std::function<void()> callback = std::bind(&WaypointListener::ready_publisher_callback, this, std::make_shared<std_msgs::msg::Bool>());
      ready_publisher_timer_ = this->create_wall_timer(std::chrono::milliseconds(50), callback, ready_publisher_callback_group_);
      RCLCPP_INFO_STREAM(logger_, "Node is spinning, ready to take waypoints");
    }

  private:
    moveit::planning_interface::MoveGroupInterface::Plan plan;     //For storing the trajectory for execution
    moveit::planning_interface::MoveGroupInterface* current_group; //For storing the relevant group for FK purposes
    std::string end_effector_link;                                 //storing the relevant end effector, used to find FK such that in_position can be set
    bool in_position = false;                                      //For notifying if the arm is in position to execute task - i.e in position to weld
    bool is_executing = false;                                     //If robot is executing a plan - used in update_thread_function()

    // rclcpp::QoS qos;                                      //storing the QoS profile, added by KK

    //Because this is developed around motoros2 the main group is:
    //   follow_joint_trajectory
    //that is that motoros2 listens to actions for the group follow_joint_trajectory
    //however, if the controller is able to listen to multiple action topics, this may not be a good way to define this
    const std::shared_ptr<rclcpp::Node> followJointTrajectoryNode = std::make_shared<rclcpp::Node>("follow_joint_trajectory", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
    moveit::planning_interface::MoveGroupInterface move_group_interface{followJointTrajectoryNode, "follow_joint_trajectory"}; //=MoveGroupInterface(followJointTrajectoryNode, "follow_joint_trajectory");
    moveit::core::RobotState robot_state = getRobotStateFromMoveGroupInterface(move_group_interface); //for storing the current robotstate and calculating FK for the system

    //these variables are used to store information about the current robot state - listening on /joint_states
    //the order of joints from controller may not be the same as from moveit
    std::vector<std::string> joint_names;                                 //reading the joint_names from joint_state topic
    std::vector<double> joint_positions;                                  //reading the joint_positions from joint_state topic
    std::vector<std::pair<bool, geometry_msgs::msg::Pose>> pairWaypoints; //used to store waypoint and the corresponding isJob - see update_thread_function()
    
    //multi-threading, need to lock
    std::mutex joint_position_mutex;
    std::mutex joint_names_mutex;
    std::mutex in_position_mutex;

    double end_effector_skip = 0.01;
    double jump_threshold    = 0.0;
    double end_effector_pose_tolerance = 0.05; // in meter. The tolerance of which end effector and waypoints is compared against in update_thread_function.
    int planner_time_limit   = 20; //sec, default 5


    bool allow_replanning = true;



    //________________________ Class Functions _________________________//

    //joint_states_callback - listen to joint_states and updates the private variables joint_position and joint_names
  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    this->joint_position_mutex.lock();
    this->joint_names_mutex.lock();

    this->joint_positions = msg->position;
    this->joint_names = msg->name;

    this->joint_names_mutex.unlock();
    this->joint_position_mutex.unlock();



  }

    //plan_callback - callback for listening to waypoints - calls createPlan
    void plan_callback(const std::shared_ptr<moveit_group_planner_interfaces::srv::Plan::Request>  req,
                             std::shared_ptr<moveit_group_planner_interfaces::srv::Plan::Response> res){
      RCLCPP_INFO_STREAM(this->logger_, "Waypoints recived");
      std::string group_name = req->waypoints.groupname;
      std::vector<geometry_msgs::msg::Pose> waypoints = req->waypoints.waypoints;
      float speed = (req->waypoints.speed <= 0.0) ? 100000.0 : req->waypoints.speed; //if speed is set to <= 0, set the speed to a high value, else set to given value
      RCLCPP_INFO_STREAM(this->logger_, (req->waypoints.speed < 0.0) ? "Speed not valid. No limit set" :
               ((req->waypoints.speed == 0.0) ? "No limit set" : ("Speed limit set to: " + std::to_string(speed) + "m/s")));
      

      //Asserts that the length of isJob and waypoints are the same
      //This is used to publish if job can be done
      //for example if waypoint is a part of a welding-path or not
      std::vector<bool> isJob = req->waypoints.is_job;
      if (isJob.size() not_eq waypoints.size()){
        RCLCPP_WARN_STREAM(this->logger_, "Length of isJob list is not equals length of waypoint list, sets all job to false");
        isJob = std::vector<bool>(waypoints.size(), false);
      }
      //stores in privat vector
      for (size_t i = 0; i < isJob.size(); i ++){
        pairWaypoints.push_back(std::make_pair(isJob.at(i), waypoints.at(i)));
      }

      //pass to createPlan
      double trajectory_fraction = createPlan(group_name, waypoints, speed);
      RCLCPP_INFO_STREAM(this->logger_, "Plan created, call /execute_plan service to execute");
      res->set__trajectory_fraction(static_cast<float>(trajectory_fraction));
    }

    //ready_publisher_callback - callback for publishing if robot is in position to start job or not
    void ready_publisher_callback(std::shared_ptr<std_msgs::msg::Bool> msg){
        this->in_position_mutex.lock();
        msg->data = this->in_position;
        this->ready_publisher_->publish(*msg);
        this->in_position_mutex.unlock();

    }

    //execute_callback - callback for executing planned trajectory if service is called - passes to moveit planning_interfaces
    void execute_callback(const std::shared_ptr<moveit_group_planner_interfaces::srv::Execute::Response> res){
      //using moveit::planning_interface::MoveGroupInterface;
      //auto const followJointTrajectoryNode = std::make_shared<rclcpp::Node>("follow_joint_trajectory", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
      //auto move_group_interface = MoveGroupInterface(followJointTrajectoryNode, "follow_joint_trajectory");

      //execute the plan
      //while this should optionally return true if execute is finished, we can not get current state from this framework
      //if last point of end effector is end of waypoint list == return true
      //potentional workaround = listen to followjointtrajectory/result

      //if task is successfully sent down the pipeline, plan is executing, else something went wrong
      //this could be error in motoros or setup, for example wrong namespace or bad connection


      //FK thread because .execute wait for execution and need to update fk while this is happening
      this->is_executing = true;
      std::thread update_thread = std::thread(&WaypointListener::update_thread_function, this);

      if (this->move_group_interface.execute(this->plan).SUCCESS){
        res->set__success(true);
      }
      else{
        res->set__success(false);
      }

      this->is_executing = false;
      update_thread.join();
    }



    void update_thread_function(){
    

      int timeout = this->plan.trajectory_.joint_trajectory.points.back().time_from_start.sec;
      int start_time_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count(); // convert to milliseconds

      float tolerance = this->end_effector_pose_tolerance; //temp +- 5 cm may be too loose
      if (this->joint_positions.empty()){return;} //if joint position is empty, no FK is availible, segfault : return
      for(auto pair:this->pairWaypoints){//for each waypoint
        while(true){//wait untill waypoint is reached
          std::this_thread::yield();//update positions
          if(this->is_executing == false){
            //robot not executing - either finished or something went wrong
            this->in_position = false;
            return;
          }
          int current_time_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
          if((current_time_ms - start_time_ms) > ((1 + timeout) * 1000)){//if this has been going on longer than the path is expected, break
            RCLCPP_INFO_STREAM(this->logger_, "Execution timeout in thread, exiting thread");
            this->in_position = false;
            return;
          }

          if (this->joint_names_mutex.try_lock() || this->joint_position_mutex.try_lock()){
          //because the order of values from motoros2/robotcontroller may not be the same as values from moveit - need to reorder
          //restructure_vectors is a long process such that joint_state_listener may try access variables while the function is running leading to segfault
          //if these are locked we can not read values
            restructure_vectors(this->plan.trajectory_.joint_trajectory.joint_names, this->joint_names, this->joint_positions);
            this->joint_names_mutex.unlock();    //allow for joint_state_listener to update positions
            this->joint_position_mutex.unlock(); //allow for joint_state_listener to update positions
          }
          else{
            //this is updated quickly so we can skip one iteration in the "while loop"
            continue; //wait untill mutex is availible
          }
          //calculates FK
          this->robot_state.setVariablePositions(this->joint_positions); 
          auto const fk = this->robot_state.getGlobalLinkTransform(this->end_effector_link);
          //fk is now a 4x4 Transformation matrix, while waypoints is a pose - need to convert one of them such that we can compare
          geometry_msgs::msg::Pose fk_pose = tf2::toMsg(fk);
          

          if(posesEqual(pair.second, fk_pose, tolerance)){
            if(this->in_position_mutex.try_lock()){ //this loop is way quicker than the publisher such that skipping an iteration in this is less bad than a publish
              this->in_position = pair.first; 
              this->in_position_mutex.unlock();
              break; //waypoint reached
            }
          }
        }
      }
      this->in_position = false; //Asserts that the tool is off when finished.
      return;
    }

    //createPlan - takes group_name, a vector with waypoints and end effector speed
    //NOTE: THIS CAN NOT BE USED FOR <6 DOF
    //-Will create a cartesian path for the given group [group_name]
    //-Use iterative time parametrization to manipulate the trajectory of the end effector such that the desired speed is reached
    //-Expand the trajectory such that the trajectory contains the values for each group in the system as system = [group_a, group_b, ...]
    //-Stores the plan class variable: plan
    double createPlan(std::string name, std::vector<geometry_msgs::msg::Pose> waypoints, float speed){
      using moveit::planning_interface::MoveGroupInterface;
      //makes a planning node and movegroup interface for calculating path for given group
      auto const group_node = std::make_shared<rclcpp::Node>(name, rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
      auto group_move_interface = MoveGroupInterface(group_node, name);
      group_move_interface.setPlanningTime(this->planner_time_limit); //Standard = 5 sec
      group_move_interface.allowReplanning(this->allow_replanning);

      //Since FK for the end effector only will work if the system can recognize the kinematics (i.e the link/joint pair-set from base to tip of the robot)
      //We need to store the relevant group for getting relevant FK later.
      this->current_group = &group_move_interface; //seg fault
      this->end_effector_link = group_move_interface.getEndEffectorLink();


      //contains plan for only a given group
      //stupidPlanCreator creates a plan with only some values filled, enough to work
      moveit::planning_interface::MoveGroupInterface::Plan tempPlan = stupidPlanCreator(name, group_node);

      //calculating cartesian path
      //If fraction < 1, the planner has exited before visiting all waypoints
      //this can be because of collision, orientation or out-of-reach
      moveit_msgs::msg::RobotTrajectory traj;
      double pathFraction = group_move_interface.computeCartesianPath(waypoints, this->end_effector_skip, this->jump_threshold,traj);      
      RCLCPP_INFO_STREAM(this->logger_, "Fraction of trajectory found: " << pathFraction);
      if (pathFraction < 1){
          //todo find a way to detect what kind of error
          RCLCPP_WARN_STREAM(this->logger_, "Could not compute path for the whole set of waypoints, this is likely because of point out of reach, collision or orientation not reachable");
          RCLCPP_WARN_STREAM(this->logger_, "Does orientation-values have enough decimals?");
      }

      //limits the end effector velocity with method by : Benjamin Scholz, Thies Oelerich 
      //--------------------------------------------------------------------------------//
      //The method used robot_Trajectory as input, while this code has used moveit_msgs::msg::RobotTrajectory
      //convert moveit_msgs::msg::RobotTrajectory to robot_trajectory::RobotTrajectory
      auto robot_state = getRobotStateFromMoveGroupInterface(group_move_interface);
      robot_trajectory::RobotTrajectory limitedTraj(robot_state.getRobotModel(), name);


      //sets robot_state.traj = traj
      //traj is the calculated path for the given group
      limitedTraj.setRobotTrajectoryMsg(robot_state, traj);

      //sets the speed with iterative time parametrization
      this->end_effector_link = group_move_interface.getEndEffectorLink();
      if(this->end_effector_link.empty()){RCLCPP_WARN_STREAM(this->logger_, "No end effector found for group: " << name);}
      trajectory_processing::limitMaxCartesianLinkSpeed(limitedTraj, speed, group_move_interface.getEndEffectorLink());
      // Because we need to know if the end effector 
      // is in desired position (for example if we are ready to weld)
      // we need to store the poses for each position in the trajectory


      //inserts back into a moveit_msgs::msg::RobotTrajectory
      limitedTraj.getRobotTrajectoryMsg(traj);

      //makes a new plan for the group. The above step only creates the trajectory and we need start_state etc
      moveit::planning_interface::MoveGroupInterface::Plan newPlan = newPlanFromStartState(tempPlan, "this is not used", traj.joint_trajectory.joint_names.size(), findIndex(traj.joint_trajectory.joint_names, tempPlan.start_state_.joint_state.name));
      newPlan.trajectory_ = traj;
 
      // From motoros2, because of limited memory in the controller a trajactory can not be too long
      // There are no check for if the trajectory is too long, and depending of how many points and joints in the system,
      // this limit is not well-defined.
      // for a system consisting of 15 joint, this occured at 166 points, while the developer of motoros noticed 200 points
      // Warns a warning
      int size_of_trajectory = newPlan.trajectory_.joint_trajectory.points.size();
      RCLCPP_INFO_STREAM(this->logger_, "Trajectory has " << size_of_trajectory << " points");
      if(size_of_trajectory > 150){
        RCLCPP_WARN_STREAM(this->logger_, "WARNING: Path long, may cause crash in motoros2 as controller may not have enough memory");
      }

      //The above steps only for single group and not the whole system
      //need this path into a wider path containg all the joint in system
      moveit::planning_interface::MoveGroupInterface::Plan mergedPlan = expandTrajectory(newPlanFromStartState(newPlan, "this is not used", newPlan.start_state_.joint_state.name.size(), 0), newPlan.trajectory_.joint_trajectory.points.size());

      //adds the plan from groupplan into systemplan
      addToPlan(mergedPlan, newPlan);
      //stores the plan for later execution
      this->plan = mergedPlan;


      //________Stores pos and vel data for experimental purposes__________//
      //this will not work if "zivid" not user, couts a warning
      std::cout << "Storing velocity and position data for experimental purposes - saves to user 'zivid' and may cause error if other user" << std::endl;
      std::ofstream posfile("/home/zivid/debugs/positions.txt");
      std::ofstream velfile("/home/zivid/debugs/velocities.txt");
      for(auto const& it : this->plan.trajectory_.joint_trajectory.points){
        //for each point in plan
        posfile << it.time_from_start.sec <<"."<<it.time_from_start.nanosec<<" ";
        velfile << it.time_from_start.sec <<"."<<it.time_from_start.nanosec<<" ";
        for(auto const& jointpos : it.positions){
          //for each joint
          posfile << jointpos << " ";
        }
        posfile << std::endl;
        for(auto const& jointvel : it.velocities){
          //for each joint
          velfile << jointvel << " ";
        }
        velfile << std::endl;
      }
      posfile.close();
      velfile.close();
          
      //_______________________ END _________________________________//
      return pathFraction; //returns the fraction of the planned trajectory vs desired trajectory

    }


    //initializing ros-defined classes
    rclcpp::TimerBase::SharedPtr ready_publisher_timer_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;//??
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr ready_publisher_;
    rclcpp::Service<moveit_group_planner_interfaces::srv::Plan>::SharedPtr plan_service_;
    rclcpp::Service<moveit_group_planner_interfaces::srv::Execute>::SharedPtr execute_service_;
    rclcpp::Logger logger_;
    // rclcpp::QoS qos_;   

    //used for multi-threading callbacks
    rclcpp::CallbackGroup::SharedPtr joint_state_callback_group_;
    rclcpp::CallbackGroup::SharedPtr execute_callback_group_;
    rclcpp::CallbackGroup::SharedPtr ready_publisher_callback_group_;



};





int main(int argc, char * argv[])
{
  //std::cout << argv[1]; could use this as global group name (/follow_joint_trajectory)
  rclcpp::init(argc, argv);
  //need executor as we require multiple callbacks at the same time
  //rclcpp::spin(std::make_shared<WaypointListener>()); //single thread

  rclcpp::executors::MultiThreadedExecutor executor;                  //creates executor
  auto waypoint_listener_node = std::make_shared<WaypointListener>(); //create node
  executor.add_node(waypoint_listener_node);                          //add node to executor
  executor.spin();                                                    //spins the executor - runs the program
  
  rclcpp::shutdown();
  return 0;
}