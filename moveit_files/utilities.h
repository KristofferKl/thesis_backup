
#include <cstdio>
#include <memory>

#include "iostream"

#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include <moveit_msgs/msg/robot_state.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

//#include <moveit/trajectory_processing/limit_cartesian_speed.h>

#include "limit_cartesian_speed/limit_cartesian_speed.h"


int log(std::string data, const int verbose = 0);
std::vector<std::pair<std::string, std::pair<size_t, size_t>>> findNumberOfMatches(const std::vector<std::string>& keywords, const std::vector<std::string>& dataset);
int printTrajectory(moveit::planning_interface::MoveGroupInterface::Plan plan);
bool compareByIndex(const std::pair<std::string, std::pair<size_t, size_t>>& a, const std::pair<std::string, std::pair<size_t, size_t>>& b);
bool isGroupInPlan(const std::string& group, const moveit::planning_interface::MoveGroupInterface::Plan& plan);
int8_t isGroupInPlans(const std::string& group, const std::vector<moveit::planning_interface::MoveGroupInterface::Plan>& plans);
moveit::planning_interface::MoveGroupInterface::Plan expandTrajectory(moveit::planning_interface::MoveGroupInterface::Plan plan, size_t lengthOfTrajectory);
moveit::planning_interface::MoveGroupInterface::Plan newPlanFromStartState(moveit::planning_interface::MoveGroupInterface::Plan templatePlan, std::string name, size_t numberOfJoints, size_t startIndex);
int8_t findIndex(std::vector<std::string> subset, std::vector<std::string> set);
sensor_msgs::msg::JointState concatenateStates(const std::vector<moveit::planning_interface::MoveGroupInterface::Plan>& plans);
std::vector<moveit::planning_interface::MoveGroupInterface::Plan> findPlans(const std::vector<geometry_msgs::msg::Pose>& points, const std::string& group, const std::shared_ptr<rclcpp::Node> node);
moveit_msgs::msg::JointConstraint createJointConstrain(std::string joint_name, double lower_limit, double upper_limit);
moveit_msgs::msg::Constraints createJointConstrains(std::vector<std::string> joint_names, std::vector<double> lower_constrains, std::vector<double> upper_constrains);
std::vector<geometry_msgs::msg::Pose> createStraightPathPoints(std::vector<double> xyz_start, std::vector<double> xyz_stop, std::vector<double> xyzw_orientation, int num_points);
void addToPlan(moveit::planning_interface::MoveGroupInterface::Plan& plan, const moveit::planning_interface::MoveGroupInterface::Plan planToAdd, int startidx=0);
geometry_msgs::msg::Pose createPose(double x, double y, double z, double ox, double oy, double oz, double ow);
moveit::planning_interface::MoveGroupInterface::Plan stupidPlanCreator(const std::string& group, const std::shared_ptr<rclcpp::Node> node);
builtin_interfaces::msg::Duration addDurations(builtin_interfaces::msg::Duration dur1, builtin_interfaces::msg::Duration dur2);
builtin_interfaces::msg::Duration divideDuration(const builtin_interfaces::msg::Duration& d, float value); 
bool posesEqual(const geometry_msgs::msg::Pose& pose1, const geometry_msgs::msg::Pose& pose2, double tolerance);
moveit::core::RobotState getRobotStateFromMoveGroupInterface(const moveit::planning_interface::MoveGroupInterface& mgi);
void restructure_vectors(std::vector<std::string>& a, std::vector<std::string>& b, std::vector<double>& y);