#!/usr/bin/env python3
import numpy as np
import rospy
from geometry_msgs.msg import Twist,PoseStamped
from nav_msgs.msg import Odometry
import tf2_ros
import tf
from gazebo_msgs.msg import ModelStates
from move_base_msgs.msg import MoveBaseGoal

class Controller:
    def __init__(self):
        self.x = np.arange(-10, 10.5, 0.25)
        self.y = np.arange(-10, 10.5, 0.25)

        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.zero_vel = Twist()
        self.zero_vel.linear.x = 0
        self.zero_vel.angular.z = 0

        self.final_goal = [0, 0,0]
        self.delx = np.zeros_like(self.X)
        self.dely = np.zeros_like(self.Y)
        self.s = 1
        self.r = 0.2

        self.current_x=None
        self.current_y=None
        self.current_theta=None
        self.goal_reached = True
        self.orient_correct = True
        for i in range(len(self.x)):
            for j in range(len(self.y)):
                d = np.sqrt((self.final_goal[0] - self.Y[i][j]) ** 2 + (self.final_goal[1] - self.X[i][j]) ** 2)
                theta = np.arctan2(self.Y[i][j], self.X[i][j])

                if d < self.r:
                    self.delx[i][j] = 0
                    self.dely[i][j] = 0
                elif d > self.r + self.s:
                    self.delx[i][j] = int(10 * self.s * d)
                    self.dely[i][j] = int(10 * self.s * d)
                else:
                    self.delx[i][j] = int(10 * (d))
                    self.dely[i][j] = int(10 * (d))
        self.goal_costmap = self.delx + self.dely
        self.costmap=self.goal_costmap
        #print(self.costmap)



    def pid_controller(self, goal_x, goal_y, current_x, current_y, current_theta):
        linear_error = np.sqrt((goal_x - current_x) ** 2 + (goal_y - current_y) ** 2)
        angular_error = np.arctan2(goal_y - current_y, goal_x - current_x) - current_theta
        if angular_error > np.pi:
            angular_error -= 2 * np.pi
        elif angular_error < -np.pi:
            angular_error += 2 * np.pi

        kp_linear = 1*(np.pi - abs(angular_error))/np.pi
        ki_linear = 0.00001
        kp_angular = 1

        self.linear_integral += linear_error

        linear_velocity = kp_linear * linear_error + ki_linear * self.linear_integral
        angular_velocity = kp_angular * angular_error

        cmd_vel = Twist()
        cmd_vel.linear.x = linear_velocity
        cmd_vel.angular.z = angular_velocity

        return cmd_vel
    def update_costmap(self,obstacles):
        delx = np.zeros_like(self.X)
        dely = np.zeros_like(self.Y)
        obstacle_costs = np.zeros_like(self.delx)
        for obstacle in obstacles:
            for i in range(len(self.x)):
                for j in range(len(self.y)):
                    
                    
                    d = np.sqrt((obstacle[0] - self.Y[i][j]) ** 2 + (obstacle[1] - self.X[i][j]) ** 2)


                    # using the Formula of avoiding obstacle
                    if d< 0.6:
                        delx[i][j] = 200
                        dely[i][j] = 200
                    elif d>1:
                        delx[i][j] = 0
                        dely[i][j] = 0
                    else:
                        delx[i][j] = int(100 *(1.2-d))
                        dely[i][j] = int(100 * (1.2-d))
            obstacle_costs=obstacle_costs+delx+dely
        self.costmap=self.goal_costmap+obstacle_costs

    def goal_set(self, current_x, current_y):
        for i in range(len(self.x)):
            if abs(current_x - self.x[i]) <= 0.13:
                xpos_map = self.X[0][i]
                x_index = i
            if abs(current_y - self.y[i]) <= 0.13:
                ypos_map = self.Y[i][0]
                y_index = i
        min_cost = self.costmap[x_index][y_index]
        #print(xpos_map, ypos_map)
        for i in range(x_index - 3, x_index + 4):
            for j in range(y_index - 3, y_index + 4):
                if min_cost > self.costmap[i][j]:
                    min_cost = self.costmap[i][j]
                    xpos_map = self.X[0][i]
                    ypos_map = self.Y[j][0]
        #print("------", xpos_map, ypos_map, min_cost)
        return xpos_map, ypos_map

    def model_states_callback(self, data):
        self.current_x=data.pose[2].position.x
        self.current_y=data.pose[2].position.y
        quaternion = (
            data.pose[2].orientation.x,
            data.pose[2].orientation.y,
            data.pose[2].orientation.z,
            data.pose[2].orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.current_theta= euler[2]
        self.update_costmap([[data.pose[i].position.x,data.pose[i].position.y] for i in [1,3,4]])
       
    def goal_callback(self, data):
        quaternion = (
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.final_goal[2] = euler[2]
        self.final_goal[0] = data.pose.position.x   #+1.2*np.sin(self.final_goal[2]+np.pi/2)
        self.final_goal[1] = data.pose.position.y   #-1.2*np.cos(self.final_goal[2]+np.pi/2)
        #print("--------------------------------",self.final_goal)
        for i in range(len(self.x)):
            for j in range(len(self.y)):
                d = np.sqrt((self.final_goal[0] - self.Y[i][j]) ** 2 + (self.final_goal[1] - self.X[i][j]) ** 2)
                theta = np.arctan2(self.Y[i][j], self.X[i][j])

                if d < self.r:
                    self.delx[i][j] = 0
                    self.dely[i][j] = 0
                elif d > self.r + self.s:
                    self.delx[i][j] = int(20 * self.s * d)
                    self.dely[i][j] = int(20 * self.s * d)
                else:
                    self.delx[i][j] = int(20 * (d))+3
                    self.dely[i][j] = int(20 * (d))+3
        self.goal_costmap = self.delx + self.dely
        self.costmap=self.goal_costmap
        #print(self.costmap)
        self.goal_reached = False
        self.orient_correct = False
        self.linear_integral=0

    def controller(self):
        rospy.init_node('controller_node2', anonymous=False)
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)

        cmd_vel_pub = rospy.Publisher('volta_2/nav/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback,queue_size=1)
        rospy.Subscriber('/bot2/goal', PoseStamped, self.goal_callback,queue_size=1)
        rospy.sleep(1)

        rate = rospy.Rate(15) 
        while self.current_x is None:
            pass

        while not rospy.is_shutdown():
            if not self.goal_reached:
                try:
                    # trans = tfBuffer.lookup_transform('world', 'volta_0/base_link', rospy.Time(), rospy.Duration(1.5))
                    # self.current_x = trans.transform.translation.x
                    # self.current_y = trans.transform.translation.y
                    # quaternion = (
                    #     trans.transform.rotation.x,
                    #     trans.transform.rotation.y,
                    #     trans.transform.rotation.z,
                    #     trans.transform.rotation.w
                    # )
                    # euler = tf.transformations.euler_from_quaternion(quaternion)

                    # current_theta = euler[2]
                    #print(self.current_x, self.current_y, self.current_theta)
                    goal_x, goal_y = self.goal_set(self.current_x, self.current_y)

                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.logwarn('Failed to get transform')

                cmd_vel = self.pid_controller(goal_x, goal_y, self.current_x, self.current_y, self.current_theta)

                cmd_vel_pub.publish(cmd_vel)
                if np.sqrt((self.final_goal[0] - self.current_x) ** 2 + (self.final_goal[1] - self.current_y) ** 2) < 0.2:
                    self.goal_reached=True
                    cmd_vel_pub.publish(self.zero_vel)
                    #print("goal_reached")
            
            if self.goal_reached and not self.orient_correct:
                cmd_vel = Twist()
                angular_error = self.final_goal[2] - self.current_theta
                if angular_error > np.pi:
                    angular_error -= 2 * np.pi
                elif angular_error < -np.pi:
                    angular_error += 2 * np.pi
                cmd_vel.angular.z = angular_error
                cmd_vel_pub.publish(cmd_vel)
                if abs(self.final_goal[2]-self.current_theta) < 0.1:
                    self.orient_correct=True
                    cmd_vel_pub.publish(self.zero_vel)
                    #print("orient_corrected")

            rate.sleep()
        
        


if __name__ == '__main__':
    controller = Controller()
    controller.controller()
