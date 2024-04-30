#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates
from math import sqrt
import tf
import numpy as np




class GoalAssigner:
    def __init__(self):
        self.current_poses = {1:None, 2:None, 3:None, 4:None}
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.model_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_callback,queue_size=1)
        self.goal_pubs = {
            1:rospy.Publisher('/bot1/goal', PoseStamped, queue_size=10),
            2:rospy.Publisher('/bot2/goal', PoseStamped, queue_size=10),
            3:rospy.Publisher('/bot3/goal', PoseStamped, queue_size=10),
            4:rospy.Publisher('/bot4/goal', PoseStamped, queue_size=10)
        }
        

    def goal_callback(self, msg):
        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        orient = euler[2]
        goal_x=msg.pose.position.x
        goal_y=msg.pose.position.y
        # Assign goals to bots based on current pose and least possible distance
        goal_poses=[[goal_x-1.2*np.sin(orient),goal_y+1.2*np.cos(orient)],[goal_x+1.2*np.sin(orient),goal_y-1.2*np.cos(orient)],[goal_x-1.2*np.sin(orient+np.pi/2),goal_y+1.2*np.cos(orient+np.pi/2)],[goal_x+1.2*np.sin(orient+np.pi/2),goal_y-1.2*np.cos(orient+np.pi/2)]]
        if self.current_poses is not None:
            # Calculate distances from current pose to each bot
            min_distance=10000000000
            indices=[1,2,3,4]
            goal_index=indices
            for i in indices:
                for j in [index for index in indices if index!=i]:
                    for k in [index for index in indices if index!=i and index!=j]:
                            
                            l=[index for index in indices if index!=i and index!=j and index!=k][0]
                            #print(l)
                            distance1 = sqrt((self.current_poses[i][0] - goal_poses[0][0]) ** 2 +(self.current_poses[i][1] - goal_poses[0][1]) ** 2)
                            distance2 = sqrt((self.current_poses[j][0] - goal_poses[1][0]) ** 2 +(self.current_poses[j][1] - goal_poses[1][1]) ** 2)
                            distance3 = sqrt((self.current_poses[k][0] - goal_poses[2][0]) ** 2 +(self.current_poses[k][1] - goal_poses[2][1]) ** 2)
                            distance4 = sqrt((self.current_poses[l][0] - goal_poses[3][0]) ** 2 +(self.current_poses[l][1] - goal_poses[3][1]) ** 2)
                            # if min_distance > (dist:=distance1+distance2+distance3+distance4):
                            #     min_distance=dist
                            if abs(distance1-distance2)<0.3 and abs(distance3-distance4)<0.3 and abs(distance4-distance1)<0.3 and abs(distance1-distance3)<0.3 and abs(distance2-distance4)<0.3 and abs(distance2-distance3)<0.3:
                                goal_index=[i,j,k,l]
            #print(goal_index)
            for i in range(4):
                msg.header.stamp = rospy.Time.now()
                msg.pose.position.x=goal_poses[i][0]
                msg.pose.position.y=goal_poses[i][1]
                self.goal_pubs[goal_index[i]].publish(msg)



    def model_callback(self, msg):
        for i in range(1,5):
            self.current_poses[i]=[msg.pose[i].position.x,msg.pose[i].position.y]

if __name__ == '__main__':
    rospy.init_node('goal_assigner')
    goal_assigner = GoalAssigner()
    rospy.spin()