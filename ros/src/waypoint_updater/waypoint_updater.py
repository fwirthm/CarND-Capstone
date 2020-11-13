#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
#LOOKAHEAD_WPS = 50

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
	rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
	self.stopline_wp_idx = -1
	
	self.decel_limit = rospy.get_param('~decel_limit', -5)

        self.loop()
        
    def loop(self):
        #main loop of the waypoint updater
        #ensures that the waypoint updater runs at 30Hz - as fast as the waypoint follower
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree:
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()
    
    def get_closest_waypoint_idx(self):
        #calculates the closest waypoints index based on all waypoints and the vehicles position
        #x = self.pose.pose.position.x
        #y = self.pose.pose.position.y
        #closest_waypoint_idx = self.waypoint_tree.query([x, y], 1)[1]
        closest_waypoint_idx = self.get_closest_wp_idx()
        
        #extract the coordinates of the closest waypoint and its previous point
        closest_coord = self.waypoints_2d[closest_waypoint_idx]
        prev_coord = self.waypoints_2d[closest_waypoint_idx-1]
        
        #check that the vehicle is between the two positions, 
        #than the closest point lies ahead of the vehicle
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([self.pose.pose.position.x, self.pose.pose.position.y])
        
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        #if both points are behind the vehicle we need to take the next point
        # the modulo operator ensures that the begining of the new round is handled as desired
        if (val>0):
            closest_waypoint_idx = (closest_waypoint_idx+1) % len(self.waypoints_2d)
        return(closest_waypoint_idx)
    
    def publish_waypoints(self, closest_waypoint_idx):
        #publishes the list of the next waypoints
        lane = self.generate_lane()
        self.final_waypoints_pub.publish(lane)
        
    def generate_lane(self):
    	#generates a new lane object which can be published based on traffic lights and waypoints
    	lane = Lane()
        lane.header = self.base_waypoints.header
        
        closest_idx = self.get_closest_wp_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        lane.waypoints = self.base_waypoints.waypoints[closest_idx: farthest_idx]
        
        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx>=farthest_idx):
        	pass
       	else:
       		lane.waypoints = self.decel_wps(lane.waypoints, closest_idx)
        return(lane)
        
    def decel_wps(self, waypoints, closest_idx):
    	#calculates waypoints to stop at the upcoming stop line
    	output = []
    	stop_idx = max(self.stopline_wp_idx - closest_idx - 4, 0)
    	
    	for i, wp in enumerate(waypoints):
    		p = Waypoint()
    		p.pose = wp.pose
    		
    		dist = self.distance(waypoints, i, stop_idx)
    		vel = math.sqrt(2*dist*abs(self.decel_limit))
    		if (vel<1.):
    			vel = 0.
    		p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
    		output.append(p)

    	return(output)
    
    	
    def get_closest_wp_idx(self):
    	#calculates the closest waypoints index based on all waypoints and the vehicles position
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_waypoint_idx = self.waypoint_tree.query([x, y], 1)[1]
        return(closest_waypoint_idx)
        
    def pose_cb(self, msg):
        # sets the sensed pose as the cars pose
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # safes the list of base waypoints to data structures ensuring fast search operations
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
        self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # saves the waypoint of the next stopline from the detection module
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
