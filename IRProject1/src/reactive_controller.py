#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Since we are using python 2 we need to make sure we include utf-8 encoding

import rospy
import random
import math
import time

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from kobuki_msgs.msg import BumperEvent
from tf.transformations import euler_from_quaternion

# Conversion
FT_TO_M = 0.3048

# Use a class to be able to reference an object for ease of use
class ReactiveController(object):
    def __init__(self):
        # define the node that is this controller
        rospy.init_node('reactive_controller', anonymous=False)

        # Parameters for robot movement either get from defined paramter or use default value here
        # These are in meters per second or in radians per second
        self.forward_speed       = rospy.get_param('~forward_speed', 0.18)
        self.turn_speed          = rospy.get_param('~turn_speed', 0.6)
        self.avoid_range_m       = rospy.get_param('~avoid_range_m', 1.0*FT_TO_M) 
        self.symmetric_eps_m     = rospy.get_param('~symmetric_eps_m', 0.05)
        self.random_turn_rad     = rospy.get_param('~random_turn_rad', math.radians(15.0)) 
        self.escape_spread_rad   = rospy.get_param('~escape_spread_rad', math.radians(30.0))
        self.teleop_timeout      = rospy.get_param('~teleop_timeout', 0.4)         
        self.random_stride_m     = rospy.get_param('~random_stride_m', 1.0*FT_TO_M)
        # Use teleop instead of cmd_vel
        self.teleop_topic        = rospy.get_param('~teleop_topic', '/teleop_cmd') 
        self.log_every           = rospy.get_param('~log_every', 10)

        # Creates a publisher that sends commands to the turtlebot
        self.cmd_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
        # subscribes to events on the turtlebots bumper 
        rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self.bump_cb)
        # Subscribes to the turtlebots odometer events
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        # Subscribes to the turtlebots scan events
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        # Subscribes to teleop commands
        rospy.Subscriber(self.teleop_topic, Twist, self.teleop_cb)

        # This is the base variable state
        self.halt = False
        self.mode = 'FORWARD'
        self.last_teleop = None
        self.last_teleop_time = 0.0

        # the turn angle
        self.angle = 0.0
        self.turn_target = None
        self.last_odom = None
        self.dist_since_random = 0.0

        self.has_scan = False
        self.d_front = float('inf')
        self.d_left  = float('inf')
        self.d_right = float('inf')


        self._dbg_i = 0
        # Use a 20 for the loop rate
        self.rate = rospy.Rate(20) 

    # If the bumper is pressed stop the turtlebot
    def bump_cb(self, msg):
        if msg.state == BumperEvent.PRESSED:
            self.halt = True
            self.mode = 'HALT'

    # update the teleop message and time
    def teleop_cb(self, msg):
        self.last_teleop = msg
        self.last_teleop_time = time.time()

    # Track the angle and distance traveled
    def odom_cb(self, msg):
        # Extract the angle from the turtlebot
        q = msg.pose.pose.orientation
        (_, _, angle) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.angle = angle

        # Track the odomotors forward movement 
        if self.last_odom is not None:
            p0 = self.last_odom.pose.pose.position
            p1 = msg.pose.pose.position
            # Get displacement
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            step = math.sqrt(dx*dx + dy*dy)
            # Track only if turtlebot is actively controlling its movement
            if self.mode in ['FORWARD', 'AVOID'] and not self.teleop_active():
                self.dist_since_random += step
        # Update the odomotor message
        self.last_odom = msg

    # Scan surroundings and analyze scans
    def scan_cb(self, scan):
        # If no distances in scan return 
        n = len(scan.ranges)
        if n == 0:
            self.has_scan = False
            return

        # get center/ straight ahead
        mid = n // 2  

        # convert angles to sector lengths
        def bins_for_deg(deg):
            return max(1, int(abs(math.radians(deg)) / scan.angle_increment))

        # get angles for front side and offset 
        w_front = bins_for_deg(10.0)  
        w_side  = bins_for_deg(15.0)  
        off30   = bins_for_deg(30.0)   


        def sector_min_by_index(center_idx, half_width):
            # calculate low and high ends of sectors
            lo = max(0, center_idx - half_width)
            hi = min(n - 1, center_idx + half_width)
            # add distances in the range
            vals = []
            for r in scan.ranges[lo:hi+1]:
                if not (math.isinf(r) or math.isnan(r)):
                    vals.append(r)
            # return the closest obstacle in sector
            return min(vals) if vals else float('inf')

        # find closest obstacle in each sector
        self.d_front = sector_min_by_index(mid, w_front)
        self.d_left  = sector_min_by_index(min(n-1, mid + off30), w_side)
        self.d_right = sector_min_by_index(max(0,    mid - off30), w_side)
        self.has_scan = True

    # Check if teleop is still active 
    def teleop_active(self):
        return (self.last_teleop is not None) and ((time.time() - self.last_teleop_time) <= self.teleop_timeout)

    # Get small angle difference
    def ang_diff(self, a, b):
        d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
        return d

    # Get a desired angle and set the turn target
    def set_turn_target(self, delta):
        self.turn_target = (self.angle + delta + math.pi) % (2.0 * math.pi) - math.pi

    # Check if turtlebot is at target
    def at_target(self, tol=0.05):
        return (self.turn_target is not None) and (abs(self.ang_diff(self.turn_target, self.angle)) < tol)

    
    def choose_behavior(self):
        # Check if turtlebot bumper is pressed and in halt mode
        if self.halt:
            self.mode = 'HALT'
            return Twist() 

        # Check if teleop is active
        if self.teleop_active():
            self.mode = 'TELEOP'
            return self.last_teleop

        #  Check if scan is due
        if self.has_scan:
            # check if both sides are nearly equal
            left_close  = self.d_left  < self.avoid_range_m
            right_close = self.d_right < self.avoid_range_m
            symmetric   = left_close and right_close and (abs(self.d_left - self.d_right) < self.symmetric_eps_m)

            # Check if in escape more or should be in escape mode
            if symmetric or self.mode == 'ESCAPE':
                if self.mode != 'ESCAPE':
                    # Calculate escape and turn target 
                    jitter = random.uniform(-self.escape_spread_rad/2.0, self.escape_spread_rad/2.0)
                    self.set_turn_target(math.pi + jitter)
                    self.mode = 'ESCAPE'
                cmd = Twist()
                cmd.angular.z = self.turn_speed * (1.0 if self.ang_diff(self.turn_target, self.angle) > 0.0 else -1.0)
                # If at target go forward
                if self.at_target():
                    self.mode = 'FORWARD'
                    self.turn_target = None
                return cmd

            # Check if obstacle is too close
            if min(self.d_left, self.d_front, self.d_right) < self.avoid_range_m:
                # Go into avoid mode
                self.mode = 'AVOID'
                cmd = Twist()
                cmd.linear.x = 0.08  

                # Turn away from closest obstacle 
                closest = min((self.d_left, 'L'), (self.d_front, 'F'), (self.d_right, 'R'), key=lambda t: t[0])[1]
                if closest == 'L':
                    cmd.angular.z = -self.turn_speed
                elif closest == 'R':
                    cmd.angular.z = +self.turn_speed
                else:
                    # front is closest -> choose the side with larger clearance
                    if self.d_left > self.d_right:
                        cmd.angular.z = +self.turn_speed
                    else:
                        cmd.angular.z = -self.turn_speed
                return cmd

        # Choose a random side to turn to
        if self.mode == 'RANDTURN':
            cmd = Twist()
            cmd.angular.z = self.turn_speed * (1.0 if self.ang_diff(self.turn_target, self.angle) > 0.0 else -1.0)
            if self.at_target():
                self.mode = 'FORWARD'
                self.turn_target = None
            return cmd
        else:
            if self.dist_since_random >= self.random_stride_m:
                delta = random.uniform(-self.random_turn_rad, +self.random_turn_rad)
                self.set_turn_target(delta)
                self.mode = 'RANDTURN'
                self.dist_since_random = 0.0
                return Twist()  # will start turning next tick

        # Go forward
        self.mode = 'FORWARD'
        cmd = Twist()
        cmd.linear.x = self.forward_speed
        return cmd

    # While turtlebot is active run loop, choose behaviors, publish commands
    # to turtebot, and self debug 
    def spin(self):
        while not rospy.is_shutdown():
            cmd = self.choose_behavior()
            self.cmd_pub.publish(cmd)

            self._dbg_i += 1
            if self.log_every > 0 and (self._dbg_i % self.log_every == 0):
                rospy.loginfo("mode=%s dl=%.2f df=%.2f dr=%.2f",
                              self.mode, self.d_left, self.d_front, self.d_right)

            self.rate.sleep()


if __name__ == '__main__':
    # Run reactive contoller code
    ReactiveController().spin()
