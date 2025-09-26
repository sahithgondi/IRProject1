#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reactive controller for TurtleBot (ROS Melodic, Python 2).
Priority (high -> low):
  1) HALT on bumper press (latched)
  2) TELEOP override (recent Twist from teleop topic)
  3) ESCAPE: symmetric close obstacles (< 1 ft on both flanks, nearly equal)
  4) AVOID: asymmetric close obstacle (< 1 ft on any front sector)
  5) RANDTURN: small random heading change every 1 ft traveled
  6) FORWARD: default cruise

Notes:
- 1 ft = 0.3048 m (FT_TO_M)
- Subscribes: /mobile_base/events/bumper, /odom, /scan, ~teleop_topic (default /teleop_cmd)
- Publishes:  /mobile_base/commands/velocity (geometry_msgs/Twist)
- Requires a /scan topic (LiDAR, or depth->scan bridge).
"""

import rospy
import random
import math
import time

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from kobuki_msgs.msg import BumperEvent
from tf.transformations import euler_from_quaternion

FT_TO_M = 0.3048


class ReactiveController(object):
    def __init__(self):
        rospy.init_node('reactive_controller', anonymous=False)

        # ----------------- Tunable params -----------------
        self.forward_speed       = rospy.get_param('~forward_speed', 0.18)         # m/s
        self.turn_speed          = rospy.get_param('~turn_speed', 0.6)             # rad/s
        self.avoid_range_m       = rospy.get_param('~avoid_range_m', 1.0*FT_TO_M)   # 1 ft
        self.symmetric_eps_m     = rospy.get_param('~symmetric_eps_m', 0.05)       # how close L vs R to count as symmetric
        self.random_turn_rad     = rospy.get_param('~random_turn_rad', math.radians(15.0))  # ±15°
        self.escape_spread_rad   = rospy.get_param('~escape_spread_rad', math.radians(30.0)) # ±30° around 180°
        self.teleop_timeout      = rospy.get_param('~teleop_timeout', 0.4)         # seconds
        self.random_stride_m     = rospy.get_param('~random_stride_m', 1.0*FT_TO_M)# 1 ft stride
        self.teleop_topic        = rospy.get_param('~teleop_topic', '/teleop_cmd') # remap your keyop /cmd_vel -> /teleop_cmd
        self.log_every           = rospy.get_param('~log_every', 10)                # log every N cycles

        # ----------------- IO -----------------
        self.cmd_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)
        rospy.Subscriber('/mobile_base/events/bumper', BumperEvent, self.bump_cb)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        rospy.Subscriber(self.teleop_topic, Twist, self.teleop_cb)

        # ----------------- State -----------------
        self.halt = False
        self.mode = 'FORWARD'
        self.last_teleop = None
        self.last_teleop_time = 0.0

        self.yaw = 0.0
        self.turn_target = None
        self.last_odom = None
        self.dist_since_random = 0.0

        self.has_scan = False
        self.d_front = float('inf')
        self.d_left  = float('inf')
        self.d_right = float('inf')

        self._dbg_i = 0
        self.rate = rospy.Rate(20)  # 20 Hz main loop

    # ----------------- Callbacks -----------------
    def bump_cb(self, msg):
        if msg.state == BumperEvent.PRESSED:
            self.halt = True
            self.mode = 'HALT'

    def teleop_cb(self, msg):
        self.last_teleop = msg
        self.last_teleop_time = time.time()

    def odom_cb(self, msg):
        # Extract yaw
        q = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = yaw

        # Accumulate forward distance only when in forward-ish modes and not teleop
        if self.last_odom is not None:
            p0 = self.last_odom.pose.pose.position
            p1 = msg.pose.pose.position
            dx = p1.x - p0.x
            dy = p1.y - p0.y
            step = math.sqrt(dx*dx + dy*dy)
            if self.mode in ['FORWARD', 'AVOID'] and not self.teleop_active():
                self.dist_since_random += step
        self.last_odom = msg

    def scan_cb(self, scan):
        # Use index-based sectors for robustness: assume forward ~ middle of the array.
        n = len(scan.ranges)
        if n == 0:
            self.has_scan = False
            return

        mid = n // 2  # center bin ~ forward
        # sector half-widths in bins from desired angles
        def bins_for_deg(deg):
            return max(1, int(abs(math.radians(deg)) / scan.angle_increment))

        w_front = bins_for_deg(10.0)   # ±10°
        w_side  = bins_for_deg(15.0)   # ±15°
        off30   = bins_for_deg(30.0)   # 30° offset from center

        def sector_min_by_index(center_idx, half_width):
            lo = max(0, center_idx - half_width)
            hi = min(n - 1, center_idx + half_width)
            vals = []
            for r in scan.ranges[lo:hi+1]:
                if not (math.isinf(r) or math.isnan(r)):
                    vals.append(r)
            return min(vals) if vals else float('inf')

        self.d_front = sector_min_by_index(mid, w_front)
        self.d_left  = sector_min_by_index(min(n-1, mid + off30), w_side)
        self.d_right = sector_min_by_index(max(0,    mid - off30), w_side)
        self.has_scan = True

    # ----------------- Helpers -----------------
    def teleop_active(self):
        return (self.last_teleop is not None) and ((time.time() - self.last_teleop_time) <= self.teleop_timeout)

    def ang_diff(self, a, b):
        # smallest signed angle a - b in [-pi, pi]
        d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
        return d

    def set_turn_target(self, delta):
        # set a desired yaw = current yaw + delta (wrapped to [-pi, pi])
        self.turn_target = (self.yaw + delta + math.pi) % (2.0 * math.pi) - math.pi

    def at_target(self, tol=0.05):
        return (self.turn_target is not None) and (abs(self.ang_diff(self.turn_target, self.yaw)) < tol)

    # ----------------- Behavior selection -----------------
    def choose_behavior(self):
        # 1) HALT (latched on bumper press)
        if self.halt:
            self.mode = 'HALT'
            return Twist()  # zeros

        # 2) TELEOP override
        if self.teleop_active():
            self.mode = 'TELEOP'
            return self.last_teleop

        # 3) ESCAPE (symmetric close obstacles) & 4) AVOID (asymmetric)
        if self.has_scan:
            # symmetric if both sides are close and nearly equal
            left_close  = self.d_left  < self.avoid_range_m
            right_close = self.d_right < self.avoid_range_m
            symmetric   = left_close and right_close and (abs(self.d_left - self.d_right) < self.symmetric_eps_m)

            if symmetric or self.mode == 'ESCAPE':
                if self.mode != 'ESCAPE':
                    # choose pi +/- (escape_spread_rad/2)
                    jitter = random.uniform(-self.escape_spread_rad/2.0, self.escape_spread_rad/2.0)
                    self.set_turn_target(math.pi + jitter)
                    self.mode = 'ESCAPE'
                cmd = Twist()
                cmd.angular.z = self.turn_speed * (1.0 if self.ang_diff(self.turn_target, self.yaw) > 0.0 else -1.0)
                if self.at_target():
                    self.mode = 'FORWARD'
                    self.turn_target = None
                return cmd

            # AVOID if any front sector is closer than threshold
            if min(self.d_left, self.d_front, self.d_right) < self.avoid_range_m:
                self.mode = 'AVOID'
                cmd = Twist()
                cmd.linear.x = 0.08  # creep while avoiding

                # Turn logic:
                # - If left is the closest, turn right.
                # - If right is the closest, turn left.
                # - If front is the closest, turn toward the more open side.
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

        # 5) RANDTURN: small random heading change every stride
        if self.mode == 'RANDTURN':
            cmd = Twist()
            cmd.angular.z = self.turn_speed * (1.0 if self.ang_diff(self.turn_target, self.yaw) > 0.0 else -1.0)
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

        # 6) FORWARD (default)
        self.mode = 'FORWARD'
        cmd = Twist()
        cmd.linear.x = self.forward_speed
        return cmd

    # ----------------- Main loop -----------------
    def spin(self):
        while not rospy.is_shutdown():
            cmd = self.choose_behavior()
            self.cmd_pub.publish(cmd)

            # periodic debug: current mode and sector distances
            self._dbg_i += 1
            if self.log_every > 0 and (self._dbg_i % self.log_every == 0):
                rospy.loginfo("mode=%s dl=%.2f df=%.2f dr=%.2f",
                              self.mode, self.d_left, self.d_front, self.d_right)

            self.rate.sleep()


if __name__ == '__main__':
    ReactiveController().spin()
