#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, BoolStamped, VehicleCorners, WheelEncoderStamped
import os
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from geometry_msgs.msg import Point32
import time
from math import pi

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
RED_MASK = [(151, 155, 84), (179, 255, 255)]
DEBUG = False
ENGLISH = False

class LaneFollowNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh_name = rospy.get_namespace().strip("/")
        self.host = str(os.environ['VEHICLE_NAME'])
        self.veh = rospy.get_param("~veh")
        self.height_ratio = 0.8
        self.height_ratio_red = 0.8
        
        self.move = True
        self.safe_dist = 30
        
        self.detection = False  # detect tag or not
        self.distance = 100000
        self.process_intersection = False
        self.vehicle_ahead = False
        self.vehicle_direction = 0 #-1, 0,1 left stright right
        

        # Publishers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)
        self.pub_car_commands = rospy.Publisher(f"/{self.veh_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_augmented_image = rospy.Publisher(f'/{self.veh_name}/lane_following_node/image/compressed', CompressedImage, queue_size=1)
        
        # Subscribers
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.sub_x = rospy.Subscriber("/{}/duckiebot_detection_node/x".format(self.host), Float32, self.cb_x, queue_size=1)
        self.sub_detection = rospy.Subscriber("/{}/duckiebot_detection_node/detection".format(self.host), BoolStamped, self.cb_detection, queue_size=1)
        self.sub_distance_to_robot_ahead = rospy.Subscriber("/{}/duckiebot_distance_node/distance".format(self.host), Float32, self.cb_distance, queue_size=1)
        self.right_tick_sub = rospy.Subscriber(f'/{self.veh_name}/right_wheel_encoder_node/tick', 
        WheelEncoderStamped, self.right_tick,  queue_size = 1)
        self.left_tick_sub = rospy.Subscriber(f'/{self.veh_name}/left_wheel_encoder_node/tick', 
        WheelEncoderStamped, self.left_tick,  queue_size = 1)
        self.r = rospy.get_param(f'/{self.veh_name}/kinematics_node/radius', 100)
        
        # Assistant module
        self.bridge = CvBridge()
        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # PID Variables
        self.proportional = None
        if ENGLISH:
            self.offset = -220
        else:
            self.offset = 220
            
        self.velocity = 0.23
        self.move_velocity = self.velocity
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.049
        self.D = -0.004
        self.last_error = 0
        self.last_time = rospy.get_time()

        
        self.rt_initial_set = False
        self.rt = 0
        self.rt_initial_val = 0

        self.lt_initial_set = False
        self.lt = 0
        self.lt_initial_val = 0

        self.prv_rt = 0
        self.prv_lt = 0

        self.cur_dist = 0

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def right_tick(self, msg):
        if not self.rt_initial_set:
            self.rt_initial_set = True
            self.rt_initial_val = msg.data
        self.rt = msg.data - self.rt_initial_val
        # self.rt_dist = (2 * pi * self.r * self.rt_val) / 135

    def left_tick(self, msg):
        if not self.lt_initial_set:
            self.lt_initial_set = True
            self.lt_initial_val = msg.data
        self.lt = msg.data - self.lt_initial_val
        
    def cb_x(self, x):
        """
        call back function for x cordinate of center of leader's back
        """
        self.x = x.data
        # if x < 233:
        #     self.vehicle_direction = -1
        #     self.turn_left()
        # elif x<= 466:
        #     self.vehicle_direction = 0
        #     self.go_straight()
        # elif x< 640:
        #     self.vehicle_direction = 1
        #     self.turn_right()
    
    def cb_detection(self, bool_stamped):
        """
        call back function for leader detection
        """
        self.detection = bool_stamped.data 
            
    def cb_distance(self, distance):
        """
        call back function for leader distance
        """
        self.distance = 100 * (distance.data)
        rospy.loginfo(f'Distance from the robot in front: {self.distance}')
        
        if self.distance < self.safe_dist:
            self.vehicle_ahead = True
        else:
            self.vehicle_ahead = False

    def lane_follow(self, image_hsv, crop_width, crop):
        """
        detect yellow line to do lane following
        """
        mask = cv2.inRange(image_hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                first = M['m10']
                sec = M['m00']
                # rospy.loginfo(f'cx:{cx}, first: {first}, sec:{sec}')
                
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None
        
        self.drive()
            
    def mover(self, image_hsv, crop_width, crop, center=-1):
        """
        control logic
        """

        self.velocity = self.move_velocity

        if not self.process_intersection:
            # image_hsv = image_hsv[60:,:,:]
            # red line detection
            lower_red = np.array([0,50,50])
            upper_red = np.array([10,255,255])
            mask0 = cv2.inRange(image_hsv, lower_red, upper_red)

            # upper mask (170-180)
            lower_red = np.array([170,50,50])
            upper_red = np.array([180,255,255])
            mask1 = cv2.inRange(image_hsv, lower_red, upper_red)

            # join my masks
            mask = mask0+mask1
            target_size = np.sum(mask/255.) / mask.size
            # rospy.loginfo(f'Target size: {target_size}, image size: {image_hsv.shape}')

            if target_size > 0.15:
                self.move = False
                self.proportional = int(crop_width / 2)
                self.drive()
                rospy.sleep(3)
                self.move = True
                self.drive()
                self.process_intersection = True
                self.intersection_dist = self.cur_dist + 0.65
                if center < 260: 
                    rospy.loginfo('Going Left!')
                elif center < 380:
                    rospy.loginfo('Going Straight')
                else:
                    rospy.loginfo('Going Right')

        if self.process_intersection and self.cur_dist > self.intersection_dist:
            self.process_intersection = False

        if self.process_intersection:
            if center != -1:
                self.proportional = ((center) - int(crop_width / 2)) / 3.5
                return
        
        self.lane_follow(image_hsv, crop_width, crop)

    def callback(self, msg):
        """
        call back function for compressed image
        """
        img = self.jpeg.decode(msg.data)
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        delta_rt = self.rt - self.prv_rt
        delta_lt = self.lt - self.prv_lt

        self.prv_rt = self.rt
        self.prv_lt = self.lt

        delta_rw_dist = (2 * pi * self.r * delta_rt) / 135
        delta_lw_dist = (2 * pi * self.r * delta_lt) / 135

        delta_dist_cover = (delta_rw_dist + delta_lw_dist)/2

        self.cur_dist = self.cur_dist + delta_dist_cover

        if not self.detection:
            self.move = True
            self.mover(hsv, crop_width, crop)
        elif self.vehicle_ahead:
            self.move = False
            self.proportional = int(crop_width / 2)
            self.drive()
        else:
            self.move = True
            self.mover(hsv, crop_width, crop, center=self.x)
            
        self.drive()
            

    def drive(self):
        """
        use PID to drive
        """
        if self.proportional is None:
            self.twist.omega = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            if self.move:
                self.twist.v = self.velocity
                self.twist.omega = P + D
            else:
                self.twist.v = 0
                self.twist.omega = 0
                
            rospy.loginfo(f'v:{self.twist.v}, omega: {self.twist.omega}')
                
            if DEBUG:
                self.loginfo(self.proportional, P, D, self.twist.omega, self.twist.v)

        self.vel_pub.publish(self.twist)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        # node.drive()
        rate.sleep()