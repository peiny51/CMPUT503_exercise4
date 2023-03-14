#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, BoolStamped, VehicleCorners
import os
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from geometry_msgs.msg import Point32
import time

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
        self.stop_time = 0.0
        self.stop_cooldown = 3.0
        self.stop_duration = 1.5
        
        self.detection = False  # detect tag or not
        self.distance = 100000
        self.lane_follow = True
        self.robot_follow = False
        self.turn = False
        self.stop = False
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
        # Subscribers
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        
        # Subscriber
        # self.sub_image_compressed = rospy.Subscriber(f"/{self.veh_name}/camera_node/image/compressed", CompressedImage, self.cb_compressed_image)
        # self.sub_centers = rospy.Subscriber("/{}/duckiebot_detection_node/centers".format(self.host), VehicleCorners, self.cb_centers, queue_size=1)
        self.sub_x = rospy.Subscriber("/{}/duckiebot_detection_node/x".format(self.host), Float32, self.cb_x, queue_size=1)
        self.sub_circlepattern_image = rospy.Subscriber("/{}/duckiebot_detection_node/detection_image/compressed".format(self.host), CompressedImage, self.cb_circlepattern_image, queue_size=1)
        self.sub_detection = rospy.Subscriber("/{}/duckiebot_detection_node/detection".format(self.host), BoolStamped, self.cb_detection, queue_size=1)
        self.sub_distance_to_robot_ahead = rospy.Subscriber("/{}/duckiebot_distance_node/distance".format(self.host), Float32, self.cb_distance, queue_size=1)
        self.compressed_image_cache = None
        
        # Publisher
        # self.pub_executed_commands = rospy.Publisher("/{}/wheels_driver_node/wheels_cmd".format(self.veh_name), WheelsCmdStamped, queue_size=1)
        self.pub_car_commands = rospy.Publisher(f"/{self.veh_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_augmented_image = rospy.Publisher(f'/{self.veh_name}/lane_following_node/image/compressed', CompressedImage, queue_size=1)
        
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
            
        self.velocity = 0.2
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.049
        self.D = -0.004
        self.last_error = 0
        self.last_time = rospy.get_time()

        # Shutdown hook
        rospy.on_shutdown(self.hook)
        
    def move(self, v, omega):
        new_cmd = Twist2DStamped()
        new_cmd.header.stamp = rospy.Time.now()
        new_cmd.header.frame_id = "~/car_cmd"
        new_cmd.v = v
        new_cmd.omega = omega
        self.pub_car_commands.publish(new_cmd)
        
        
    def cb_centers(self, vehicle_corners):
        rospy.loginfo(f'centers:{vehicle_corners}')
        
    def cb_x(self, x):
        rospy.loginfo(f'x:{x}')
        self.x = x.data
        if x < 233:
            self.vehicle_direction = -1
            self.turn_left()
        elif x<= 466:
            self.vehicle_direction = 0
            self.go_straight()
        elif x< 640:
            self.vehicle_direction = 1
            self.turn_right()
    
    def cb_circlepattern_image(self, compressed_image):
        pass
    
    def cb_detection(self, bool_stamped):
        # rospy.loginfo(f'detection bool stamp:{bool_stamped}')
        self.detection = bool_stamped.data
        if self.detection:
            self.robot_follow = True
        else:
            self.lane_follow = True
        rospy.loginfo(f'detection:{self.detection}')    
            
    def cb_distance(self, distance):
        # rospy.loginfo(f'distance:{distance}')
        self.distance = 100 * (distance.data)
        rospy.loginfo(f"here is the distance{self.distance}")
        
        if self.distance < 30:
            self.vehicle_ahead = True
        else:
            self.vehicle_ahead = False
            
    def control_logic(self):
        """
        control logic
        """

        curr_time = rospy.Time.now()
        stop_time_diff = curr_time - self.stop_time

        if (self.stop and not self.process_intersection):
            self.stop_time = curr_time

            self.process_intersection = True
            # self.set_lights("stop")
            self.move(0, 0)
            rospy.sleep(self.stop_duration)
        
        elif self.vehicle_ahead:
            self.move(0, 0)
            # self.set_lights("stop")
            
        elif self.process_intersection:
            if self.vehicle_direction == 0:
                # self.set_lights("off")
                self.go_straight()
            elif self.vehicle_direction == 1:
                # self.set_lights("right")
                self.turn_right()
            elif self.vehicle_direction == -1:
                # self.set_lights("left")
                self.turn_left()
            else:
                # self.set_lights("off")
                self.go_straight()
            
            self.process_intersection = False
        else:
            v, omega = self.controller.getNextCommand(target_point, current_point)

            self.move(v, omega)

        self.rate.sleep()
        
    def turn_right(self):
        """Make a right turn at an intersection"""
        self.lane_pid_controller.disable_controller()
        
        self.move(v=0.3, omega=0)
        rospy.sleep(1)
        self.move(v=0.3, omega = -4)
        rospy.sleep(1.5)
        self.stop = False
        self.lane_pid_controller.enable_controller()

    def turn_left(self):
        """Make a left turn at an intersection"""
        self.lane_pid_controller.disable_controller()
        self.move(v=0.3, omega = 3.5)
        rospy.sleep(4)
        self.stop = False
        self.lane_pid_controller.enable_controller()

    def go_straight(self):
        """Go straight at an intersection"""
        self.lane_pid_controller.disable_controller()
        self.move(v = 0.4, omega = 0.0)
        rospy.sleep(2)
        self.cmd_stop = False
        self.lane_pid_controller.enable_controller()
    

    def callback(self, msg):
        img = self.jpeg.decode(msg.data)
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # red line detection
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        height, width, _ = image.shape
        imhsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([151, 155, 84], dtype = "uint8") 
        upper_red= np.array([179, 255, 255], dtype = "uint8")
        
        
        mask_red = cv2.inRange(imhsv, lower_red, upper_red)
        mask_red[0:int(height*(self.height_ratio_red)), 0:width] = 0
        
        
        # Search for lane in front
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
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None

        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)

    def drive(self):
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

            self.twist.v = self.velocity
            self.twist.omega = P + D
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
        node.drive()
        rate.sleep()