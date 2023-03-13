#!/usr/bin/env python3

import rospy
import cv2
import os
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from lane_controller import LaneController
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import BoolStamped, VehicleCorners
from geometry_msgs.msg import Point32
from std_msgs.msg import String, Float32
import time
import statistics
import numpy as np

class LaneFollowingNode(DTROS):
    def __init__(self, node_name) -> None:
        super(LaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = rospy.get_namespace().strip("/")
        self.host = str(os.environ['VEHICLE_NAME'])
        self.lane_offset = -100
        self.height_ratio = 0.8
        self.height_ratio_red = 0.8
        
        # Subscriber
        self.sub_image_compressed = rospy.Subscriber(f"/{self.veh_name}/camera_node/image/compressed", CompressedImage, self.cb_compressed_image)
        self.sub_centers = rospy.Subscriber("/{}/duckiebot_detection_node/centers".format(self.host), VehicleCorners, self.cb_centers, queue_size=1)
        self.sub_circlepattern_image = rospy.Subscriber("/{}/duckiebot_detection_node/detection_image/compressed".format(self.host), CompressedImage, self.cb_circlepattern_image, queue_size=1)
        self.sub_detection = rospy.Subscriber("/{}/duckiebot_detection_node/detection".format(self.host), BoolStamped, self.cb_detection, queue_size=1)
        self.sub_distance_to_robot_ahead = rospy.Subscriber("/{}/duckiebot_distance_node/distance".format(self.host), Float32,self.cb_distance, queue_size=1)
        self.compressed_image_cache = None
        
        # Publisher
        # self.pub_executed_commands = rospy.Publisher("/{}/wheels_driver_node/wheels_cmd".format(self.veh_name), WheelsCmdStamped, queue_size=1)
        self.pub_car_commands = rospy.Publisher(f"/{self.veh_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_augmented_image = rospy.Publisher(f'/{self.veh_name}/lane_following_node/image/compressed', CompressedImage, queue_size=1)
        
        # Assistant module
        self.bridge = CvBridge()
        self.controller = LaneController()
    
    def cb_compressed_image(self, compressed_image):
        self.compressed_image_cache = compressed_image
    
    def cb_centers(self, vehicle_corners):
        pass
    
    def cb_circlepattern_image(self, compressed_image):
        pass
    
    def cb_detection(self, bool_stamped):
        pass
    
    def cb_distance(self, distance):
        pass
    
    def move(self, v, omega):
        new_cmd = Twist2DStamped()
        new_cmd.header.stamp = rospy.Time.now()
        new_cmd.header.frame_id = "~/car_cmd"
        new_cmd.v = v
        new_cmd.omega = omega
        self.pub_car_commands.publish(new_cmd)
        
    def undistort(self, img):
        return cv2.undistort(img,
                             self.camera_mat,
                             self.dist_coef,
                             None,
                             self.camera_mat)
        
    def get_med(self, cont):
        med = 0
        for c in cont:
            area = cv2.contourArea(c)
            mom = cv2.moments(c)    # Get momentum of a contour
            if mom["m00"] == 0.0:
                # If denomenator of momentum is 0
                continue
            med = int(mom["m10"] / mom["m00"])
                
        return med
    
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.compressed_image_cache is not None:
                image = self.bridge.compressed_imgmsg_to_cv2(self.compressed_image_cache)
                height, width, _ = image.shape
                #undistort
                #image = self.undistort(image)
                # Filter colour
                imhsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                #lower = np.array([20, 150, 150], dtype="uint8") 
                #upper = np.array([55, 255, 255], dtype="uint8")
                lower_yellow = np.array([20, 70, 170], dtype="uint8") 
                upper_yellow = np.array([32, 255, 255], dtype="uint8")
                lower_red = np.array([151, 155, 84], dtype = "uint8") 
                upper_red= np.array([179, 255, 255], dtype = "uint8")
                
                mask_yellow = cv2.inRange(imhsv, lower_yellow, upper_yellow)
                mask_yellow[0:int(height*(1 - self.height_ratio)), 0:width] = 0
                mask_yellow[int(height*self.height_ratio):height, 0:width] = 0
                
                mask_red = cv2.inRange(imhsv, lower_red, upper_red)
                mask_red[0:int(height*(self.height_ratio_red)), 0:width] = 0
                # # size = np.sum(mask_red/255.) / mask_red.size
                # pixels = np.sum(mask_red/255.)
                # rospy.loginfo(f'red pixels:{pixels}')
                # mask_red = cv2.inRange(imhsv[int(height*(self.height_ratio_red)):, :], lower_red, upper_red)
                # size = np.sum(mask_red/255.) / mask_red.size
                # pixels = np.sum(mask_red)
                # rospy.loginfo(f'red pixels:{pixels}')
                
                # mask_red[int(height*self.height_ratio):height, 0:width] = 0e
                # Find contours -> [np.array of shape (n, 1, 2)]
                contours, hierarchy = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours_red, hierarchy = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   
                # Comopute target point
                current_point = [width/2 + self.lane_offset, height/2]
                rospy.loginfo(f'current: {current_point}')
                             
                if len(contours) > 0:
                    # if detect contours, find the one with the largest area
                    contour = max(contours, key=cv2.contourArea) 
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
                else:
                    contour = np.array(current_point).reshape((1, 1, 2))
                    
                if len(contours_red) > 0:
                    contour_red = max(contours_red, key=cv2.contourArea)
                    yx = self.get_med(contours_red)
                    self.pv = []
                    self.pv.append(yx)
                    yx = int(np.mean(self.pv))
                    # yx = int(np.mean(self.pv[max(-10, -len(self.pv)):]))
                    
                    rospy.loginfo(f'red line deteced:{yx}')
                    self.move(0,0)
                    rospy.sleep(3)
                else:
                    target_point = contour.mean(axis=0).squeeze()
                    rospy.loginfo(f'target: {target_point}')
                    # Call controller to get the next action
                    v, omega = self.controller.getNextCommand(target_point, current_point)
                    # Send Command
                    self.move(v, omega)
                    # Publish Image
                    self.pub_augmented_image.publish(self.bridge.cv2_to_compressed_imgmsg(image))
                    rospy.loginfo('red line not deteced')
            rate.sleep()

if __name__ == "__main__":
    node = LaneFollowingNode(node_name="lane_following_node")
    node.run()
    rospy.spin()
