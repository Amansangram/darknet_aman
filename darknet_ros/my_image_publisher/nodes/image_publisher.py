#!/usr/bin/env python
import rospy
import sys
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
bridge = CvBridge()

def start_node(filename):
    rospy.init_node('image_pub')
    rospy.loginfo('image_pub node started')
    img = cv2.imread(filename)    
    imgMsg = bridge.cv2_to_imgmsg(img, "bgr8")
    pub = rospy.Publisher('image', Image, queue_size=10)
    while not rospy.is_shutdown():
        pub.publish(imgMsg)
        rospy.Rate(20).sleep()  # 1 Hz

if __name__ == '__main__':
    try:
        start_node( rospy.myargv(argv=sys.argv)[1] )
    except rospy.ROSInterruptException:
        pass
