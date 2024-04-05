#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def webcam_publisher():
    # Initialize ROS node
    print("hii")
    rospy.init_node('webcam_publisher', anonymous=True)
    print("hello")
    # Initialize OpenCV
    cap = cv2.VideoCapture(0)
    bridge = CvBridge()

    # Create publisher to publish frames
    image_pub = rospy.Publisher('webcam_frames', Image, queue_size=10)

    rate = rospy.Rate(30)  # Set the publishing rate (30 frames per second in this case)

    while not rospy.is_shutdown():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to ROS image message
        ros_image = bridge.cv2_to_imgmsg(frame, "bgr8")

        # Publish ROS image message
        image_pub.publish(ros_image)

        # Sleep to control the publishing rate
        rate.sleep()

    # Release the capture
    cap.release()

if __name__ == '__main__':
    try:
        webcam_publisher()
    except rospy.ROSInterruptException:
        pass
