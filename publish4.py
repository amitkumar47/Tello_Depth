#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32

def number_publisher():
    # Initialize ROS node
    rospy.init_node('number_publisher', anonymous=True)

    # Create publisher to publish numbers
    number_pub = rospy.Publisher('number_topic', Int32, queue_size=10)

    rate = rospy.Rate(1)  # Set the publishing rate (1 Hz in this case)

    count = 0

    while not rospy.is_shutdown():
        # Publish the number
        number_pub.publish(count)
        rospy.loginfo("Published number: %d", count)

        # Increment the number
        count += 1

        # Sleep to control the publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        number_publisher()
    except rospy.ROSInterruptException:
        pass
