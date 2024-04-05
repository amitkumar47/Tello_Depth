"""this is the code for publisher and subscriber in a single node working simultaneous"""
#!/usr/bin/env python

import rospy
from std_msgs.msg import String

class PubSubNode:
    def __init__(self):
        rospy.init_node('pub_sub_node', anonymous=True)

        # Publisher
        self.pub = rospy.Publisher('chatter', String, queue_size=10)

        # Subscriber
        rospy.Subscriber('chatter', String, self.callback)

    def callback(self, data):
        rospy.loginfo('I heard: %s', data.data)

    def run(self):
        rate = rospy.Rate(1)  # 1Hz
        count = 0
        while not rospy.is_shutdown():
            msg = "Hello world %d" % count
            rospy.loginfo(msg)
            self.pub.publish(msg)
            count += 1
            rate.sleep()

if __name__ == '__main__':
    try:
        node = PubSubNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
