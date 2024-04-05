#!/usr/bin/python3

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from djitellopy import Tello
import time
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from std_msgs.msg import Int32



class publish_image:
    def __init__(self):    
        rospy.init_node("my_cam", anonymous=True)        
        self.bridge = CvBridge()
        self.tello = Tello()
        self.tello.connect()
        print("Tello Battery :",self.tello.query_battery())
        time.sleep(5)
        self.tello.streamon()
        self.frame_read = self.tello.get_frame_read()
        print("frame reading")
        self.encoder = 'vits'

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(self.encoder)).to(self.DEVICE)
        self.depth_anything.eval()
        self.transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        ])

        #publisher'
        self.pub = rospy.Publisher("image_raw", Image, queue_size=10)

        #subscriber
        rospy.Subscriber('action', Int32 , self.callback)

    def callback(self, data):
        #you have to write the take action
        self.take_action=1  

    
    def publisher(self):
        rate = rospy.Rate(1) 
        while not rospy.is_shutdown():        
            frame = self.tello.get_frame_read().frame
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            h, w = frame.shape[:2]
            image = self.transform({'image': frame})['image']
            image = torch.from_numpy(image).unsqueeze(0).to(self.DEVICE)
            with torch.no_grad():
                depth = self.depth_anything(image)
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            
            depth = depth.cpu().numpy().astype(np.uint8)
            depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            # Capture a frame
            
            if depth_color is None:
                rospy.ERROR("Could not grab a frame!")
                break
            # Publish the image to the topic image_raw
            try:
                img_msg = self.bridge.cv2_to_imgmsg(depth_color, "bgr8")
                self.pub.publish(img_msg)
            except CvBridgeError as error:
                print(error)
    
if __name__== "__main__":
    
    try:
        node = publish_image()
        node.publisher()
        
    except KeyboardInterrupt:
        print("Shutting down!")