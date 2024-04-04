#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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



def publish_image():

    tello = Tello()
    tello.connect()
    print("Tello Battery :",tello.query_battery())

    tello.streamon()
    frame_read = tello.get_frame_read()

    encoder = 'vits'
    video_path = 1
    margin_width = 50
    caption_height = 60

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE)
    depth_anything.eval()
    transform = Compose([
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
        
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = rospy.Publisher('image_topic', Image, queue_size=10)
    rate = rospy.Rate(1) 
    bridge = CvBridge()
    
    while not rospy.is_shutdown():
        frame = tello.get_frame_read().frame
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        h, w = frame.shape[:2]
        image = transform({'image': frame})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            depth = depth_anything(image)
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        ros_image = bridge.cv2_to_imgmsg(depth_color, "bgr8")
        image_pub.publish(ros_image)
        rate.sleep()
        
    

if __name__ == "__main__":
    
    try:
        publish_image()
    except rospy.ROSInterruptException:
        pass