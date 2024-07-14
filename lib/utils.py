import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField

import numpy as np

def numpy_to_ros_pointcloud2(point_cloud, frame_id='world'):
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    if(point_cloud.shape[1] == 4):
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensity', 12, PointField.FLOAT32, 1)]
    else:
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        
    cloud_msg = pc2.create_cloud(header, fields, point_cloud.view(np.float32).reshape(point_cloud.shape + (-1,)))
    return cloud_msg

def update_intensity_with_labels(point_cloud, labels):

    # 验证点云数据和标签数据的长度是否匹配
    if len(point_cloud) != len(labels):
        raise ValueError("Point cloud and labels must have the same number of points.")
    
    # 假设点云数据的最后一列是强度
    intensity_index = -1
    
    # 更新点云数据中的强度值
    point_cloud[:, intensity_index] = labels
    
    return point_cloud