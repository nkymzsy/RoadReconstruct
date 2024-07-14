import rospy
import numpy as np
import torch

import lib.heightMLP as HeightMLP
import lib.readKitti as ReadKitti
from lib.train import TrainInGpu
from lib.utils import *

dataRead = ReadKitti.ReadKitti(r"/home/data/dataset/05/velodyne", r"/home/data/dataset/05/labels")
heightMLP = HeightMLP.HeightMLP(3, 5).to('cuda')
try:
    heightMLP.load_state_dict(torch.load('src/RoadReconstruct/lib/base_model.pth'))
except:
    print("no model")
trainInGpu = TrainInGpu(heightMLP)

x = np.arange(-50, 50, 0.2)
y = np.arange(-50, 50, 0.2)
X, Y = np.meshgrid(x, y)
mesh = np.column_stack((X.ravel(), Y.ravel()))
mesh_torch = torch.from_numpy(mesh / 100).float().to('cuda')

rospy.init_node('pub_height_map', anonymous=True)
pubRawCloud = rospy.Publisher('cloud', PointCloud2, queue_size=10)
pubSemanticCloud = rospy.Publisher('SemanticCloud', PointCloud2, queue_size=10)
pubGoundCloud = rospy.Publisher('GroundCloud', PointCloud2, queue_size=10)
pubGoundMesh = rospy.Publisher('GroundMesh', PointCloud2, queue_size=10)
rate = rospy.Rate(10)

while(not rospy.is_shutdown()):
    data = dataRead.read_kitti_data()
    if(data["state"] == False or len(data["cloud"]) == 0):
        break
    groundCloud = data["cloud"][data["label"] == 40]

    loss = 1
    xy = torch.from_numpy(groundCloud[:,:2]).float().to('cuda')
    z_real = torch.tensor(groundCloud[:,2:3]).float().to('cuda')
    xy = xy / 100.0
    while(loss > 2e-2):
        zPred = heightMLP(xy)
        loss = trainInGpu.UpdateOnce(z_real, zPred)/ len(groundCloud)
        print(str(dataRead.index) +":"+ str(loss))
    
    z_mesh = heightMLP(mesh_torch)
    ground_mesh = np.column_stack((mesh, z_mesh.cpu().detach().numpy()))

    torch.save(heightMLP.state_dict(), 'src/RoadReconstruct/lib/base_model.pth')

    pubGoundMesh.publish(numpy_to_ros_pointcloud2(ground_mesh.astype(np.float32)))
    pubRawCloud.publish(numpy_to_ros_pointcloud2(data["cloud"]))
    pubSemanticCloud.publish(numpy_to_ros_pointcloud2(update_intensity_with_labels(data["cloud"],data["label"])))
    pubGoundCloud.publish(numpy_to_ros_pointcloud2(groundCloud))
    
    rate.sleep()



