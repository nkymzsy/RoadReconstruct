#编写一个类读取KITTI的数据集的bin点云个label
import numpy as np
import os


class ReadKitti:
    """ 读取KITTI数据集的bin点云个label """ 
    def __init__(self, bin_path, lable_path):
        self.label_list = os.listdir(lable_path)
        self.label_list.sort()
        self.label_list = [os.path.join(lable_path, i) for i in self.label_list]

        self.bin_list = os.listdir(bin_path)
        self.bin_list.sort()
        self.bin_list = [os.path.join(bin_path, i) for i in self.bin_list]

        self.index = 0

    def read_labels(label_filename):
        """ 读取并返回 SemanticKITTI 数据集的标签文件 """
        label = np.fromfile(label_filename, dtype=np.uint32)
        # SemanticKITTI 使用 32位无符号整数存储标签，高16位为实例标签，低16位为语义标签
        semantic_label = label & 0xFFFF  # 提取低16位的语义标签
        return semantic_label

    def read_bin(bin_filename):
        """ 读取并返回 SemanticKITTI 数据集的点云文件 """
        return np.fromfile(bin_filename, dtype=np.float32).reshape(-1, 4)

    def read_kitti_data(self):
        """ 以字典的形式按顺序返回一帧点云和标签 """
        self.index += 1
        if(self.index < len(self.bin_list)):
            label = ReadKitti.read_labels(self.label_list[self.index])
            cloud = ReadKitti.read_bin(self.bin_list[self.index])
            return {'state': True, 'label': label, 'cloud': cloud}
        return {'state': False}
