#
# project: lidar point cloud completion in the 
#          large-scale scene
#          (based on make-it-dense)
# author:  anpei
# email:   anpei@wit.edu.cn
# data:    05.08.2022
# 

'''
prepare gt dense point cloud

please create a new file 'velodyne_dense' in dataset path:
'../sequences/07/'

'''

import time
import numpy  as np
import open3d as o3d
import tqdm

from make_it_dense.dataset import voxelField
from make_it_dense.dataset import KittiDenseVoxelDataset, KittiDenseVoxelModule
from make_it_dense.utils import load_config

def main():
    base_path = "/media/anpei/DiskC/dense_project/data/sequences/"
    seq_id    = "07"
    seq_id    = "00"
    seq_id    = "01"
    seq_id    = "02"
    seq_id    = "03"
    seq_id    = "04"
    # seq_id    = "05"
    # seq_id    = "06"
    # seq_id    = "08"
    # seq_id    = "09"
    # seq_id    = "10"
    voxel_range = [0, -40, -3, 70.4, 40, 1]
    voxel_range = [-35.2, -40, -3, 35.2, 40, 1]
    voxel_size  = [0.05, 0.05, 0.1]

    solver = voxelField(base_path, seq_id, voxel_range, voxel_size)
    num_files = solver.num_files

    for i in (range(num_files)):
        if (i % 100 == 0) & (i <= 1000):
            print(i, "/", num_files)
            solver.prepare_gt_voxel_single(idx=i, is_save=True)

    # for debug
    # solver.prepare_gt_voxel_single(idx=428, is_save=True)
    # solver.read_large_scale_test(idx=55)

def test_dataloader():
    config = load_config("./kitti.yaml")
    tmp_dataset = KittiDenseVoxelDataset(config, sequence_id="07")

    st = time.time()
    tmp_dataset.__getitem__(idx=9)
    ed = time.time()
    print("get_item time: ", ed-st)

    # dataset      = KittiDenseVoxelModule(config)
    # train_loader = dataset.train_dataloader()
    # val_loader   = dataset.val_dataloader()

if __name__ == "__main__":
    main()
    # test_dataloader()