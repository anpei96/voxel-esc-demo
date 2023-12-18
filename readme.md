
## pre-cache

python precache.py -s 07
python dump_training_data.py -s 07

python precache.py --sequence 00

## train

python train.py --config kitti.yaml 

## test 

python test_scan.py --cuda  data/sequences/00/velodyne/000000.bin
python test_scan.py --cuda  data/sequences/07/velodyne/000007.bin


