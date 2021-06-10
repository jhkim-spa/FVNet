python tools/visualize_fv.py\
	--pcd ./data/kitti/training/velodyne_reduced\
	--img ./data/kitti/training/image_2\
	--calib ./data/kitti/training/calib\
	--label ./data/kitti/training/label_2\
	--config ./configs/fvnet/fv_inbox_unet_single_cyclic_80e.py\
	--checkpoint ./data/kitti/epoch_38.pth