python tools/visualize_fv.py\
	--pcd ./data/kitti/validation/velodyne_reduced\
	--img ./data/kitti/validation/image_2\
	--calib ./data/kitti/validation/calib\
	--label ./data/kitti/validation/label_2\
	--config ./configs/fvnet/fv_inbox_unet_single_step_150e.py\
	--checkpoint ./work_dirs/complete/fv_inbox_unet_single_step_150e/epoch_72.pth