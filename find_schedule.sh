for epoch in 80 160
do
	for lr in 0.0005 0.0001 0.001
	do
		./tools/dist_train.sh configs/pvgnet/pvgnet_kitti-3d-car_voxel_fusion.py 4\
			--options optimizer.lr=$lr\
			--work-dir work_dirs/fusion_lr_${lr}_e${epoch}
	done
done