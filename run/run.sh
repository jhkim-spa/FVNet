# interp_fg1_no_normalize
# interp_fg1_normalize
# interp_fg15_no_normalize
# interp_fg15_normalize
# no_interp_fg1_no_normalize
# no_interp_fg1_normalize
# no_interp_fg15_no_normalize
# no_interp_fg15_normalize

./tools/dist_train.sh configs/pvgnet/pvgnet_kitti-3d-car_anchor.py 4\
	--options model.interpolation=True\
			  model.bbox_head.fg_weight=1\
			  model.bbox_head.bbox_coder.normalize=False\
 	--work-dir work_dirs/interp_fg1_no_normalize &&\
./tools/dist_train.sh configs/pvgnet/pvgnet_kitti-3d-car_anchor.py 4\
	--options model.interpolation=True\
			  model.bbox_head.fg_weight=1\
			  model.bbox_head.bbox_coder.normalize=True\
 	--work-dir work_dirs/interp_fg1_normalize &&\
./tools/dist_train.sh configs/pvgnet/pvgnet_kitti-3d-car_anchor.py 4\
	--options model.interpolation=True\
			  model.bbox_head.fg_weight=15\
			  model.bbox_head.bbox_coder.normalize=False\
 	--work-dir work_dirs/interp_fg15_no_normalize &&\
./tools/dist_train.sh configs/pvgnet/pvgnet_kitti-3d-car_anchor.py 4\
	--options model.interpolation=True\
			  model.bbox_head.fg_weight=15\
			  model.bbox_head.bbox_coder.normalize=True\
 	--work-dir work_dirs/interp_fg15_normalize &&\

./tools/dist_train.sh configs/pvgnet/pvgnet_kitti-3d-car_anchor.py 4\
	--options model.interpolation=False\
			  model.bbox_head.fg_weight=1\
			  model.bbox_head.bbox_coder.normalize=False\
 	--work-dir work_dirs/no_interp_fg1_no_normalize &&\
./tools/dist_train.sh configs/pvgnet/pvgnet_kitti-3d-car_anchor.py 4\
	--options model.interpolation=False\
			  model.bbox_head.fg_weight=1\
			  model.bbox_head.bbox_coder.normalize=True\
 	--work-dir work_dirs/no_interp_fg1_normalize &&\
./tools/dist_train.sh configs/pvgnet/pvgnet_kitti-3d-car_anchor.py 4\
	--options model.interpolation=False\
			  model.bbox_head.fg_weight=15\
			  model.bbox_head.bbox_coder.normalize=False\
 	--work-dir work_dirs/no_interp_fg15_no_normalize &&\
./tools/dist_train.sh configs/pvgnet/pvgnet_kitti-3d-car_anchor.py 4\
	--options model.interpolation=False\
			  model.bbox_head.fg_weight=15\
			  model.bbox_head.bbox_coder.normalize=True\
 	--work-dir work_dirs/no_interp_fg15_normalize