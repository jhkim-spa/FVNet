for wd in 0.01 0.003 0.001
do
	for lr in 0.003 0.001 0.0003
	do
		./tools/dist_train.sh configs/fvnet/base.py 4\
			--options optimizer.lr=$lr\
						  optimizer.weight_decay=$wd\
			--work-dir work_dirs/base_lr_${lr}_wd_${wd}
	done
done