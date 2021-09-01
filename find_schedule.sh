for wd in 0.01 0.005 0.001
do
	for lr in 0.0001 0.00005
	do
		./tools/dist_train.sh configs/pvgnet/fusion.py 4\
			--options optimizer.lr=$lr optimizer.weight_decay=$wd\
			  total_epochs=160\
			--work-dir work_dirs/fusion_lr_${lr}_${wd}_e320
	done
done