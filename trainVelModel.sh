
no_proxy=localhost python trainVal4dVel.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/ --name velModel3 --model Auto --direction AtoB --input_nc 15 --output_nc 1 --display_port 9997 --n_epochs 2 --n_epochs_decay 2 --batch_size 7  --gpu_ids 0,1,2,3,4,5,6,7 --no_html --display_freq 1 --print_freq 200 --lr 0.0001 --verbose --save_epoch_freq 2
