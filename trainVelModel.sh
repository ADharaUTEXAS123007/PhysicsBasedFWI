
no_proxy=localhost python trainVal4dVel.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/ --name NoRelu --model New1U --direction AtoB --input_nc 10 --output_nc 1 --display_port 9997 --n_epochs 100 --n_epochs_decay 0  --batch_size 200  --gpu_ids 0,1,2,3,4,5,6,7 --no_html --display_freq 1 --print_freq 200 --lr 0.0001 --verbose --save_epoch_freq 100
