
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Marmousi/ --name VaeLatent2Phy10Oct --model VaeLatent2NoPhy --direction AtoB --input_nc 1 --output_nc 1 --display_port 9996 --n_epochs 200 --n_epochs_decay 0  --batch_size 32  --gpu_ids 0,2,3,4,5,6,7 --no_html --display_freq 5 --print_freq 5 --lr 0.0001 --verbose --save_epoch_freq 500 --continue_train
