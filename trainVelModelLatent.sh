
no_proxy=localhost python trainVal4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/data/train_data/SimulateData/  --name Vaevel24Sep --model Auto --direction AtoB --input_nc 1 --output_nc 1 --display_port 9998 --n_epochs 1000 --n_epochs_decay 0  --batch_size 16 --gpu_ids 2,3,4,5,6,7 --no_html --display_freq 1 --print_freq 10 --lr 0.0001 --verbose --save_epoch_freq 100 
