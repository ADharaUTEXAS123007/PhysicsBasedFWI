
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/Marmousi/ --name AutoModel18OMarmousi --model Auto2 --direction AtoB --input_nc 32 --output_nc 1 --display_port 9996 --n_epochs 1000 --n_epochs_decay 0  --batch_size 7  --gpu_ids 3,1,2,4,5,6,7 --no_html --display_freq 1 --print_freq 1 --lr 0.0001 --verbose --save_epoch_freq 10000
