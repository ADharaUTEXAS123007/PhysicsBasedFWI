
mkdir ./marmousi24
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiOpenFWI24/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiN/ --name MarmousiOpenFWI24 --model Auto24 --direction AtoB --input_nc 24 --output_nc 1 --display_port 9998 --n_epochs 2000 --n_epochs_decay 1500  --batch_size 8 --gpu_ids 0 --no_html --display_freq 1 --print_freq 1 --lr 0.01 --verbose --save_epoch_freq 25 --continue_train
