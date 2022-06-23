rm -rf ./marmousiEl18June
mkdir ./marmousiEl18June
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiElFullMar22/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/ --name MarmousiElFullMar22 --model AutoElFullMar22 --direction AtoB --input_nc 45 --output_nc 1 --display_port 9998 --n_epochs 4000  --n_epochs_decay 2500 --batch_size 8 --gpu_ids 1 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 30 --init_type kaiming --epoch 2520 --continue_train
