#rm -rf ./marmousiEl4JanInit3
#mkdir ./marmousiEl4JanInit3
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiElFull4JanInit2/*.tx
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiElInit/ --name MarmousiElFull4JanInit2 --model AutoElFullRhoScaleMar22 --direction AtoB --input_nc 46 --output_nc 1 --display_port 9998 --n_epochs 4000  --n_epochs_decay 2500 --batch_size 8 --gpu_ids 4 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 40 --init_type kaiming
