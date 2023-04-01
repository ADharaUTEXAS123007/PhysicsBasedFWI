rm -rf ./marmousiEl9Mar4
mkdir ./marmousiEl9Mar4
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiElLinear22/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEl/ --name MarmousiElLinear22 --model AutoElLinear22 --direction AtoB --input_nc 28 --output_nc 1 --display_port 9998 --n_epochs 1500 --n_epochs_decay 500 --batch_size 8 --gpu_ids 5 --no_html --display_freq 1 --print_freq 1 --lr 0.0025 --verbose --save_epoch_freq 1
