rm -rf ./marmousiBP4JanInit
mkdir ./marmousiBP4JanInit
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiBP4JanInit/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/SEAM2/ --name MarmousiBP4JanInit --model AutoElBPRhoScaleMar22 --direction AtoB --input_nc 51 --output_nc 1 --display_port 9998 --n_epochs 4000  --n_epochs_decay 2500 --batch_size 8 --gpu_ids 5 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 40 --init_type kaiming
