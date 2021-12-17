
mkdir ./marmousi23
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiOpenFWI23/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/SEAM/ --name MarmousiOpenFWI23 --model Auto23 --direction AtoB --input_nc 30 --output_nc 1 --display_port 9991 --n_epochs 2500 --n_epochs_decay 2000  --batch_size 8 --gpu_ids 2 --no_html --display_freq 1 --print_freq 1 --lr 0.01 --verbose --save_epoch_freq 25
