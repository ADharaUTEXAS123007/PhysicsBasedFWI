
#rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/Auto2SaltPhy64/*.txt
mkdir ./VPMODEL
#rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiOpenFWI2/*.txt
no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/P/ --name Auto88Y --model Auto88 --direction BtoA --input_nc 13 --output_nc 1 --display_port 8999 --n_epochs 2000 --n_epochs_decay 2000  --batch_size 8 --gpu_ids 3 --no_html --display_freq 1 --print_freq 1 --lr 0.1 --verbose --save_epoch_freq 25
