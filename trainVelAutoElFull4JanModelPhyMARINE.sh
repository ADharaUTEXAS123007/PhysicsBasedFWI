rm -rf ./marmousiEl4Jan
mkdir ./marmousiEl4Jan
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiElFull4Jan23/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarineEP/ --name MarmousiElFull4Jan23 --model AutoElFull22Mar23 --direction AtoB --input_nc 2 --output_nc 2 --display_port 9999 --n_epochs 4000  --n_epochs_decay 2500 --batch_size 8 --gpu_ids 5 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 20 
