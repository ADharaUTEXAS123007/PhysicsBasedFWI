rm -rf ./marmousiEl4Jan
mkdir ./marmousiEl4Jan
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiElFull4Jan23/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarineEP/ --name MarmousiElFull4Jan23 --model AutoElFull22Mar23 --direction AtoB --input_nc 382 --output_nc 841 --display_port 9999 --n_epochs 5000  --n_epochs_decay 5000 --batch_size 8 --gpu_ids 1 --no_html --display_freq 1 --print_freq 1 --lr 10 --verbose --save_epoch_freq 23
