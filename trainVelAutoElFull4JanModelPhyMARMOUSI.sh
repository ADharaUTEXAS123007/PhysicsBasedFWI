#rm -rf ./marmousiEl4Jan
#mkdir ./marmousiEl4Jan
rm -rf /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiElFull4Jan/*.txt
no_proxy=localhost python trainValLatent4dVel2Elastic.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/MarmousiEP/ --name MarmousiElFull4Jan22 --model AutoElFull22Mar22 --direction AtoB --input_nc 90 --output_nc 1 --display_port 9998 --n_epochs 4000  --n_epochs_decay 2500 --batch_size 8 --gpu_ids 5 --no_html --display_freq 1 --print_freq 1 --lr 0.001 --verbose --save_epoch_freq 20 
