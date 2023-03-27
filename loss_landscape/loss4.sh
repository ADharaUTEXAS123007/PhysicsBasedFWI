#mpirun -np 36  /disk/student/adhara/WORK/DeniseFWI/virginFWI/DENISE-Black-Edition/bin/denise  ./LOSS_CURVE_DATA/seis.inp ./LOSS_CURVE_DATA/seis_fwi.inp
##python plot_surface2.py --cuda --model elasticnet --x=-1:1:25 --y=-1:1:25 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/1_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic8Mar1.h5 --plot
##python plot_surface2.py --cuda --model elasticnet --x=-1:1:25 --y=-1:1:25 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/5_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic8Mar5.h5 --plot
##python plot_surface2.py --cuda --model elasticnet --x=-1:1:25 --y=-1:1:25 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/15_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic8Mar15.h5 --plot
##python plot_surface2.py --cuda --model elasticnet --x=-1:1:25 --y=-1:1:25 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/50_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic8Mar50.h5 --plot
python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/100_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic8Mar100.h5 --plot
python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/150_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic8Mar150.h5 --plot
python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/250_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic8Mar250.h5 --plot
python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/350_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic8Mar350.h5 --plot
python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/500_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic8Mar500.h5 --plot
#plot_2D.plot_2d_contour('elastic19Sep.h5','train_loss',2.65,20.69,0.4,True)
