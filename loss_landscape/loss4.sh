#mpirun -np 36  /disk/student/adhara/WORK/DeniseFWI/virginFWI/DENISE-Black-Edition/bin/denise  ./LOSS_CURVE_DATA/seis.inp ./LOSS_CURVE_DATA/seis_fwi.inp
python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/10_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic8Mar1.h5 --plot
#python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/5_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic21Sep5.h5 --plot
#python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/15_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic21Sep15.h5 --plot
#python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/50_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic21Sep50.h5 --plot
#python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/100_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic21Sep100.h5 --plot
#python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/150_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic21Sep150.h5 --plot
#python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/250_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic21Sep250.h5 --plot
#python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/350_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic21Sep350.h5 --plot
#python plot_surface2.py --cuda --model elasticnet --x=-1:1:20 --y=-1:1:20 --model_file /disk/student/adhara/Fall2021/PhysicsBasedFWI/checkpoints/MarmousiEl22/500_net_G.pth --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --surf_file elastic21Sep500.h5 --plot
#plot_2D.plot_2d_contour('elastic19Sep.h5','train_loss',2.65,20.69,0.4,True)
