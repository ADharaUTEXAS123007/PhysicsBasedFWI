This repository contains code from my two papers published

1. Dhara, A. and Sen, M.K., 2022. Physics-guided deep autoencoder to overcome the need for a starting model in full-waveform inversion. The Leading Edge, 41(6), pp.375-381.

2. Dhara, A. and Sen, M.K., 2023. Elastic Full Waveform Inversion using a Physics guided deep convolutional encoder-decoder. IEEE Transactions on Geoscience and Remote Sensing.


The origanization of the code is inspired from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Although the project doesn't use GAN, I started building off this project since I liked the origanization of the code and also ability to visualize the convergence on visdom.


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

```

The fastest way to learn about the network architecture is by running the bash script 

trainVelAuto23ModelPhy.sh  (for the acoustic case)

which runs the command 

no_proxy=localhost python trainValLatent4dVel2.py --dataroot /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/SEAMA/ --name MarmousiOpenFWI23 --model Auto23 --direction AtoB --input_nc 20 --output_nc 1 --display_port 9991 --n_epochs 2000 --n_epochs_decay 700  --batch_size 8 --gpu_ids 5 --no_html --display_freq 1 --print_freq 1 --lr 0.005 --verbose --save_epoch_freq 25

You can now view the network architecture at "class AutoMarmousi23_Net(nn.Module):" at the networks.py file (corresponding to model Auto23 at the top of the file)

If you want to still run the code, you will have to generated observed seismic data and put it in /disk/student/adhara/Fall2021/FCNVMB-Deep-learning-based-seismic-velocity-model-building/SEAMA/ using the deepwave software


For the elastic case, you have to run the command

trainVelAutoEl22ModelPhy.sh

The repository further contains experimentation with other network architectures, for example :

trainVelAutoNFModelPhy.sh : FWI with Invertible Neural Networks










