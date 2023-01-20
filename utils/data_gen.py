import numpy as np
import torch
import os
import h5py

def process_rand_flow(folder, file):
    folder_name = file[:-5]
    print(folder_name)
    print(folder + file)
    randomTF_data = h5py.File(folder + file, 'r')
    group_keys = list(randomTF_data.keys())
    print(randomTF_data[group_keys[0]].shape)
    B, T, H, W = randomTF_data[group_keys[0]].shape
    if not os.path.exists(folder + folder_name):
        os.mkdir(folder + folder_name)
    split_data_x = [torch.from_numpy(randomTF_data[group_keys[0]][k*1000:(k+1)*1000,:,:,:]) for k in range(10)]
    split_data_y = [torch.from_numpy(randomTF_data[group_keys[1]][k*1000:(k+1)*1000,:,:,:]) for k in range(10)]
    for i in range(len(split_data_x)):
        torch.save(torch.FloatTensor(torch.cat((split_data_x[i], split_data_y[i]), dim = 2).reshape((1000, T, 2, H, W))), folder + folder_name + "\sample_" + str(i) + ".pt")
        


def data_gen(file):
    # read data
    data = torch.load(file)

    # standardization
    std = (data)
    avg = torch.mean(data)
    data = (data - avg)/std
    data = data[:,:,::4,::4]

    # divide each rectangular snapshot into 7 subregions
    # data_prep shape: num_subregions * time * channels * w * h
    data_prep = torch.stack([data[:,:,:,k*64:(k+1) * 64] for k in range(7)])

    # use sliding windows to generate 9870 samples
    # training 6000, validation 2000, test 1870
    for i in range(7):
        if not os.path.exists(r"dataset\2D\CFD\Turbulent_Flow\rbc_data\sample_" + str(i) + ".pt"):
            with open(r"dataset\2D\CFD\Turbulent_Flow\rbc_data\sample_" + str(i) + ".pt", "x") as f:
                pass
        torch.save(torch.FloatTensor(data_prep[i]), r"dataset\2D\CFD\Turbulent_Flow\rbc_data\sample_" + str(i) + ".pt")

if __name__ == "__main__":
    data_gen(r"dataset\2D\CFD\Turbulent_Flow\rbc_data.pt")
    process_rand_flow(r"dataset\2D\CFD\2D_Train_Rand" + chr(92), r"2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5")
