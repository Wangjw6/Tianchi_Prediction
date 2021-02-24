import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from netCDF4 import Dataset as dt

def nc_npy():
    def read_data(path):
        CMIP_label = dt(path + '/enso_round1_train_20210201/CMIP_label.nc', 'r')
        CMIP_train = dt(path + '/enso_round1_train_20210201/CMIP_train.nc', 'r')
        SODA_label = dt(path + '/enso_round1_train_20210201/SODA_label.nc', 'r')
        SODA_train = dt(path + '/enso_round1_train_20210201/SODA_train.nc', 'r')

        return CMIP_label.variables, CMIP_train.variables, SODA_label.variables, SODA_train.variables

    # try:
    path = os.path.abspath(os.pardir)
    print('generate data root', path)
    CMIP_label, CMIP_train, SODA_label, SODA_train = read_data(path=path)
    # except:
    # path = '/home/a405/wym/work/tc'
    # print('generate data root', path)
    CMIP_label, CMIP_train, SODA_label, SODA_train = read_data(path=path)

    year = SODA_train['year'][:]
    month = SODA_train['month'][:]
    lat = SODA_train['lat'][:]
    lon = SODA_train['lon'][:]
    va = SODA_train['va'][:]
    ua = SODA_train['ua'][:]
    sst = SODA_train['sst'][:]
    t300 = SODA_train['t300'][:]
    attributes = [sst,t300,ua,va]
    data = np.zeros([4,100,36,24,72])
    for i in range(len(attributes)):
        data[i,:,:,:,:] = attributes[i].data
    np.save('SODA_train.npy', data)

    year = SODA_label['year'][:]
    month = SODA_label['month'][:]
    nino = SODA_label['nino'][:]
    print( nino.data.shape)
    np.save('SODA_label.npy', nino.data)

    year = CMIP_label['year'][:]
    month = CMIP_label['month'][:]
    nino = CMIP_label['nino'][:]
    folder_path6 = path+'/CMIP6/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path6)
    folder_path5 = path + '/CMIP5/'
    if not os.path.exists(folder_path5):
        os.makedirs(folder_path5)

    # for CIMP6
    for mode in range(15):
        data= nino.data[151*mode:151*(mode+1),:]
        np.save(folder_path6 +str(mode)+'_label.npy', data)
    # for CIMP5
    for mode in range(17):
        data= nino.data[140*mode+2265:140*(mode+1)+2265,:]
        np.save(folder_path5 +str(mode)+'_label.npy', data)

    attributes = ['sst','t300', 'ua','va']
    # for CIMP6
    for mode in range(15):
        data = np.zeros([4,151,36,24,72])
        for i in range(len(attributes)):
            data[i,:,:,:,:] = CMIP_train[attributes[i]][mode*151:(mode+1)*151,:,:,:].data
        print('CIMP6 mode%d'%(mode))
        np.save(folder_path6 +str(mode)+'_train.npy', data)
    # for CIMP5
    for mode in range(17):
        data = np.zeros([4,140,36,24,72])
        for i in range(len(attributes)):
            data[i,:,:,:,:] = CMIP_train[attributes[i]][140*mode+2265:(140)*(mode+1)+2265,:,:,:].data
        print('CIMP5 mode%d'%(mode))
        np.save(folder_path6 +str(mode)+'_train.npy', data)

