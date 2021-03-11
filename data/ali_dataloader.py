import numpy as np
import  pandas as pd

def expand_year(data):
    data_expand=[]
    for i in range(data.shape[0]):
        data_expand.append(data[i][:12])
    data_expand = np.array(data_expand).reshape(-1,)
    return data_expand
def expand_year2(data):
    data_expand=[]
    for i in range(data.shape[0]):
        data_expand.append(data[i][:12])
    data_expand = np.array(data_expand)
    return data_expand
def data_to_pd(path_feature='G:\\base\\aliyun\\SODA_train.npy',path_target='G:\\base\\aliyun\\SODA_label.npy'):
    target = np.load(path_target)
    target = expand_year(target)
    pd_data = pd.DataFrame()
    t = []
    y=1800
    for m in range(target.shape[0]):
        if (m+1)%12==0:
            t.append('{}-{}-{}'.format(y,12,'01'))
            y+=1
        else:
            if m+1>=10:
                t.append('{}-{}-{}'.format(y,(m + 1) % 12, '01'))
            else:
                t.append('{}-0{}-{}'.format(y,(m + 1) % 12, '01'))

    pd_data['date'] = pd.to_datetime(t, errors='coerce')
    feature = np.load(path_feature)
    f =  ['sst','t300', 'ua','va']
    for i in range(feature.shape[0]):
        s = expand_year2(feature[i])
        s1 = s.reshape(target.shape[0],-1)
        for j in range(s1.shape[1]):
            pd_data[f[i]+'_'+str(j)] = s1[:,j].tolist()

    pd_data['nino'] = target.tolist()
    return pd_data


def get_test():
    return
if __name__ == "__main__":
    data = data_to_pd()
    print(data.columns)