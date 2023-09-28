import numpy as np
import matplotlib.pyplot as plt

check_feature = 'consume_uv_1d'

data = np.load('data/dv_count.npy', allow_pickle=True)
for i in range(100):
    print(i)
    exp_id = i
    y = data[exp_id,:,:]
    y = y[np.argsort(y[:,1])]
    print(y)
    #y = y[:,4] # 2:'expo_uv_1d',3:'click_uv_1d',4:'consume_uv_1d'

    # t = np.arange(len(y))
    # plt.figure(figsize=(18, 4))
    # plt.title('Example '+ str(exp_id), fontsize=25)
    # plt.grid()
    # plt.xlabel('Time (in days)')
    # plt.plot(t, y, 'black', lw=2, label=check_feature)
    # plt.tick_params(labelsize=25)
    # plt.savefig('img/series/'+check_feature+'/'+check_feature+'_'+str(exp_id)+'.jpg',dpi=300)
