import numpy as np
import matplotlib.pyplot as plt

check_feature = 'consume_uv_1d'

data = np.load('data/dv_count.npy', allow_pickle=True)  # (39601, 97, 5)
# 5 columns: content_id, visit_time, expo_uv_1d, click_uv_1d, consume_uv_1d
num_content = data.shape[0]

y_all = []
for i in range(num_content):
    exp_id = i
    y = data[exp_id,:,:]
    y = y[np.argsort(y[:,1])]
    y = y[:,4] # 2:'expo_uv_1d',3:'click_uv_1d',4:'consume_uv_1d'
    y_c = []
    for d in range(7, len(y)-7):
        y_past = y[d-7:d]
        y_head = y[d:d+7]
        if np.sum(y_past) == 0:
            y_inc = 0
        else:
            y_inc = np.sum(y_head)/np.sum(y_past)
        y_c.append(y_inc)
    y_all.append(y_c)
    if i % 100 == 0:
        print(i)


    # t = np.arange(len(y))
    # plt.figure(figsize=(18, 4))
    # plt.title('Example '+ str(exp_id), fontsize=25)
    # plt.grid()
    # plt.xlabel('Time (in days)')
    # plt.plot(t, y, 'black', lw=2, label=check_feature)
    # plt.tick_params(labelsize=25)
    # plt.savefig('img/series/'+check_feature+'/'+check_feature+'_'+str(exp_id)+'.jpg',dpi=300)

y_all = np.array(y_all)
print(np.percentile(y_all,  99))
print(np.percentile(y_all,  98))
print(np.percentile(y_all,  97))
print(np.percentile(y_all,  95))
print(np.percentile(y_all,  90))
print(np.percentile(y_all,  85))
print(np.percentile(y_all,  80))