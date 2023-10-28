import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

check_feature = 'consume_uv_1d'


# dv_count: (39601, 97, 5) → 5 columns: content_id, visit_time, expo_uv_1d, click_uv_1d, consume_uv_1d
data = np.load('data/dv_count2.npy', allow_pickle=True) # (9916, 97, 12)
# columns：'content_id', 'visite_time', 'click_uv_1d', 'consume_uv_1d_valid', 'favor_uv_1d', 'comment_uv_1d',
#               'share_uv_1d', 'collect_uv_1d', 'attention_uv_1d', 'lead_shop_uv_1d', 'cart_uv_1d', 'consume_uv_1d'

num_content = data.shape[0]

# check the variable correlation
for j in range(1):
    y = data[j, :, :]
    series_fea_j = y[np.argsort(y[:, 1])][:, 2:].astype(np.float64)  # (time_step, fea_dim), remove 'content_id', 'visite_time'
    for day in range(30,97):
        y_past = series_fea_j[day - 3:day, -1]  # time for evaluate explosive: 3 days
        y_head = series_fea_j[day:day + 3, -1]
        if np.sum(y_past) == 0:
            y = 0
        else:
            y = np.sum(y_head) / np.sum(y_past)
        series_fea = series_fea_j[day-30:day,:].T # -> (fea_dim, K)
        wA = 1-pairwise_distances(series_fea,metric='correlation') # [-1,1] means correlation
        wA = np.array(wA)
        wA = np.nan_to_num(wA, copy=False)
        print(j, day, np.around(y,decimals=2))
        print(np.around(wA,decimals=2))

# # y_all = []
# for i in range(100):
#     exp_id = i
#     y = data[exp_id,:,:]
#     y = y[np.argsort(y[:,1])]

## for dv labeling
#     y_c = []
#     for d in range(7, len(y)-7):
#         y_past = y[d-7:d]
#         y_head = y[d:d+7]
#         if np.sum(y_past) == 0:
#             y_inc = 0
#         else:
#             y_inc = np.sum(y_head)/np.sum(y_past)
#         y_c.append(y_inc)
#     y_all.append(y_c)
#     if i % 100 == 0:
#         print(i)

    # # time series visualization
    # y_click = y[:, 2].astype(np.float64)
    # y_consume_valid = y[:, 3].astype(np.float64)
    # y_favor = y[:, 4].astype(np.float64)
    # y_comment = y[:, 5].astype(np.float64)
    # y_share = y[:, 6].astype(np.float64)
    # y_collect = y[:, 7].astype(np.float64)
    # y_attention = y[:, 8].astype(np.float64)
    # y_lead_shop = y[:, 9].astype(np.float64)
    # y_cart = y[:, 10].astype(np.float64)
    # y_consume = y[:, 11].astype(np.float64)
    #
    # t = np.arange(97)
    # plt.figure(figsize=(18, 4))
    # plt.title('Example '+ str(exp_id), fontsize=25)
    # plt.grid()
    # plt.xlabel('Time (in days)')
    # plt.plot(t, y_favor, 'black', lw=2, label='favor')
    # plt.plot(t, y_comment, 'blue', lw=2, label='comment')
    # plt.plot(t, y_share, 'red', lw=2, label='share')
    # plt.plot(t, y_collect, 'orange', lw=2, label='collect')
    # plt.plot(t, y_attention, 'green', lw=2, label='attention')
    # plt.tick_params(labelsize=25)
    # plt.legend(fontsize=20)
    # #plt.show()
    # plt.savefig('img/series/engage_'+str(exp_id)+'.jpg',dpi=300)

# dv labeling (threshold of the explosive product)
# y_all = np.array(y_all)
# print(np.percentile(y_all,  99))
# print(np.percentile(y_all,  98))
# print(np.percentile(y_all,  97))
# print(np.percentile(y_all,  95))
# print(np.percentile(y_all,  90))
# print(np.percentile(y_all,  85))
# print(np.percentile(y_all,  80))