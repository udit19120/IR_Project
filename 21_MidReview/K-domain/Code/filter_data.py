import pandas as pd
import os


def filter_data(filePath):
    data = []
    ratings = pd.read_csv(filePath, delimiter=",", encoding="latin1")
    ratings.columns = ['userId', 'itemId', 'Rating', 'review_time']

    rate_size_dic_i = ratings.groupby('itemId').size()
    choosed_index_del_i = rate_size_dic_i.index[rate_size_dic_i < 10]
    ratings = ratings[~ratings['itemId'].isin(list(choosed_index_del_i))]

    user_unique = list(ratings['userId'].unique())
    movie_unique = list(ratings['itemId'].unique())

    u = len(user_unique)
    i = len(movie_unique)
    rating_num = len(ratings)
    return u, i, rating_num, user_unique, ratings


def get_min_group_size(ratings):
    df = pd.DataFrame(ratings, columns=['userId', 'itemID', 'Rating'])
    rate_size_dic_u = df.groupby('userId').size()
    return min(rate_size_dic_u)


def reindex_data(ratings1, dic_u=None):
    data = []
    if dic_u is None:
        user_unique = list(ratings1['userId'].unique())
        user_index = list(range(0, len(user_unique)))
        dic_u = dict(zip(user_unique, user_index))
    movie_unique1 = list(ratings1['itemId'].unique())
    movie_index1 = list(range(0, len(movie_unique1)))
    dic_m1 = dict(zip(movie_unique1, movie_index1))
    for element in ratings1.values:
        data.append((dic_u[element[0]], dic_m1[element[1]], 1))
    data = sorted(data, key=lambda x: x[0])
    return data, dic_u


def get_common_data(data, user_common):
    ratings = []
    for d in data:
        rating_new = d[d['userId'].isin(user_common)]
        ratings.append(rating_new)
    return ratings


def get_unique_lenth(ratings):
    r_n = len(ratings)

    user_unique = list(ratings['userId'].unique())
    movie_unique = list(ratings['itemId'].unique())
    u = len(user_unique)
    i = len(movie_unique)
    return u, i, r_n


def filter_user(ratings_new):

    ch_del = []
    for r in ratings_new:
        # df = pd.DataFrame(r, columns=['userId'])
        rate_size_dic = r.groupby('userId').size()
        ch_del.append(rate_size_dic.index[rate_size_dic < 5])

    gl_del = list(ch_del[0])
    for i in range(1, len(ch_del)):
        gl_del += list(ch_del[i])

    ratings_new_filtered = []
    for r in ratings_new:
        ratings_new_filtered.append(r[~r['userId'].isin(gl_del)])

    return ratings_new_filtered


def write_to_txt(data, file):
    f = open(file, 'w+')
    for i in data:
        line = '\t'.join([str(x) for x in i])+'\n'
        f.write(line)
    f.close


def get_common_user(data1, data2):
    common_user = list(set(data1).intersection(set(data2)))
    return len(common_user), common_user


k = 3

data_sets = ['cell_phones', 'digital', 'movies']

data_dic = {'cell_phones': 'cell_phones_unique_sentiment', 'digital': 'digital_music_unique_sentiment',
            'movies': 'movies_unique_sentiment'}

file_path = []
counter = 1
for i in data_dic:
    file_path.append(data_dic[i] + '.csv')
    counter = counter + 1

print(file_path)

global common_user
global c_n
u_num, i_num, r_num, common_user, data1 = filter_data(file_path[0])
print('raw_data 1 info : ', (u_num, i_num, r_num))

print(len(common_user))

data_arr = []
data_arr.append(data1)

for i in range(1, len(data_sets)):
    u_num2, i_num2, r_num2, user_unique2, data2 = filter_data(file_path[i])
    data_arr.append(data2)
    c_n, common_user = get_common_user(common_user, user_unique2)
    print('raw_data ' + str(i+1) + ' info : ', (u_num2, i_num2, r_num2))
    print(len(common_user))

print(len(common_user))
print(len(data_arr))

new_data = get_common_data(data_arr, common_user)
print(type(new_data[0]))
filtered_data = filter_user(new_data)

for t in filtered_data:
    u, i, r = get_unique_lenth(t)
    print(u, i, r)

f_data = []
f_dic_u = []
for d in filtered_data:
    if len(f_dic_u) == 0:
        f_datai, f_dic_ui = reindex_data(d)
    else:
        f_datai, f_dic_ui = reindex_data(d, f_dic_u[len(f_dic_u)-1])
    f_data.append(f_datai)
    f_dic_u.append(f_dic_ui)

minl = []

for d in f_data:
    minv = get_min_group_size(d)
    minl.append(minv)

for j in range(0, len(f_dic_u)):
    if (j+1) != len(f_dic_u):
        assert f_dic_u[j] == f_dic_u[j+1], 'user_dic not same'

print('min user group size is: ')
for m in minl:
    print(m)

counter = 0
for d in f_data:
    write_to_txt(d, data_sets[counter] + 'new_reindex.txt')
    counter = counter+1

print('write data finished!')
