'Alterative_least_square used for the creations of the recommendations in the hybrid approach'



import numpy as np
import pandas as pd
import scipy.sparse as sps
import implicit as impl
from sklearn.preprocessing import MinMaxScaler


top_pop = ['2778525','1244196','1386412', '657183', '2791339']
target_users = np.genfromtxt("/Users/matteopagliari/PycharmProjects/RecSys/Dataset/target_users.csv", delimiter="\t",
                             dtype=np.dtype(int), skip_header=1)
users = pd.read_csv('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/user_profile.csv', delimiter="\t")
items = pd.read_csv('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/item_profile.csv', delimiter="\t")
interactions = pd.read_csv('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/interactions.csv',delimiter='\t')
zero_int_users = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/zero_int_users.csv')
zero_int_users = np.asarray(zero_int_users)



# drop useless columns
inter = interactions.drop(['interaction_type', 'created_at'], 1)

# count duplicates
inter = inter.groupby(['user_id', 'item_id']).size().reset_index()
inter = inter.rename(columns={0: 'count'})



# users and items dict : associate user/item id with index
users_dict = dict(zip(users['user_id'], users.index))
items_dict = dict(zip(items['id'], items.index))
inv_users_dict = {v: k for k, v in users_dict.items()}  # inverse of users_dict
inv_items_dict = {v: k for k, v in items_dict.items()}  # inverse of items_dict



# add users/items indexes columns
inter['user_index'] = [users_dict.get(i) for i in inter['user_id']]
inter['item_index'] = [items_dict.get(i) for i in inter['item_id']]



# non-active items ids/indexes
items_nact_ids = items[items['active_during_test'] == 0]['id'].drop_duplicates().values
items_nact = [items_dict.get(i) for i in items_nact_ids]


# URM : user-rating matrix
urm = sps.csr_matrix((inter['count'], (inter['user_index'], inter['item_index'])))


# ALS: Alternative Least Squares

alpha = 40
factors = 300
regularization = 0.01
iterations = 20 

user_vecs, item_vecs = impl.alternating_least_squares((urm * alpha).astype('double'), factors, regularization, iterations)

l = len(target_users)

subm = np.zeros((10000,1000),dtype=int)
scores = np.zeros((10000,1000),dtype=float)

#Create recommendations and scores matrices
n=0
for u_id in target_users:

    i +=1

    u = users_dict.get(u_id)

    # users with no interactions
    if(u_id==zero_int_users[0]):
        subm[n,:5]=top_pop
        zero_int_users=zero_int_users[1:]

    # users with interactions
    else:

        # preferences of user index u
        pref = urm[u, :].toarray()
        pref = pref.reshape(-1) + 1

        # already interacted items equal to 0
        pref[pref > 1] = 0

        # non-active items equal to 0
        #pref[items_nact] = 0

        # dot product of user vector with all item vectors
        rec_vector = user_vecs[u, :].dot(item_vecs.T)
        # scale recommendation vector rec_vector between 0 and 1
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]

        # already interacted and non-active items indexes multiplied by 0
        rec = pref*rec_vector_scaled

        # takes 5 best items indexes
        ritem_indexes = np.argsort(rec)[::-1][:1000]    #[:5]

        # scores of the top 1000 items
        user_scores = rec[ritem_indexes]

        # item ids from indexes
        ritem_ids = [inv_items_dict.get(i) for i in ritem_indexes]

        # fill submission matrix
        subm[n,:] = np.reshape(ritem_ids,(1000, ))
        scores[n,:] = np.reshape(user_scores,(1000, ))

    n +=1

    l -= 1
    print(l)


np.savetxt("/Users/matteopagliari/PycharmProjects/RecSys/Dataset/als_hybrid_all.csv", subm, delimiter=",")
np.savetxt("/Users/matteopagliari/PycharmProjects/RecSys/Dataset/als_hybrid_all_score.csv", scores, delimiter=",")

