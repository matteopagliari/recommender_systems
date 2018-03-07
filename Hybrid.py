'Hybrid approach CF, ALS, tags, titles'

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


top_pop = ['2778525','1244196','1386412', '657183', '2791339']

# CF
cf = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/cf_hybrid_all.csv',delimiter=',',header=None)
cf = np.asmatrix(cf)
cf_scores = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/cf_hybrid_all_score.csv',delimiter=',',
                          header=None)
cf_scores = np.asmatrix(cf_scores)

# ALS
als = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/als_hybrid_all.csv',delimiter=',',header=None)
als = np.asmatrix(als)
als_scores = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/als_hybrid_all_score.csv',delimiter=',',
                           header=None)
als_scores = np.asmatrix(als_scores)

# TAGS
tags = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/tags_hybrid_all.csv',delimiter=',',header=None)
tags = np.asmatrix(tags)
tags_scores = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/tags_hybrid_all_score.csv',delimiter=',',
                            header=None)
tags_scores = np.asmatrix(tags_scores)

# TITLES
title = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/title_hybrid_all.csv',delimiter=',',header=None)
title = np.asmatrix(title)
title_scores = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/title_hybrid_all_score.csv',delimiter=',',
                            header=None)
title_scores = np.asmatrix(title_scores)

# ZERO INTERACTIONS USERS
zero_int_users = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/zero_int_users.csv')
zero_int_users = np.asarray(zero_int_users)

# TARGET
target_users = np.genfromtxt("/Users/matteopagliari/PycharmProjects/RecSys/Dataset/target_users.csv", delimiter="\t",
                             dtype=np.dtype(int), skip_header=1)

# ITEMS
items = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/item_profile.csv')
active_items = items[items['active_during_test']==1] # active items
active_items = active_items['id']
active_items = np.asarray(active_items)
inactive_items = items[items['active_during_test']==0] # inactive items
inactive_items = inactive_items['id']
inactive_items = np.asarray(inactive_items)


# inizialize recommendations matrix
recommendations=np.zeros((target_users.size,5),dtype=int)

for i in range(target_users.size):

    # Print completion
    if(i%100==0):
        print(i)

    # Users with no interactions
    if (target_users[i] == zero_int_users[0]):
        recommendations[i,:] = top_pop
        zero_int_users=zero_int_users[1:]


    # Users with interactions
    else:

        cf_vec = cf[i,:] # item recommended by CF
        cf_scores_vec = cf_scores[i,:] # item scores CF

        als_vec = als[i,:] # item recommended by ALS
        als_scores_vec = als_scores[i,:] # item scores ALS

        tags_vec = tags[i,:] # item recommended by tags
        unique_tags_idx = np.unique(np.asarray(tags_vec[0,:]), return_index=True)[1]
        #[tags_vec[:, unique_tags_idx] for index in sorted(unique_tags_idx)] # unique tags

        tags_scores_vec = tags_scores[i,:] # item scores tags
        #[tags_scores_vec[:, unique_tags_idx] for index in sorted(unique_tags_idx)]  # unique tags scores

        title_vec = title[i,:] # item recommended by title
        unique_title_idx = np.unique(np.asarray(title_vec[0,:]), return_index=True)[1]
        #[title_vec[:, unique_title_idx] for index in sorted(unique_title_idx)] # unique title
        title_scores_vec = title_scores[i,:] # item scores title
        #[title_scores_vec[:, unique_title_idx] for index in sorted(unique_title_idx)]  # unique title scores

        # scales values CF between 0 and 1
        min_max = MinMaxScaler()
        cf_scores_vec = min_max.fit_transform(cf_scores_vec.reshape(-1, 1))[:, 0]

        scores = [als_scores[i,k] for k in range(1000)] # Set initial scores equal to ALS scores
        scores=np.asarray(scores)

        idx_common_als = np.arange(als_vec.size)[np.in1d(als_vec,cf_vec)] # idx common elements in ALS-CF
        idx_common_cf = np.arange(cf_vec.size)[np.in1d(cf_vec,als_vec)] # idx common elements in CF-ALS
        idx_common_tags = np.arange(tags_vec.size)[np.in1d(tags_vec,als_vec)] # idx common elements in TAGS-ALS
        idx_common_als_tags = np.arange(als_vec.size)[np.in1d(als_vec,tags_vec)] # idx common elements in ALS-TAGS
        idx_common_title = np.arange(title_vec.size)[np.in1d(title_vec,als_vec)] # idx common elements in TITLE-ALS
        idx_common_als_title = np.arange(als_vec.size)[np.in1d(als_vec,title_vec)] # idx common elements in ALS-TITLE


        # Common items and scores ALS and CF
        common = [np.int(cf_vec[:, idx_common_cf[k]]) for k in range(idx_common_als.size)]
        common_scores_cf = [np.float(cf_scores_vec[idx_common_cf[k]]) for k in range(idx_common_als.size)]
        common_scores_als= [np.float(als_scores_vec[:,idx_common_als[k]]) for k in range(idx_common_als.size)]

        # Common items and scores ALS and TAGS
        common_tags = [np.int(als_vec[:, idx_common_als_tags[k]]) for k in range(idx_common_als_tags.size)]
        common_scores_als_tags = [np.float(als_scores_vec[:,idx_common_als_tags[k]]) for k in range(idx_common_als_tags.size)]
        common_scores_tags = [np.float(tags_scores_vec[:,idx_common_tags[k]]) for k in range(idx_common_tags.size)]

        # Common items and scores ALS and TITLE
        common_title = [np.int(als_vec[:, idx_common_als_title[k]]) for k in range(idx_common_als_title.size)]
        common_scores_als_title = [np.float(als_scores_vec[:,idx_common_als_title[k]]) for k in range(idx_common_als_title.size)]
        common_scores_title = [np.float(title_scores_vec[:,idx_common_title[k]]) for k in range(idx_common_title.size)]


        # Creation dictionary item-score ALS-CF
        item_cf = []
        item_cf_scores = []
        item_als = []
        item_als_scores = []

        for n in range(idx_common_als.size):
            item_cf.append(common[n])
            item_cf_scores.append(common_scores_cf[n])
            item_als.append(common[n])
            item_als_scores.append(common_scores_als[n])

        # dictionaries item-score CF and ALS
        cf_common_scores = dict(zip(item_cf,item_cf_scores))
        als_common_scores = dict(zip(item_als,item_als_scores))

        # Creation dictionaries item-score ALS-TAGS
        item_tags = []
        item_tags_scores = []
        item_als_tags = []
        item_als_tags_scores = []

        for m in range(idx_common_als_tags.size):
            item_tags.append(common_tags[m])
            item_tags_scores.append(common_scores_tags[m])
            item_als_tags.append(common_tags[m])
            item_als_tags_scores.append(common_scores_als_tags[m])

        # dictionaries item-score TAGS and ALS
        tags_common_scores = dict(zip(item_tags,item_tags_scores))
        als_tags_common_scores = dict(zip(item_als_tags,item_als_tags_scores))

        # Creation dictionaries item-score ALS-TITLE
        item_title = []
        item_title_scores = []
        item_als_title = []
        item_als_title_scores = []

        for m in range(idx_common_als_title.size):
            item_title.append(common_title[m])
            item_title_scores.append(common_scores_title[m])
            item_als_title.append(common_title[m])
            item_als_title_scores.append(common_scores_als_title[m])

        # dictionary item-score TITLE and ALS
        title_common_scores = dict(zip(item_title,item_title_scores))
        als_title_common_scores = dict(zip(item_als_title,item_als_title_scores))


        #scores of common elements ALS-CF
        alpha = 1 #1
        beta = 1
        #common_scores1 = [alpha * cf_common_scores.get(com) + beta * als_common_scores.get(com) for com in common]
        common_scores1 = [cf_common_scores.get(com) for com in common]

        #scores of common elements ALS-TAGS
        gamma = 1.1 #1
        delta = 0.9 #0.9
        common_scores2 = [tags_common_scores.get(coms) for coms in common_tags]

        #scores of common elements ALS-TITLE
        omega = 1 #1
        theta = 0.5
        common_scores3 = [title_common_scores.get(coms) for coms in common_title]



        # sum and put new scores ALS-CF in score array
        for c in range(idx_common_als.size):
            scores[idx_common_als[c]] = alpha * scores[idx_common_als[c]] + beta * common_scores1[c]

        # sum and put new scores ALS-TAGS in score array
        for c in range(idx_common_als_tags.size):
            scores[idx_common_als_tags[c]] = gamma * scores[idx_common_als_tags[c]] + delta * common_scores2[c]

        # sum and put new scores ALS-TITLE in score array
        for c in range(idx_common_als_title.size):
            scores[idx_common_als_title[c]] = omega * scores[idx_common_als_title[c]] + theta * common_scores3[c]


        # put 0 to non-active items
        idx_inactive_items = np.arange(als_vec.size)[np.in1d(als_vec,inactive_items)] # idx common elements in ALS
        scores[idx_inactive_items] = 0

        # idx of ordered scores
        scores_ordered_idx = np.argsort(scores)[::-1]

        # items ordered by score
        als_vec_ordered = als_vec[:,scores_ordered_idx]

        # put items in recommendations matrix
        topK_user = [als_vec_ordered[:,v] for v in range(5)]
        recommendations[i,:] = topK_user

np.savetxt("/Users/matteopagliari/PycharmProjects/RecSys/Dataset/recommendations_jobroles_test15.csv", recommendations, delimiter=",")


# Submit recommendations
def submit(tgt,recoms):

    result=np.column_stack((tgt,recoms))
    submission = open('/Users/matteopagliari/PycharmProjects/RecSys/Submission/cf_als_tags_title_all_1000.test15.csv','w')
    header = 'user_id,recommended_items'
    submission.write(header + '\n')

    for row in result:
        for i in range(np.size(row)):
            if i==0:
                line = str(row[0]) + ','
            elif i==5:
                line = line + str(row[i])
            else:
                line = line + str(row[i]) + ' '
        submission.write(line + '\n')

    submission.close()

submit(target_users,recommendations)









