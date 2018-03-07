'Recommend items for users based on tags'



import numpy as np
import pandas as pd

top_pop = ['2778525','1244196','1386412', '657183', '2791339']
zero_int_users = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/zero_int_users.csv')
zero_int_users = np.asarray(zero_int_users)
similarity_tags = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/sim_tags_200_test1.csv', delimiter=',')
interactions = pd.read_csv('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/interactions.csv',delimiter='\t',
                           usecols=['item_id', 'user_id'])
interactions = interactions.drop_duplicates(['user_id', 'item_id'])

target_users = np.genfromtxt("/Users/matteopagliari/PycharmProjects/RecSys/Dataset/target_users.csv", delimiter="\t",
                             dtype=np.dtype(int), skip_header=1)

items = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/item_profile.csv')
active_items = items[items['active_during_test']==1]
active_items = active_items['id']
active_items = np.asarray(active_items)

inactive_items = items[items['active_during_test']==0]
inactive_items = inactive_items['id']
inactive_items = np.asarray(inactive_items)

# merge interactions with similarity_tags
merging = pd.merge(similarity_tags,interactions, how='left', right_on='item_id', left_on='item')

recommendations_list = 1000

recommendations=np.zeros((target_users.size,recommendations_list),dtype=int)
recommendations_scores=np.zeros((target_users.size,recommendations_list),dtype=float)

index = 0

for user in target_users:

    # Print completion
    if(index%100==0):
        print(index)

    # Users with no interactions
    if (user == zero_int_users[0]):

        recommendations[index,:5] = top_pop
        zero_int_users=zero_int_users[1:]

    # Users with interactions
    else:

        interactions_user = merging[merging['user_id'].isin([user])] # retrieve similar for each user
        recommendable = np.asarray(interactions_user['similar']) # recommendable items
        scores = np.asarray(interactions_user['score']) # items score

        # put 0 to inactive items
        #idx_inactive_items = np.arange(recommendable.size)[np.in1d(recommendable,inactive_items)] # idx common elements in ALS
        #scores[idx_inactive_items] = 0

        # ordered scores
        scores_ordered_idx = np.argsort(scores)[::-1]
        recommendable_ordered = recommendable[scores_ordered_idx]

        # top items
        top = recommendable[:recommendations_list]
        recommendations[index,:top.size] = top

        # top scores items
        scores_ordered = scores[scores_ordered_idx]
        top_scores = scores_ordered[:recommendations_list]
        recommendations_scores[index,:top.size] = top_scores


    index +=1 # to put in correct position elements in recommendation matrix

np.savetxt("/Users/matteopagliari/PycharmProjects/RecSys/Dataset/tags_hybrid_all.csv", recommendations, delimiter=",")
np.savetxt("/Users/matteopagliari/PycharmProjects/RecSys/Dataset/tags_hybrid_all_score.csv", recommendations_scores, delimiter=",")


# Submit recommendations
def submit(tgt,recoms):

    result=np.column_stack((tgt,recoms))
    submission = open('/Users/matteopagliari/PycharmProjects/RecSys/Submission/tags_recommendations_2.csv','w')
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

#submit(target_users,recommendations)





