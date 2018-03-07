'Generate similar items for targe users based on titles'

import numpy as np
import pandas as pd
import graphlab as gl


path ='/Users/matteopagliari/PycharmProjects/RecSys/Dataset'

items  =pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/item_profile.csv')
active_items = items[items['active_during_test']==1]
items_id = np.asarray(items['id'])

title = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/item_profile.csv')
title = title['title']
title = np.asarray(title)

target = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/target_users.csv')

interactions = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/interactions.csv')
interactions_idx = interactions['item_id'] # items clicked by users in interactions
interactions_idx = interactions_idx.drop_duplicates()
#interactions_idx = interactions_idx[:2]
interactions_idx = gl.SArray(interactions_idx)
interactions_target = pd.merge(target, interactions, how='inner',left_on='user_id', right_on='user_id')
interactions_target = interactions_target['item_id'].drop_duplicates()
interactions_target = pd.DataFrame(interactions_target)
interactions_target.columns = ['item']
interactions_target = gl.SFrame(interactions_target)
target = np.asarray(target)

#Retun list of interactions of the user
def inter_user(user):

    #user=[user]
    it =interactions[interactions['user_id'].isin(user)]
    return it['item_id']

#Retrive title in correct format
def separate_title(title):

    title_separate = []
    lenght = 0
    for l in range(items_id.size):

        if(title[l] != title[l]):
            title_separate.append(np.zeros((1,1),dtype=int))
            lenght += 1

        else:
            title_user = np.fromstring(title[l],dtype=int,sep=',')
            title_separate.append(title_user)
            lenght = lenght+title_user.size

    return lenght,title_separate

lenght_title, title_sep = separate_title(title)


def dictionary_item_title():

    item = []
    title = []

    for i in range(items_id.size):
        item.append(items_id[i])
        title.append(title_sep[i])

    item_title = dict(zip(item,title))

    return item_title



#Model as Pandas Dataframe
def create_matrix_model(lenght,title,items):

    model = np.zeros((lenght,2),dtype=int)
    row = 0

    for i in range(items.size):

        model[row:row+title[i].size,0]=items[i]
        model[row:row+title[i].size,1]=title[i]
        row=row+title[i].size

    model = pd.DataFrame(model)
    model.columns = ['item','title']

    return model


def generate_similar_items(model,tgt):
    model = gl.SFrame(model)
    similarity = gl.recommender.item_similarity_recommender.create(model,user_id='title',item_id='item',only_top_k=200)
    print 'compute similar'
    similar_items = similarity.get_similar_items(items=interactions_idx,k=200,verbose=True)
    similar_items.export_csv('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/sim_title_200_test1.csv')
    print 'finish'
    #si=np.asarray(similar_items)
    #print si
    return similar_items

item_title = dictionary_item_title()
m = create_matrix_model(lenght_title,title_sep,items_id)
sim5 = generate_similar_items(m,interactions_target)
#sim5.export_csv('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/sim5_title.csv')


