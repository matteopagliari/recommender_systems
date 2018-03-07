'Basic collaborative filtering'

import graphlab as gl
import pandas as pd
import numpy as np

interactions = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/interactions.csv',delimiter='\t',
                           usecols=['user_id','item_id','created_at'])

interactions = interactions.drop_duplicates(['user_id','item_id'])
interactions = gl.SFrame(interactions)
target = gl.SFrame.read_csv('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/target_users.csv')

#Base model
model = gl.recommender.item_similarity_recommender.create(interactions,user_id='user_id',item_id='item_id',
                                                          seed_item_set_size=0,only_top_k=1000)

#Returns 1000 recommended items for each target user
recommendations = model.recommend(users=target,k=1000)


recommendations.export_csv('/Users/matteopagliari/PycharmProjects/RecSys/Submission/cf_1000.csv')