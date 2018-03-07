'Create recommendations based on CF for hybrid model'

import graphlab as gl
import numpy as np
import pandas as pd

#Import and motification useful dataset
top_pop = ['2778525','1244196','1386412', '657183', '2791339']

count = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/count.csv',delimiter=',')
rec = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Submission/cf_1000.csv',delimiter=',')
target = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/target_users.csv')
target = np.asarray(target)
items = pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/item_profile.csv')
active_items = items[items['active_during_test']==1]
active_items = active_items['id']
active_items = np.asarray(active_items)
count = count[['item_id','count']]
count['prob'] = 0
count = count.reset_index(drop=True)



count=np.asarray(count,dtype=float)
counts=count[count[:,1].argsort()][::-1]
counts=np.asmatrix(counts)
top100=counts[:,0]
top100=top100[0:100]
top100=np.reshape(top100,(100,))



zero_int_rec=pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/zero_int_rec2.csv',delimiter='\t')
zero_int_rec=np.asmatrix(zero_int_rec)

interactions=pd.read_table('/Users/matteopagliari/PycharmProjects/RecSys/Dataset/interactions.csv',delimiter='\t',
                          usecols=['user_id','item_id','created_at'])


timestamp=np.asarray(interactions['created_at'],dtype=float)

#Return interactions by a user
def int_user(user):

    i=interactions[interactions['user_id'].isin(user)]
    i=i['item_id']
    i=i.drop_duplicates()
    return i




#Normalize timestamp
def norm(ts):

    #ts=ts-1440000000 #every ts starts with 1440000000
    min=np.min(ts)
    max=np.max(ts)
    delta=max-min
    ts_norm=np.asarray((ts-min)/delta)

    return ts_norm

#Creation dictionary item-timestamp
timestamp=norm(timestamp)
interactions['created_at']=timestamp
interactions=interactions.drop_duplicates(['user_id','item_id'])
int_dict=interactions[['item_id','created_at']]
int_items=np.asarray(interactions['item_id'],dtype=int)
int_ts=np.asarray(interactions['created_at'],dtype=float)
ints=[]
ts=[]
for i in range(int_items.shape[0]):
        ints.append(int_items[i])
        ts.append(int_ts[i])
dict_int_ts=dict(zip(ints,ts))




#Creation dictionary item-count with shrink term
prob=np.zeros((count.shape[0],1),dtype=float)
prob=[count[p,1]/count.shape[0] for p in range(count.shape[0])]
shrink=np.zeros((count.shape[0],1),dtype=float)
H=5
shrink=[count[p,1]/(count[p,1]+H) for p in range(count.shape[0])]

def create_dictionary():

    it=[]
    p=[]
    for i in range(count.shape[0]):
        it.append(count[i,0])
        p.append(shrink[i])
        #p.append(count[i,2])
    dic=dict(zip(it, p))
    return dic

dic=create_dictionary()



#Weighted scores with counts and timestamps
recs=np.asmatrix(rec)
validation=10000 #number of users to recommend
topk=1000 #number of topk elements for each user
rel_list=topk*validation
rel=np.zeros((rel_list,1),dtype=float)

#Apply timestamp weights
rel=[recs[r,2]*dic.get(recs[r,1])*dict_int_ts.get(recs[r,1]) for r in range(0,rel_list)] #with error in timestamps
rel=np.reshape(rel,(rel_list,))
for r in range(0,rel_list):
    recs[r,2]=rel[r]

#Order elements in matrix
l_originale=validation
for m in range(0,l_originale):
    scores=np.asarray(recs[m*topk:m*topk+topk])
    b=scores[scores[:,2].argsort()][::-1]
    recs[m*topk:m*topk+topk]=np.asmatrix(b)

rec_items=np.asarray(recs[:,2])
recs=np.asarray(recs[:,1])
recp=np.zeros((validation, topk),dtype=int)
rec_items_matrix=np.zeros((validation,topk),dtype=float)


#Reshape matrix in order to use recommend() function
for i in range(0,validation):
    for j in range(0,topk):
        rec=np.zeros((topk, 1))
        recp[i,j]=recs[i*topk+j]
        rec_items_matrix[i,j]=rec_items[i*topk+j]



#Create matrix with 5 recommendations in order eliminating inactive items
def recommend():

    recommendations=np.zeros((validation,1000),dtype=int)
    recommendations_score=np.zeros((validation,1000),dtype=float)

    for j in range(validation):

        if(j%100==0):
            print(j) #print completion

        topK_user_j=recp[j,:]
        topK_user_j_score=rec_items_matrix[j,:]
        recommendable=topK_user_j[0:topk]
        recommendations[j,0:topk]=recommendable
        recommendations_score[j,0:topk]=topK_user_j_score

    return recommendations,recommendations_score


#Create csv for submission
def submit(tgt,recoms):

    result=np.column_stack((tgt,recoms))
    submission = open('/Users/matteopagliari/PycharmProjects/RecSys/Submission/CFforhybrid.csv','w')
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





#------------------Main part----------------------#




recommendations,recommendations_score=recommend()
np.savetxt("/Users/matteopagliari/PycharmProjects/RecSys/Dataset/cf_hybrid_all.csv", recommendations, delimiter=",")
np.savetxt("/Users/matteopagliari/PycharmProjects/RecSys/Dataset/cf_hybrid_all_score.csv", recommendations_score, delimiter=",")





