import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
# from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy.spatial.distance import cdist
import sklearn
import os
import json
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer #https://github.com/UKPLab/sentence-transformers

# this function can be used to get accuracy scores for predictions of difficulty on the dev set
def CreateModel():
    df = pd.read_json("../snli_1.0/snli_1.0_dev.jsonl",lines=True) # update this line with wherever you have SNLI stored
    agreement = [labels.count(gold_label) for labels, gold_label in zip(list(df['annotator_labels'].values), list(df['gold_label'].values)) if gold_label != '' and labels]

    # calculate the agreement
    agr = pd.DataFrame(agreement, columns=['agreement'])
    # apparently 0.0, 0.6, 0.8, 1.0 are the only possible options, so mapping them to classes
    labels = {'key_0.0': 0, 'key_0.6': 1, 'key_0.8': 2, 'key_1.0': 3}

    # create the dataframe
    df = pd.DataFrame({
        'pairID': df['pairID'],
        'text_a': df['sentence1'],
        'text_b': df['sentence2'],
        'labels': agr['agreement']/5
    })

    # get the embeedings
    vec_df = pd.read_json('./SNLI_roberta_vectors.jsonl',lines=True)


    for index, row in df.iterrows():
        df['labels'][index] = int(labels['key_' + str(row['labels'])])

    # oversampling all the minority classes
    oversample = RandomOverSampler(sampling_strategy='not majority')
    df, y = oversample.fit_resample(df, df['labels'])

    # using the oversampled dataset, compute distinct feature sets
    feature_set = {
        "from_AAAI":np.asarray([]),
        "just_rob_embed_flattened":np.asarray([]),
        "just_rob_embed_stacked":np.asarray([]),
        "rob_embed_stacked_sent_len":np.asarray([]),
        "rob_embed_stacked_cdist":np.asarray([])
        }

    print("creating feature set")
    X = []
    print("from_AAAI")
    for index in range(0,len(df)):
        # create the feature vector for each and append
        question = df.iloc[index]
        vec_premise = vec_df.loc[vec_df['qid'] == question['pairID']]["premise"]
        vec_hypothesis = vec_df.loc[vec_df['qid'] == question['pairID']]["hypothesis"]

        try:
            vec_premise = vec_premise.iloc[0]
            vec_hypothesis = vec_hypothesis.iloc[0]
        except KeyError:
            print(vec_premise)
            exit()

        # 1. from AAAI
        thisX = []
        # may be more than 1 due to oversampling
        thisX.append(cdist([vec_premise],[vec_hypothesis],"cosine")[0][0])
        thisX.append(len(question['text_a']))
        thisX.append(len(question['text_b']))
        thisX.append(len(question['text_a'].split()))
        thisX.append(len(question['text_b'].split()))

        # feature_set["from_AAAI"] = np.append(feature_set["from_AAAI"],[thisX],axis=0)
        X.append(thisX)

    feature_set['from_AAAI'] = np.asarray(X)
    X = []
    print("just flattened roberta embeedings")
    for index in range(0,len(df)):
        # create the feature vector for each and append
        question = df.iloc[index]
        vec_premise = vec_df.loc[vec_df['qid'] == question['pairID']]["premise"]
        vec_hypothesis = vec_df.loc[vec_df['qid'] == question['pairID']]["hypothesis"]

        try:
            vec_premise = vec_premise.iloc[0]
            vec_hypothesis = vec_hypothesis.iloc[0]
        except KeyError:
            print(vec_premise)
            exit()

        thisX = []
        for p in range(0,len(vec_premise)):
            thisX.append(vec_premise[p])
            thisX.append(vec_hypothesis[p])
        

        X.append(thisX)

    feature_set["just_rob_embed_flattened"] = np.asarray(X)

    X=[]
    print("just embed stacked")
    for index in range(0,len(df)):
        # create the feature vector for each and append
        question = df.iloc[index]
        vec_premise = vec_df.loc[vec_df['qid'] == question['pairID']]["premise"]
        vec_hypothesis = vec_df.loc[vec_df['qid'] == question['pairID']]["hypothesis"]

        try:
            vec_premise = vec_premise.iloc[0]
            vec_hypothesis = vec_hypothesis.iloc[0]
        except KeyError:
            print(vec_premise)
            exit()

        thisX = []
        for p in range(0,len(vec_premise)):
            thisX.append(vec_premise[p])

        for h in range(0,len(vec_premise)):
            thisX.append(vec_hypothesis[h])
        X.append(thisX)

    feature_set["just_rob_embed_stacked"] = np.asarray(X)
    X=[]

    print("rob embed plus sent len")
    for index in range(0,len(df)):
        # create the feature vector for each and append
        question = df.iloc[index]
        vec_premise = vec_df.loc[vec_df['qid'] == question['pairID']]["premise"]
        vec_hypothesis = vec_df.loc[vec_df['qid'] == question['pairID']]["hypothesis"]

        try:
            vec_premise = vec_premise.iloc[0]
            vec_hypothesis = vec_hypothesis.iloc[0]
        except KeyError:
            print(vec_premise)
            exit()

        thisX = []
        for p in range(0,len(vec_premise)):
            thisX.append(vec_premise[p])

        for h in range(0,len(vec_premise)):
            thisX.append(vec_hypothesis[h])
        
        thisX.append(len(question['text_a']))
        thisX.append(len(question['text_b']))
        thisX.append(len(question['text_a'].split()))
        thisX.append(len(question['text_b'].split()))
        
        X.append(thisX)

    feature_set["rob_embed_stacked_sent_len"] = np.asarray(X)

    X=[]
    print("rob embed + cdist")
    for index in range(0,len(df)):
        # create the feature vector for each and append
        question = df.iloc[index]
        vec_premise = vec_df.loc[vec_df['qid'] == question['pairID']]["premise"]
        vec_hypothesis = vec_df.loc[vec_df['qid'] == question['pairID']]["hypothesis"]

        try:
            vec_premise = vec_premise.iloc[0]
            vec_hypothesis = vec_hypothesis.iloc[0]
        except KeyError:
            print(vec_premise)
            exit()

        thisX = []
        for p in range(0,len(vec_premise)):
            thisX.append(vec_premise[p])

        for h in range(0,len(vec_premise)):
            thisX.append(vec_hypothesis[h])
        
        thisX.append(cdist([vec_premise],[vec_hypothesis],"cosine")[0][0])
        X.append(thisX)
    feature_set["rob_embed_stacked_cdist"] = np.asarray(X)



    # decision tree
    # print("decision tree")
    # criterion = ["entropy"]
    # max_depth = [40]
    # kfold = KFold(10, True, 1)
    # X = feature_set["from_AAAI"]
    # with open(f"decision_tree_from_AAAI_cross_val.jsonl","a") as file:
    #     for c in criterion:
    #         for md in max_depth:

    #             model = DecisionTreeClassifier(max_depth=md,criterion=c)
    #             for train, test in kfold.split(X):
    #                 row = {'depth':md,'criterion':c}
                    
    #                 X_train = X[train]
    #                 y_train = y[train]
    #                 X_test = X[test]
    #                 y_test = y[test]

    #                 est = model.fit(X_train,y_train)
    #                 acc = est.score(X_test,y_test)
    #                 row['acc'] = acc
    #                 file.write(json.dumps(row))
    #                 file.write("\n")

    X = feature_set["from_AAAI"]


    # nonlinear SVC
    # print("SVC")
    # kernel = ["rbf"]
    # C = [0.2,0.4,0.6,0.8,1.0]
    # kfold = KFold(10, True, 1)
    # for f in feature_set.keys():
    #     with open(f"svr_rbf_{f}_grid_search.jsonl","a") as file:
    #         for c in C:
    #             for i in range(0,2):
    #                 X = feature_set[f]
    #                 model = SVC(C=c)
    #                 for train, test in kfold.split(X):
    #                     row = {'C':c}
                        
    #                     X_train = X[train]
    #                     y_train = y[train]
    #                     X_test = X[test]
    #                     y_test = y[test]

    #                     est = model.fit(X_train,y_train)
    #                     acc = est.score(X_test,y_test)
    #                     row['acc'] = acc
    #                     file.write(json.dumps(row))
    #                     file.write("\n")

    # logistic regression + PCA

    # logistic regression + LSA


def ScoreDataset(difficultyModel,trainSetPath,scoredTrainSetPath):
    print("Scoring train problems...")
    encoder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    trainSet = pd.read_json(trainSetPath,lines=True)
    newTrainSet = open(scoredTrainSetPath,'w+')
    # add column for predictedEase
    trainSet['predictedEase'] = pd.Series(np.random.randn(len(trainSet)), index=trainSet.index)
    model = pickle.load(open(difficultyModel, 'rb'))
    for id in range(len(list(trainSet.index))):
        thisX = []
        premise = trainSet.iloc[id]['sentence1']
        hypothesis = trainSet.iloc[id]['sentence2']
        encoding = encoder.encode([premise,hypothesis])
        # use whatever features got you the best correlation for this step
        thisX.append(cdist([encoding[0].tolist()],[encoding[1].tolist()],"cosine")[0][0])

        thisX.append(len(premise))
        thisX.append(len(hypothesis))
        thisX.append(len(premise.split()))
        thisX.append(len(hypothesis.split()))
        trainSet.at[id, 'predictedEase'] = model.predict([thisX])[0]
    
    trainSet.to_json(newTrainSet,lines=True, orient='records')

