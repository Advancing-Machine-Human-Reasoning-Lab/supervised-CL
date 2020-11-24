import json
import numpy as np
import pickle
import pandas as pd
# import susi # for self-organizing map
import matplotlib.pyplot as plt
import pprint

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
# from susi.SOMPlots import plot_estimation_map, plot_som_histogram, plot_umatrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer #https://github.com/UKPLab/sentence-transformers

global surveyQuestions
surveyQuestions = {}

# Given the path to the survey, encode the data
# using sentence transformers and save a json
def EncodeDataset():
    print("Encoding questions...")
    model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    df = pd.read_json("../snli/snli_1.0/snli_1.0_dev.jsonl",lines=True)
    for row in range(0,len(df)):
        id = df.iloc[row]['pairID']
        premise = df.iloc[row]['sentence1']
        hypothesis = df.iloc[row]['sentence2']
        encoding = model.encode([premise,hypothesis])
        with open("SNLI_roberta_vectors.jsonl", 'a') as F:
            F.write(json.dumps({'qid':id, 'premise':encoding[0].tolist(), 'hypothesis':encoding[1].tolist()}) + '\n')

# Given the path to the encoded dataset from EncodeDataset
# fit an svr model to the dataset and save the output
# fit the model to the entire dataset
# X: The encoded premise + hypothesis vecotr
# y: percent of responses for this question that got the problem right
def CreateModel(surveyVectorsPath):
    # build the train features, feature set contains:
    #   1. The cosine distance between premise and hypothesis
    #   2. The length of premise + hypothesis in terms of characters
    #   3. The length of premise + hypothesis in terms of words
    print("Creating regression models...")
    global X, y
    X = []
    y = []

    # first create the feature set and label for each example
    with open(surveyVectorsPath, 'r') as vecFile:
        df = pd.read_json("../snli/snli_1.0/snli_1.0_dev.jsonl",lines=True)
        featureset = "just_embed_flattened"
        for line in vecFile.readlines():
            # just using the basic approach for now, % of people who got the problem right
            vector = json.loads(line)
            responses = df.loc[df['pairID'] == vector['qid']]['annotator_labels']
            correct = df.loc[df['pairID'] == vector['qid']]['gold_label']
            premise = df.loc[df['pairID'] == vector['qid']]['sentence1']
            premise = premise.values[0]
            hypothesis = df.loc[df['pairID'] == vector['qid']]['sentence2']
            hypothesis = hypothesis.values[0]
            responses = list(responses)[0]
            correct = correct.values[0]
            total_correct = 0
            total = len(responses)
            for i in responses:
                if i == correct:
                    total_correct += 1
            thisX = []
            if total == 0:
                continue
            else:
                y.append(float(total_correct/total))


            # for using just the sentence embeddings, need to flatten it to make a 2-dim input
            # for p in range(0,len(vector['premise'])):
            #     thisX.append(vector['premise'][p])
            #     thisX.append(vector['hypothesis'][p])
            
            # for h in range(0,len(vector['hypothesis'])):
            #     thisX.append(vector['hypothesis'][h])
            
            thisX.append(cdist([vector['premise']],[vector['hypothesis']],"cosine")[0][0])
            
            thisX.append(len(premise))
            thisX.append(len(hypothesis))
            thisX.append(len(premise.split()))
            thisX.append(len(hypothesis.split()))

            X.append(thisX)


            

    # Train a few regression models and do a correlation analysis
    splits = [0.9]
    
    for s in splits:
        # kfold = KFold(10, True, 1)
        split = int(s*len(X))
        # global X_test, y_test, X_train, y_train

        X = np.asarray(X)
        y = np.asarray(y)

        used to evaluate the quality of the regression
        def getCorrelation(model,X_train,X_test,y_train,y_test):
            results = model.fit(X_train, y_train).predict(X_test)
            return spearmanr(y_test, results)
        
        def getPCorrelation(model,X_train,X_test,y_train,y_test):
            results = model.fit(X_train, y_train).predict(X_test)
            return pearsonr(y_test, results)
        

        print("Train/test size:", len(X_train), len(X_test))
        print("Dropping responses with less than 0.5 accuracy")
        print("Testing decision trees...")
        do 1000 cross validation trials
        for i in range(0,11):
            X, y = shuffle(X, y)
            X = np.asarray(X)
            y = np.asarray(y)
            kfold = KFold(10, True, 1)

            print("Decision tree...")
            with open(f"decision_tree_{featureset}_cross_val.jsonl", 'a') as csvfile:
                regressor = DecisionTreeRegressor(max_depth = 40)
                for train, test in kfold.split(X,y):
                    X_train = X[train]
                    y_train = y[train]
                    X_test = X[test]
                    y_test = y[test]
                    row = {'depth':40}
                    spearman = getCorrelation(regressor,X_train,X_test,y_train,y_test)
                    correlation = spearman[0]
                    pvalue = spearman[1]
                    row['correlaton'] = correlation
                    row['p-value'] = pvalue
                    csvfile.write(json.dumps(row))
                    csvfile.write('\n')
            
            print("linear...")
            with open(f"linear_regression_{featureset}.jsonl", 'a') as csvfile:
                linreg = linear_model.LinearRegression()
                spearman = getCorrelation(linreg,X_train,X_test,y_train,y_test)
                correlation = spearman[0]
                pvalue = spearman[1]
                row['correlaton'] = correlation
                row['p-value'] = pvalue
                csvfile.write(json.dumps(row))
                csvfile.write('\n')


            print("Testing SVR...")
            print("Linear Kernels using scaled data...")
            with open(f"svr_linear_{featureset}.jsonl", 'w+') as csvfile:
                row = {'C':1.0,'gamma':'scale'} 
                model = SVR(kernel='linear', C=1.0, gamma='scale')
                speparman = getCorrelation(model,X_train,X_test,y_train,y_test)
                correlation = spearman[0]
                pvalue = spearman[1]
                row['correlaton'] = correlation
                row['p-value'] = pvalue
                csvfile.write(json.dumps(row))
                csvfile.write('\n')
            
            print("rbf kernel...")
            with open(f"svr_rbf_{featureset}_cross_val.jsonl", 'a') as csvfile:
                for c in [1.0]:
                    model = SVR(kernel='rbf', C=c, gamma='scale')
                    for train, test in kfold.split(X,y):
                        X_train = X[train]
                        y_train = y[train]
                        X_test = X[test]
                        y_test = y[test]
                        row = {'C':c,'gamma':'scale'}
                        spearman = getCorrelation(model,X_train,X_test,y_train,y_test)
                        correlation = spearman[0]
                        pvalue = spearman[1]
                        row['correlaton'] = correlation
                        row['p-value'] = pvalue
                        csvfile.write(json.dumps(row))
                        csvfile.write('\n')


            print("KNN model...")
            with open(f"knn_{featureset}.jsonl", "a") as knn_file:
                row = {'n':5,'weights':'uniform'}
                model = KNeighborsRegressor(n_neighbors=5, weights='uniform')
                spearman = getCorrelation(model,X_train,X_test,y_train,y_test)
                correlation = spearman[0]
                pvalue = spearman[1]
                row['correlaton'] = correlation
                row['p-value'] = pvalue
                knn_file.write(json.dumps(row))
                knn_file.write('\n')

            n_row = 25
            n_col = 10
            it = 1500
            start_lr = 1.0
            end_lr = 0.04
            with open(f"som_{featureset}.jsonl", "w+") as som_file:
                som = susi.SOMRegressor(
                    n_rows=n_row,
                    n_columns=n_col,
                    n_iter_supervised=it,
                    neighborhood_mode_supervised="linear",
                    learn_mode_unsupervised="min",
                    learn_mode_supervised="min",
                    learning_rate_start=start_lr,
                    learning_rate_end=end_lr,
                    random_state=None,
                    n_jobs=256)

                row = {'n_rows':n_row,'n_cols':n_col,'n_iter_supervised':it,'start_lr':start_lr,'end_lr':end_lr}
                spearman = getCorrelation(som,X_train,X_test,y_train,y_test)
                scorrelation = spearman[0]
                spvalue = spearman[1]
                # result = cross_val_score(som, X, y,scoring='neg_mean_squared_error', cv=cv)
                # row['correlaton'] = correlation
                # row['p-value'] = pvalue
                row['correation'] = scorrelation
                row['pvalue'] = spvalue
                som_file.write(json.dumps(row))
                som_file.write('\n')
        knn_file.close()
        som_file.close()

    # the decision tree does the best overall
    pickle.dump(DecisionTreeRegressor().fit(X,y), open("linear_regression_worst_model.pkl", 'wb'))

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
        trainSet.iloc[id]['predictedEase'] = model.predict([thisX])[0]
        # make sure all scores fall in the range 0 to 1
        if trainSet.iloc[id]['predictedEase'] > 1:
            trainSet.at[id,'predictedEase'] = 1
        elif trainSet.iloc[id]['predictedEase'] < 0:
            trainSet.at[id, 'predictedEase'] = 0
    
    trainSet.to_json(newTrainSet,lines=True, orient='records')
                            


# EncodeDataset()
# CreateModel("./SNLI_roberta_vectors.jsonl")
ScoreDataset("./linear_regression_worst_model.pkl","./snli_1.0/snli_1.0_train.jsonl","./snli_1.0/snli_1.0_train_sorted_worst.jsonl")
