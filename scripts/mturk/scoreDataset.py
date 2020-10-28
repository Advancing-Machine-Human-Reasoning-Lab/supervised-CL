"""
    Using the human responses, create the dataset and train a svr model
    to predict difficulty scores
"""
import json
import numpy as np
import pickle
import pandas as pd
import susi # for self-organizing map
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from susi.SOMPlots import plot_estimation_map, plot_som_histogram, plot_umatrix
# from sentence_transformers import SentenceTransformer #https://github.com/UKPLab/sentence-transformers

global surveyQuestions
surveyQuestions = {}

# load the questions from the survey into memory
# only kept participants who got better than random (33%)
# TODO: change if necessary
def LoadSurvey():
    print("Load questions...")
    answer_map = {
    'definitely correct': 'e',
    'definitely incorrect': 'c',
    'neither definitely correct nor definitely incorrect': 'n'
    }   # use to encode responses to ANLI labels

    questions = open("./ANLI_R3_Survey_Final.qsf", 'r')
    js = json.load(questions)

    response_df = pd.read_csv('./ANLI_R3_Survey_Final_September 4, 2020_09.04.csv')
    response_questions = response_df.drop(['StartDate','EndDate','Status','IPAddress','Progress','Duration (in seconds)','Finished','RecordedDate','ResponseId','RecipientLastName', \
    'RecipientFirstName','RecipientEmail','ExternalReference','LocationLatitude','LocationLongitude','DistributionChannel','UserLanguage'],axis=1)
    response_questions.rename(columns={"Random ID": "RandomId"}, inplace=True)

    labels = pd.read_json('./anli/r3/train.jsonl',lines=True)
    labels.drop(['reason','uid'],axis=1,inplace=True)

    for i in range(21, 319):    # the indices of the actual questions
        qid = js['SurveyElements'][i]['PrimaryAttribute']
        question = js['SurveyElements'][i]['Payload']['QuestionText']
        # get rid of the template
        question = question.replace("<strong>", "").replace("</strong>","").replace("""<span style="font-weight: bolder;">Given the Premise, is the Hypothesis definitely correct, definitely incorrect, or neither? You should only choose definitely correct if the Hypothesis is clearly stated in the Premise. You should only choose definitely incorrect if the Hypothesis is clearly contradicted</span><span style="font-weight: bolder;"> by the Premise.</span>
<div><br>""", "").replace("""<p><span style="font-weight: bolder;"><b>""", "").replace("""</b></span> <span style="white-space: pre-wrap;">""","").replace("</span></p>","").replace("<br><p><span><b>Hypothesis: </b></span>","").replace("</p>\n</div>","").replace("Premise: ", "")
        # separate the premise and hypothesis
        sentences = question.split("\n")
        sentences = [z for z in sentences if z != '']
        premise = sentences[0]
        hypothesis = sentences[1]
        surveyQuestions[qid] = {'premise':premise, 'hypothesis':hypothesis, 'total_annotators':0, 'total_correct_annotators':0}
    

    # calulate how many participants got each question right
    for id in range(len(list(response_questions.index))):
        answers = dict(response_questions.iloc[id].dropna())
        for k,v in answers.items():
            try:
                question = surveyQuestions[k]
            except KeyError:
                # skip attention checks
                continue
            try:
                correct_label = labels[labels['hypothesis'] == question['hypothesis']]['label'].values[0]
            except IndexError:
                print(f"'{question['hypothesis']}' not found")
                question['total_annotators'] += 1
                question['total_correct_annotators'] += 1
                surveyQuestions[k] = question
                continue
            
            if answer_map[v] == correct_label:
                question['total_annotators'] += 1
                question['total_correct_annotators'] += 1
                surveyQuestions[k] = question
            else:
                question['total_annotators'] += 1
                surveyQuestions[k] = question
        
    print("here")

        


# Given the path to the survey, encode the data
# using sentence transformers and save a json
def EncodeDataset():
    print("Encoding questions...")
    model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    # questions are stored in the survey
    questions = open("./ANLI_R3_Survey_Final.qsf", 'r')
    js = json.load(questions)
    for i in range(21, 319):    # the indices of the actual questions
        qid = js['SurveyElements'][i]['PrimaryAttribute']
        question = js['SurveyElements'][i]['Payload']['QuestionText']
        # get rid of the template
        question = question.replace("<strong>", "").replace("</strong>","").replace("""<span style="font-weight: bolder;">Given the Premise, is the Hypothesis definitely correct, definitely incorrect, or neither? You should only choose definitely correct if the Hypothesis is clearly stated in the Premise. You should only choose definitely incorrect if the Hypothesis is clearly contradicted</span><span style="font-weight: bolder;"> by the Premise.</span>
<div><br>""", "").replace("""<p><span style="font-weight: bolder;"><b>""", "").replace("""</b></span> <span style="white-space: pre-wrap;">""","").replace("</span></p>","").replace("<br><p><span><b>Hypothesis: </b></span>","").replace("</p>\n</div>","").replace("Premise: ", "")
        # separate the premise and hypothesis
        sentences = question.split("\n")
        sentences = [z for z in sentences if z != '']
        premise = sentences[0]
        hypothesis = sentences[1]
        encoding = model.encode([premise,hypothesis])
        with open("ANLI_roberta_vectors.jsonl", 'a') as F:
            F.write(json.dumps({'qid':qid, 'premise':encoding[0].tolist(), 'hypothesis':encoding[1].tolist()}) + '\n')


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
    X = []
    y = []

    # first create the feature set and label for each example
    with open(surveyVectorsPath, 'r') as vecFile:
        for line in vecFile.readlines():
            vector = json.loads(line)

            thisX = []
            thisX.append(cdist([vector['premise']],[vector['hypothesis']],"cosine")[0][0])
            thisX.append(len(surveyQuestions[vector['qid']]['premise']))
            thisX.append(len(surveyQuestions[vector['qid']]['hypothesis']))
            thisX.append(len(surveyQuestions[vector['qid']]['premise'].split()))
            thisX.append(len(surveyQuestions[vector['qid']]['hypothesis'].split()))
            X.append(thisX)

            total = surveyQuestions[vector['qid']]['total_annotators']
            total_correct = surveyQuestions[vector['qid']]['total_correct_annotators']
            if total == 0:
                y.append(0) 
            else:
                y.append(float(total_correct/total))

    # TODO: Clarify with Dr. L the exact feature set he used if necessary
    # Train a few regression models and do a correlation analysis
    split = int(0.95*len(X))
    global X_test, y_test, X_train, y_train
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    # used to evaluate the quality of the regression
    def getSpearmanCorrelation(model):
        # results = model.fit(X_train, y_train).predict(X_test)
        # results = model.fit(X_train, y_train).predict(X_test)
        y_test = [0,0,0]
        results = [0,0,0]
        return spearmanr(y_test, results)
    
    def getPearsonCorrelation(model):
        results = model.fit(X_train, y_train).predict(X_test)
        return pearsonr(y_test, results)

    print("Train/test size:", len(X_train), len(X_test))
    print("Dropping responses with less than 0.5 accuracy")
    # print("Testing decision trees...")
    # for depth in [1, 2, 3,5,10,25]:
    #     regressor = DecisionTreeRegressor(max_depth = depth)
    #     print("Score w/depth", depth, ":", getCorrelation(regressor))
    
    # linreg = linear_model.LinearRegression()
    # print("Linear regression:", getCorrelation(linreg))


    # print("Testing SVR...")
    # print("Linear Kernels using scaled data...")
    # for kernel in ['linear']:
    #     for C in [1,0.1]:
    #         for gamma in [0.01,0.001,0.0001,0.00001,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    #             model = SVR(kernel=kernel, C=C, gamma=gamma)
    #             print(f"kernel: {kernel}, C: {C}, gamma: {gamma}")
    #             print("score: ", getCorrelation(model))
    
    # print("rbf kernel...")
    # for kernel in ['rbf']:
    #     for C in [1e2,1e3,1e4,1e5,1e6,1e7,1,0.1,0.01,0.001,0.0001]:
    #         for gamma in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    #             model = SVR(kernel=kernel, C=C, gamma=gamma)
    #             print(f"kernel: {kernel}, C: {C}, gamma: {gamma}")
    #             print("score: ", getCorrelation(model))



    # print("KNN model...")
    # for n in [2,3,4,5,6,7,8,9,10]:
    #     for weights in ['uniform','distance']:
    #         print(f"neighbors: {n}, weights: {weights}")
    #         model = KNeighborsRegressor(n_neighbors=n, weights=weights)
    #         print("score: ", getCorrelation(model))
    print("Starting SOM grid search...")
    with open('results.jsonl','w+') as csvfile:
        
        n_rows = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
        n_cols = [1,2,3,4,5,6,7,8,9,10]
        n_iter_supervised = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
        learning_rate_start = [0.5,0.6,0.7,0.8,0.9,1.0]
        learning_rate_end = [0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,0.02,0.01]
        i = 0
        for n_row in n_rows:
            for n_col in n_cols:
                for it in n_iter_supervised:
                    for start_lr in learning_rate_start:
                        for end_lr in learning_rate_end:
                            if i % 1000 == 0:
                                print(i)
                            row = {'n_rows':n_row,'n_cols':n_col,'n_iter_supervised':it,'start_lr':start_lr,'end_lr':end_lr}
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
                                n_jobs=128)

                            spearman = getSpearmanCorrelation(som)
                            correlation = spearman.correlation
                            pvalue = spearman.pvalue
                            row['correlaton'] = correlation
                            row['p-value'] = pvalue
                            csvfile.write(json.dumps(row))
                            csvfile.write('\n')
                            

    # print(f"Correlation: {}")

    # # estimation map
    # estimation_map = np.squeeze(som.get_estimation_map())
    # plot_estimation_map(
    #     estimation_map, cbar_label="test", cmap="viridis_r")
    # plt.show()

    # # histogram
    # bmu_list = som.get_bmus(X_train.values, som.unsuper_som_)
    # plot_som_histogram(bmu_list, 35, 35)
    # plt.show()

    # # u-matrix
    # u_matrix = som.get_u_matrix(mode="mean")
    # plot_umatrix(u_matrix, som.n_rows, som.n_columns)
    # plt.show()

    # pickle.dump(SVR(kernel='rbf', C=100.0, gamma=0.1).fit(X,y), open("svr_model.pkl", 'wb'))



    # read human responses, remove any particiapnt you don't want included first


# Using the SVR model, encode each example in the train set
# and calulate the predictedEase
# visualize the distribution of scores
# TODO
def ScoreDataset(difficultyModel,trainSetPath,scoredTrainSetPath):
    print("Scoring train problems...")
    model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
    trainSet = pd.read_json(trainSetPath,lines=True)
    newTrainSet = open(scoredTrainSetPath,'w+')
    # add column for predictedEase
    trainSet['predictedEase'] = pd.Series(np.random.randn(len(trainSet)), index=trainSet.index)
    svr = pickle.load(open("svr_model.pkl", 'rb'))
    for id in range(len(list(trainSet.index))):
        thisX = []
        premise = trainSet.iloc[id]['premise']
        hypothesis = trainSet.iloc[id]['hypothesis']
        encoding = model.encode([premise,hypothesis])
        thisX.append(cdist([encoding[0].tolist()],[encoding[1].tolist()],"cosine")[0][0])

        thisX.append(len(premise))
        thisX.append(len(hypothesis))
        thisX.append(len(premise.split()))
        thisX.append(len(hypothesis.split()))
        trainSet.iloc[id]['predictedEase'] = svr.predict([thisX])[0]
        # make sure all scores fall in the range 0 to 1
        if difficulty.iloc[i]['predictedEase'] > 1:
            difficulty.at[i,'predictedEase'] = 1
        elif difficulty.iloc[i]['predictedEase'] < 0:
            difficulty.at[i, 'predictedEase'] = 0
    
    trainSet.to_json(newTrainSet,lines=True, orient='records')




LoadSurvey()
# EncodeDataset()
CreateModel("./ANLI_roberta_vectors.jsonl")
# ScoreDataset("./svr_model.pkl",'./anli/r3/train.jsonl','./anli/r3/trainDifficulty.jsonl')
