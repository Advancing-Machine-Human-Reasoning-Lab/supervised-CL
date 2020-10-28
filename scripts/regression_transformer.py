import json
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

global surveyQuestions
surveyQuestions = {}


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
    


LoadSurvey()

keys = surveyQuestions.keys()
surveyQuestions_clean = {}
for key in keys:
    if surveyQuestions[key]['total_annotators'] != 0:
        surveyQuestions_clean[key] = surveyQuestions[key]

surveyQuestions = surveyQuestions_clean

keys = surveyQuestions.keys()

X = [[surveyQuestions[key]['premise'], surveyQuestions[key]['hypothesis'], float(surveyQuestions[key]['total_correct_annotators']/surveyQuestions[key]['total_annotators'])] for key in keys]


from simpletransformers.classification import ClassificationModel
models = ['roberta-base', 'roberta-large']
epochs = [3,4,5,6,7,8]
lrs = [1e-3,1e-4,1e-5,1e-6,5e-3,5e-4,5e-5,5e-6]
bss = [2,4,8]
kfold = KFold(10, True, 1)

X = np.array(X)                    



for model_name in models:
    for epoch in epochs:
        for lr in lrs:
            for bs in bss:
                pearson = []
                spearman = []
                file = open("nn_results.txt","a")
                result = f'Model: {model_name} Epoch: {epoch} LR: {lr} Batch: {bs} \n'
                file.write(result)
                for train, test in kfold.split(X):
                    
                    
                    train_df = pd.DataFrame(X[train], columns=['text_a', 'text_b', 'labels'])
                    test_df = pd.DataFrame(X[test], columns=['text_a', 'text_b', 'labels'])
                    
                    
                    train_df['labels'] = pd.to_numeric(train_df['labels'])
                    test_df['labels'] = pd.to_numeric(test_df['labels'])

                    train_args={
                        'reprocess_input_data': True,
                        'overwrite_output_dir': True,
                        'num_train_epochs': epoch,
                        'train_batch_size': bs,
                        'eval_batch_size': bs,
                        'learning_rate': lr,
                        'regression': True,
                    }

                    # Create a ClassificationModel
                    model = ClassificationModel('roberta', model_name, num_labels=1, use_cuda=True, cuda_device=0, args=train_args)

                    # Train the model
                    model.train_model(train_df)
                    result, model_outputs, wrong_predictions = model.eval_model(test_df)
                    file.write(" Pearson: " + str(pearsonr(model_outputs, test_df['labels'].values)))
                    file.write('\n')
                    file.write(" Spearman: " + str(spearmanr(model_outputs, test_df['labels'].values)))
                    file.write('\n')
                    
                    os.system('rm -r outputs/')
                
                file.write('\n')
                file.close()


                
                    
                    
                    
                    