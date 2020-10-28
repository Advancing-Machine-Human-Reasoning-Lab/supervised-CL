import boto3
import pandas as pd
import re
import simplejson
import shutil
import os
import random
import math
from sklearn.utils import shuffle

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
TRAIN_DATA = './anli/r3/train.jsonl'

DEFAULT_PREMISE = "Sample premise." # used for templating
DEFAULT_HYP = "Sample hypothesis."
# The location of the template
SURVEY_TEMPLATE = "Z:/Code/Lab/CL-Inference/CL-Inference/humanStudy/ANLI_Survey_Format.json"

# These are input manually after uploading to qualtrics
SURVEY_LINKS = ["https://qfreeaccountssjc1.az1.qualtrics.com/jfe/form/SV_1yTpW83rGxH72vj",
]



def debug_mturk():
    # Should return 10,000
    print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my Sandbox account")



# Create a new hit from one of the questions drawn randomly from ANLI
# To be replace:
#   Premise:    Sample premise.
#   Hypothesis: Sample hypothesis.
# Attention Checks:
# Q1: 
#   Premise:     The cat has black hair.
#   Hypothesis:  The cat does not have hair.
#   definitely incorrect
# Q2:
#   Premise:    A dog sits with his back to the door.
#   Hypothesis: The dog is not looking at the door.
#   definitely correct

attention_checks = {'Q1':{'premise':'The cat has black hair.','hypothesis':'The cat does not have hair.'},
                    'Q2':{'premise':'A dog sits with his back to the door.','hypothesis':'The dog is not looking at the door.'}}

sample_question = "<strong><span style=\"font-weight: bolder;\">Given the Premise, is the Hypothesis definitely correct, definitely incorrect, or neither? You should only choose definitely correct if the Hypothesis is clearly stated in the Premise. You should only choose definitely incorrect if the Hypothesis is clearly contradicted</span><span style=\"font-weight: bolder;\"> by the Premise.</span></strong>\n<div><br> \n<p><span style=\"font-weight: bolder;\"><b>Premise: </b></span> <span style=\"white-space: pre-wrap;\">Sample premise.</span></p>\n\n<br><p><span><b>Hypothesis: </b></span>Sample hypothesis.</p>\n</div>"

def create_hit():    
    # for i in range(len(mturk_df.index)):
    question = open('./questions.xml','r').read()
    mturk = boto3.client('mturk',
    aws_access_key_id = "AKIA3MIA75GRVQAPYMMD",
    aws_secret_access_key = "c3gGRSxK+wmT5hHyCDX86Zv3Z0bjvhBNyp5g/9iL",
    region_name='us-east-1',
    endpoint_url = MTURK_SANDBOX
    )
    # for i in range(1,2):
        # TODO: Replace the template survey link
    new_hit = mturk.create_hit(
        Title = 'Textual Entailment',
        Description = 'Given the Premise sentence, is the Hypothesis sentence definitely correct, definitely incorrect, or neither?',
        Keywords = 'text, labeling, multiple choice',
        Reward = '0.15',
        MaxAssignments = 1,
        LifetimeInSeconds = 172800,
        AssignmentDurationInSeconds = 600,
        AutoApprovalDelayInSeconds = 14400,
        Question = question,
    )
    # TODO: save the hit ids to a json file, will need this for the processing step
    #       use this format: {hit_id:hit_id, question:question, url: hit_url, label:correct answer, responses:[worker_id:worker_answer (for every worker who responds)]}
    print("A new HIT has been created. You can preview it here:")
    print("https://workersandbox.mturk.com/mturk/preview?groupId=" + new_hit['HIT']['HITGroupId'])
    print("HITID = " + new_hit['HIT']['HITId'] + " (Use to Get Results)")
        
    # Remember to modify the URL above when you're publishing
    # HITs to the live marketplace.
    # Use: https://worker.mturk.com/mturk/preview?groupId=

# Given a subset of the training set, will populate surveys of 15 questions each
# The 2 attention checks will be seeded at random locations
# def create_qualtrics_surveys(mturk_df):
#     # TODO: Make sure you remove the ansi sequences before writing
#     ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
#     for i in range(1,23):
#         # get batch of 13 questions, remove from total pool
#         sub = mturk_df.sample(n=13,replace=False)
#         mturk_df.drop(sub.index,inplace=True)
#         # generate random locations for the attention checks
#         q1_index = math.floor(random.uniform(8,21))
#         q2_index = q1_index
#         while q1_index == q2_index:
#             q2_index = math.floor(random.uniform(8,21))
        
#         shutil.copy(SURVEY_TEMPLATE, f'Z:/Code/Lab/CL-Inference/CL-Inference/humanStudy/qualtrics_survey_{i}.qsf')
#         new_survey = open(f'Z:/Code/Lab/CL-Inference/CL-Inference/humanStudy/qualtrics_survey_{i}.qsf', 'r+', errors='surrogateescape')
#         js = simplejson.load(new_survey,'utf-8')
#         k = 0   # index for the subset location
#         for j in range(8,23):
#             # handle the attention checks
#             if j == q1_index:
#                 text = js['SurveyElements'][j]['Payload']['QuestionText']
#                 text = text.replace(DEFAULT_PREMISE, attention_checks['Q1']['premise']).replace(DEFAULT_HYP, attention_checks['Q1']['hypothesis'])
#                 js['SurveyElements'][j]['Payload']['QuestionText'] = text
            
#             elif j == q2_index:
#                 text = js['SurveyElements'][j]['Payload']['QuestionText']
#                 text = text.replace(DEFAULT_PREMISE, attention_checks['Q2']['premise']).replace(DEFAULT_HYP, attention_checks['Q2']['hypothesis'])
#                 js['SurveyElements'][j]['Payload']['QuestionText'] = text
            
#             else:
#                 text = js['SurveyElements'][j]['Payload']['QuestionText']
#                 resultp = ansi_escape.sub('', sub.iloc[k]['premise'])
#                 resulth = ansi_escape.sub('', sub.iloc[k]['hypothesis'])
#                 text = text.replace(DEFAULT_PREMISE, resultp).replace(DEFAULT_HYP, resulth)
#                 js['SurveyElements'][j]['Payload']['QuestionText'] = text
#                 k += 1

#         new_survey.close()
#         os.remove(f'Z:/Code/Lab/CL-Inference/CL-Inference/humanStudy/qualtrics_survey_{i}.qsf')
#         new_survey = open(f'Z:/Code/Lab/CL-Inference/CL-Inference/humanStudy/qualtrics_survey_{i}.qsf', 'w+')
#         new_survey.write(simplejson.dumps(js))
#         new_survey.close()


def create_qualtrics_surveys(mturk_df):
    # TODO: Make sure you remove the ansi sequences before writing
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    new_survey = open(f'Z:/Code/Lab/CL-Inference/CL-Inference/humanStudy/qualtrics_survey_.txt', 'w+', encoding='utf-8', errors='surrogateescape')
    new_survey.write("[[AdvancedFormat]]\n")
    new_survey.write(f"[[Block:1]]\n")
    for i in range(0,299):
        if i == 150:
            new_survey.write(f"[[Block:2]]\n")
        # get batch of 13 questions, remove from total pool
        # sub = mturk_df.sample(n=13,replace=False)
        # mturk_df.drop(sub.index,inplace=True)
        # generate random locations for the attention checks
        # q1_index = math.floor(random.uniform(1,13))
        # q2_index = q1_index
        # while q1_index == q2_index:
        #     q2_index = math.floor(random.uniform(1,13))
        
        
        # js = simplejson.load(new_survey,'utf-8')
        k = 0   # index for the subset location
        # for j in range(1,16):
        
        new_survey.write("[[Question:MC:SingleAnswer:Horizontal]]\n")
        # handle the attention checks
        # if j == q1_index:
        #     text = sample_question.replace(DEFAULT_PREMISE, attention_checks['Q1']['premise']).replace(DEFAULT_HYP, attention_checks['Q1']['hypothesis'])
        #     new_survey.write(f"{j}. {text}")
            
        
        # elif j == q2_index:
        #     text = sample_question.replace(DEFAULT_PREMISE, attention_checks['Q2']['premise']).replace(DEFAULT_HYP, attention_checks['Q2']['hypothesis'])
        #     new_survey.write(f"{j}. {text}")
        
        # else:
        resultp = ansi_escape.sub('', mturk_df.iloc[i]['premise'])
        resulth = ansi_escape.sub('', mturk_df.iloc[i]['hypothesis'])
        text = sample_question.replace(DEFAULT_PREMISE, resultp).replace(DEFAULT_HYP, resulth)
        new_survey.write(f"{i}. {text}")
            # k += 1
        
        new_survey.write("\n\n")
        new_survey.write("[[Choices]]\n")
        new_survey.write("definitely correct\ndefinitely incorrect\nneither definitely correct nor definitely incorrect")
        new_survey.write("\n\n")
    
        # new_survey.write("[[Block:16]]\n")
        # new_survey.write("[[Question:TE]]\n")
        # new_survey.write("Please enter you Mechanical Turk worker ID, which can be found on your worker Dashboard. You must enter your ID exactly as it appears on Mechanical Turk to receive credit.")
    new_survey.close()












# obtain a set of 300 random questions from anli
def get_random_anli_questions():
    print(f"Sampling 300 questions from: {TRAIN_DATA}")
    train_df = pd.read_json(TRAIN_DATA, lines=True)
    train_df = shuffle(train_df)
    train_df.drop(['reason'], inplace=True, axis=1)
    train_df.set_index('uid', inplace=True)
    mturk_df = train_df.sample(n=299,replace=False)
    return mturk_df

def main():
    mturk_df = get_random_anli_questions()
    create_qualtrics_surveys(mturk_df)
    # create_hit()

main()

