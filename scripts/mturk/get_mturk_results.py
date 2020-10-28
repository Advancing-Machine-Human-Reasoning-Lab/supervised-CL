import boto3
MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
mturk = boto3.client('mturk',
   aws_access_key_id = "AKIA3MIA75GRVQAPYMMD",
   aws_secret_access_key = "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",
   region_name='us-east-1',
   endpoint_url = MTURK_SANDBOX
)

# You will need the following library
# to help parse the XML answers supplied from MTurk
# Install it in your local environment with
# pip install xmltodict
import xmltodict
# Use the hit_id previously created
"""Debug hits"""

hit_id = '3K1H3NEY7LOAWYHMWIZ60GE0DM3DG3'
# We are only publishing this task to one Worker
# So we will get back an array with one item if it has been completed
worker_results = mturk.list_assignments_for_hit(HITId=hit_id, AssignmentStatuses=['Submitted'])
print(worker_results)

# parse the responses using xmltodict
# TODO: Add in disqualifications for attention checks,
#       If a worker takes fewer than about 20 seconds to answer a question, reject
#       If a worker answer more than 70% of all questions with the same choice, reject all
#       If a worker scores 50% or less (random or anti-correlation) drop their responses and reject

# TODO: After rejections, add workers responses to json
#       Save for further analysis
if worker_results['NumResults'] > 0:
   for assignment in worker_results['Assignments']:
      xml_doc = xmltodict.parse(assignment['Answer'])
      
      print ("Worker's answer was:")
      if type(xml_doc['QuestionFormAnswers']['Answer']) is list:
         # Multiple fields in HIT layout
         for answer_field in xml_doc['QuestionFormAnswers']['Answer']:
            print ("For input field: " + answer_field['QuestionIdentifier'])
            print ("Submitted answer: " + answer_field['FreeText'])
      else:
         # One field found in HIT layout
         print ("For input field: " + xml_doc['QuestionFormAnswers']['Answer']['QuestionIdentifier'])
         print ("Submitted answer: " + xml_doc['QuestionFormAnswers']['Answer']['FreeText'])
else:
   print ("No results ready yet")