import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler, SMOTE
from simpletransformers.classification import ClassificationModel
import sklearn
import os
from scipy.stats import spearmanr
from scipy.stats import pearsonr

# read the dev set
df = pd.read_json("../data/snli_1.0/snli_1.0_dev.jsonl",lines=True)
agreement = [labels.count(gold_label) for labels, gold_label in zip(list(df['annotator_labels'].values), list(df['gold_label'].values)) if gold_label != '' and labels]

# calculate the agreement
agr = pd.DataFrame(agreement, columns=['agreement'])

# create the dataframe
df = pd.DataFrame({
    'text_a': df['sentence1'],
    'text_b': df['sentence2'],
    'labels': agr['agreement']/5
})

models = ['roberta-large', 'roberta-base']
epochs = [1,2,3,4]
lrs = [1e-5,5e-6]
bss = [8,16,32]
kfold = KFold(10, True, 1)


for model_name in models:
    for epoch in epochs:
        for lr in lrs:
            for bs in bss:
                file = open("../logs/nn_regression_snli.txt","a")
                result = f'Model: {model_name} Epoch: {epoch} LR: {lr} Batch: {bs} \n'
                file.write(result)
                for train, test in kfold.split(df):
                    
                    train_df = df.iloc[train] 
                    test_df = df.iloc[test]
                    
                    train_args={
                        "output_dir": "outputs_regression/",
                        "best_model_dir": "outputs_regression/best_model/",
                        'reprocess_input_data': True,
                        'overwrite_output_dir': True,
                        'num_train_epochs': epoch,
                        'train_batch_size': bs,
                        'eval_batch_size': bs,
                        'learning_rate': lr,
                        'regression': True

                    }

                    # Create a ClassificationModel
                    model = ClassificationModel('roberta', model_name, num_labels=1, use_cuda=True, cuda_device=2, args=train_args)

                    # Train the model
                    model.train_model(train_df, eval_df=test_df)
                    result, model_outputs, wrong_predictions = model.eval_model(test_df)
                   # print(pearsonr(model_outputs, test_df['labels'].values))
                   # print(spearmanr(model_outputs, test_df['labels'].values))
                    file.write("  " + str(pearsonr(model_outputs, test_df['labels'].values)))
                    file.write('\n')
                    file.write("  " + str(spearmanr(model_outputs, test_df['labels'].values)))
                    file.write('\n')
                    
                    os.system('rm -r outputs_regression/')
                    
                
                file.write('\n')
                file.close()
