# Difficulty Estimation for Supervised NLP Tasks

This repo implements a difficulty estimation method using human annotations in supervised NLP tasks. The experiment details and results can also be found in our AACL-IJCNLP 2020 paper.
## Contents
The scripts directory contains example code to run neural and non-neural classification/regression models. To compute the feature sets, we have also provided the RoBERTa sentence embeddings for each example in the SNLI dev set in ```SNLI_roberta_vectors.zip```.

## Requirements
The following packages are needed to run this code:

1. numpy
2. scipy
3. sklearn
4. PyTorch
5. SimpleTransformers
6. SentenceTransformers (If you want to train the sentence embeddings yourself.)
7. imblearn
8. pandas

Please refer to the documentation of each specific package for instructions on installation. Note that not all packages are needed to run every experiment.

### Reference

If you use this code, please consider citing us:

```
@InProceedings{
    author = "Antonio Laverghetta Jr., Jamshidbek Mirzakahalov, John Licato",
    title = "Towards a Task-Agnostic Model of Difficulty Estimation for Supervised Learning Tasks",
    year = "2020"
}
```

The full BibTeX entry will be updated upon publication of the work.