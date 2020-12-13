# Difficulty Estimation for Supervised NLP Tasks

This repo implements a difficulty estimation method using human annotations in supervised NLP tasks. The experiment details and results can also be found in our AACL-IJCNLP 2020 paper.
## Contents
The scripts directory contains example code to run neural and non-neural classification/regression models. To compute the feature sets, we have also provided the RoBERTa sentence embeddings for each example in the SNLI dev [here](https://drive.google.com/file/d/12tWHUSROGQc0AMZOghrgYTfiTJOll9i-/view?usp=sharing).

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
@inproceedings{laverghetta2020towards,
  title={Towards a Task-Agnostic Model of Difficulty Estimation for Supervised Learning Tasks},
  author={Laverghetta Jr, Antonio and Mirzakhalov, Jamshidbek and Licato, John},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing: Student Research Workshop},
  pages={16--23},
  year={2020}
}
```