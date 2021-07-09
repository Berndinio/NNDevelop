# NNDevelop Hauptseminar
## Task
The goal is to predict the rating (stars from 1 to 5) of a user who wrote a review with a pretrained
BERT model. This requires to add a classifier on top of a pretrained BERT model. Afterwards we
want to take a look at what BERT focuses on.

## Dataset
The dataset is the “Amazon Review Dataset” of Jianmo Ni[NLM19], which was first released in 2014
and updated in October 2018. It contains many metadata and our needed review text and star
rating. The raw review data contain 233 million reviews. Due to this is too much for our purpose,
we specialized to the Cell Phones and Accessories data, which contains 10 million reviews + ratings.
[NLM19]

## Report
Report in Seminar_NNDevelop.pdf . Sadly pdf24 removes the pdf Links within the document.

## Install
- sudo apt-get install python-setuptools
- install requirements
    - pytorch
    - (tensorflow)
    - transformers
