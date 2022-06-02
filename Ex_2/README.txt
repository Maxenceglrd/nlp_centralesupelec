# NLP Exercise 2: Aspect-Based Sentiment Analysis

## 1. Authors of the project

Maxence Gélard: maxence.gelard@student-cs.fr
Marien Chenaud: marien.chenaud@student-cs.fr

## 2. Description of the final system

The goal of the exercise was to build an aspect-based sentiment analysis system, i.e we aimed
at building a classifier that would predict the polarity of opinion for any tuple <aspect_category, aspect_term,
sentence>.

We were given a train set and a dev set, in which each line gave:
- The polarity of the opinion (i.e the label we want to predict)
- Different information: sentence, aspect category, target term and
character offsets of the term.

To solve this issue, we first build a feature representation of the datasets:
a. We compute the TF-IDF scores using the sentences, from which we have remove
stopwords (given by the "english" stopwords of the sklearn TF-IDF vectorizer)

b. We encoded the aspect category using a label encoding.

c. We then focused on building more hand-crafted features to try to capture the polarity
of the targeted aspect term. For each sentence, we built a vector of size 3, each component corresponding
respectively to some knowledge on the positive, negative and neutral polarity of this sentence.
More precisely we tokenized each sentence using the SpaCy parser, and only considered tokens whose POS Tagging
was either "Adjective" (ADJ), "Adverb" (ADV) or "Interjection" (INTJ), as they would be the tokens that
are the most likely to convey any form of feeling / opinion polarity. Then for each of this retained tokens,
we get an embedding of dimension 100 using a GloVe model ("glove-wiki-gigaword-100") **.
Then using each embedding we compute three scores:
- positiveness: dot product between the token embedding and the embedding of word "positive".
- negativeness: dot product between the token embedding and the embedding of word "negative".
- neutrality: dot product between the token embedding and the embedding of word "neutral".
Then we find which of these 3 scores is the highest and add it to the corresponding entry in the sentence vector
of size 3 (i.e for example if sentence vector is [0.0, 0.0, 0.0] and we get
[positiveness, negativeness, neutrality] = [3.4, 0.9, -0.1], sentence vector will be modified to [3.4, 0.0, 0.0]).
However, before adding the corresponding score to the sentence vector, we divide it by the distance between the
considered token and the aspect term we are focusing on (this word distance is obtained by some pre-processing using
the character offsets that are given in the dataset). The intuition behind this is that tokens closer to the aspect
term are more likely to be informative about this term polarity.

Then we use a Gaussian  Process classifier (with Rational Quadratic kernel),
after having encoded the labels (polarities: positive = 0, negative = 1,
neutral = 2), and train it using the described features.

Also, different models have been tried (Logistic Regression, AdaBoost, MLPClassifier, OneVsRest SVM Classifier
and RandomForest) but they all gave worse results than the Gaussian Process Model.
 The main drawback of this classifier is the training time, compared for example to a Logistic Regression.

**the GloVe model is downloaded using the gensim API, so an Internet connection is required if you don't already
have the model locally, in order to download it.


## 3. Accuracy on the dev set

With the final system described above, we got an accuracy of 80.05% of the dev set.

