import math
import nltk
import sys
import pandas as pd
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize



print("Williams, Lauren, A20437936 solution: \n") 
if len(sys.argv) != 2:
    ignore = True
elif sys.argv[1] == "YES":
    ignore = True
elif sys.argv[1] == "NO":
    ignore = False
else:
    ignore = True

if ignore:
    print("Ignored pre-processing step: STEMMING")
else:
    print("Ignored pre-processing step: NONE")

print("Training classifier…")
ps = PorterStemmer()
stop_words_dict = dict.fromkeys(set(stopwords.words('english')))    
    
# creating dataframe from csv
dataframe = pd.read_csv("bumble_google_play_reviews.csv")

# removing the NaN values from the dataframe
dataframe = dataframe[dataframe['content'].notna()]

# reviews are in content column
# pre processing: lowercasing & removing stop words skips stemming step if ignore true
if ignore:
    dataframe["content"] = dataframe["content"].apply(lambda x: [word.lower() for word in nltk.word_tokenize(x) if word not in stop_words_dict])
else:
    dataframe["content"] = dataframe["content"].apply(lambda x: [ps.stem(word) for word in nltk.word_tokenize(x) if word not in stop_words_dict])
    
# tuples of the score and the content
#tuples = dataframe.apply(lambda row: (row['score'], row['content']), axis=1).tolist()

# int val for 80% of the corpus
train_set_size = math.ceil(len(dataframe) * 0.8)

train_data = dataframe.iloc[:train_set_size]
test_data = dataframe.iloc[train_set_size:]

# create a defaultdict to hold the FreqDist objects for each score
train_fdist_dict = defaultdict(nltk.FreqDist)
train_score_size = dataframe.groupby('score').size().to_dict()
test_list = []

# training data
# creates a dictionary for the score and relates it to a FreqDist for the count of each word
# THIS INCLUDES LAPLACE SMOOTHING
for row in train_data.index:
    tokens = train_data.loc[row, "content"]
    score = train_data.loc[row, "score"]
    for token in tokens:
        if not train_fdist_dict[score].__contains__(token):
            train_fdist_dict[score].update([token])
    train_fdist_dict[score].update(tokens)

print("Testing classifier…")

prob_one = train_score_size[1] / train_set_size
prob_two = train_score_size[2] / train_set_size
prob_three = train_score_size[3] / train_set_size
prob_four = train_score_size[4] / train_set_size
prob_five = train_score_size[5] / train_set_size
tag_probs = [prob_one, prob_two, prob_three, prob_four, prob_five]

result_data = {1: {"Total": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0},
               2: {"Total": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0},
               3: {"Total": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0},
               4: {"Total": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0},
               5: {"Total": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}}

test = []
pred = []
# testing the classfier 
for row in test_data.index:
    tokens = test_data.loc[row, "content"]
    score = test_data.loc[row, "score"]
    result_data[score]["Total"] += 1
    # test[score]["test"].append(score)
    test.append(score)
    max = (0, -200000)
    # loop calculates probability of each score based on the sample and picks the highest
    for idx, tag in enumerate(tag_probs):
        probability = tag
        for token in tokens:
            if token not in train_fdist_dict[idx+1]:
                train_fdist_dict[idx+1].update([token])
            probability = probability * train_fdist_dict[idx+1].freq(token)
        if probability > max[1]:
            max = (idx+1, probability)
    pred.append(max[0])
    # if max[0] != score:
    #     test[score]["pred"].append(0)
    # calculate TP, FP, FN for each label
    if max[0] == score:
        result_data[max[0]]["TP"] += 1
        
    else:
        result_data[max[0]]["FP"] += 1
        
    
        
print("Test results / metrics:")
# displaying results for every label
# calculating the TN and FN for each label
for score in result_data:
    total_neg = 0
    for other in result_data:
        if score != other:
            total_neg += result_data[other]["Total"]
    result_data[score]["TN"] = total_neg - result_data[score]["FP"]
    result_data[score]["FN"] = result_data[score]["Total"] - result_data[score]["TP"]
    TP = result_data[score]["TP"]
    TN = result_data[score]["TN"]
    FP = result_data[score]["FP"]
    FN = result_data[score]["FN"]
    print("\nFor label score ", score, ": ")
    print("Number of true positives: ", TP)
    print("Number of true negatives: ", TN)
    print("Number of false positives: ", FP)
    print("Number of false negatives: ", FN)
    recall = TP/(TP + FN)
    print("Sensitivity (recall): ", recall)
    print("Specificity: ", TN/(TN + FP))
    precision = TP/(TP + FP)
    print("Precision: ", precision)
    print("Negative predictive value: ", TN/(TN + FN))
    print("Accuracy: ", (TP + TN) / (TP + TN + FP + FN))
    print("F-score: ", (2 * recall * precision) / ( recall + precision) )

# function to calculate sentence probability
def calc_sentence_score( sen ):
    words = nltk.word_tokenize(sen)
    if ignore:
        words = [word.lower() for word in words if word not in stop_words_dict]
    else:
        words = [ps.stem(word.lower()) for word in words if word not in stop_words_dict]
    max = (0, -200000)
    prob_list = []
    for idx, tag in enumerate(tag_probs):
        new_prob = tag
        for word in words:
            if word not in train_fdist_dict[idx+1]:
                train_fdist_dict[idx+1].update([word])
            new_prob = new_prob * train_fdist_dict[idx+1].freq(word)
        prob_list.append(new_prob)
        if new_prob > max[1]:
            max = (idx+1, new_prob)
    return (max[0], prob_list)
        

# demo sentences
def demo():
    answer = input("\nEnter your sentence: ")
    print("Sentence S: ", answer)
    prediction = calc_sentence_score(answer)
    print("was classified as ", prediction[0], "\n")
    for idx, prob in enumerate(prediction[1]):
        print("P({score} | {s}) = {pred}".format(score=idx+1, s=answer, pred = prob))

# # cm = confusion_matrix(test, pred)
# # sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd')
# # plt.xlabel('Predicted')
# # plt.ylabel('Actual')
# # plt.show()

n_classes = 5
test = label_binarize(test, classes=range(1,n_classes+1))
pred = label_binarize(pred, classes=range(1,n_classes+1))

# # Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# # # Compute micro-average ROC curve and ROC area
# # fpr["micro"], tpr["micro"], _ = roc_curve(test.ravel(), pred.ravel())
# # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 2
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.4f})'
             ''.format(i+1, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Bumble Reviews')
plt.legend(loc="lower right")
plt.show()

# Plot the ROC curve
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()

# keep retrying if the response is not n
cont = True
while (cont):
    demo()
    answer = input("\nDo you want to enter another sentence [Y/N]? ")
    if answer.lower() == "n":
        cont = False

print("End :)")
    
