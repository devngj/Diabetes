from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

dir_path = '/content/drive/MyDrive'
drug_se_vector = np.load(dir_path + '/drug_se_vector_dict.npy', allow_pickle=True)
id2num = np.load(dir_path + '/id2num.npy', allow_pickle=True)
drug_indications = np.load(dir_path + '/drug_indications.npy', allow_pickle=True)
t2d_drugs = np.load(dir_path + '/t2d_drugs.npy', allow_pickle=True)

len(t2d_drugs.item())

drug_names = pd.read_csv(dir_path + '/drug_names.tsv', delimiter='\t')
symptoms = pd.read_csv(dir_path + '/symptoms.tsv', delimiter='\t')

answer_list = []
for i in drug_se_vector.item().keys():
    if i in np.atleast_1d(t2d_drugs)[0]:
        answer_list.append(i)

len(answer_list)

labels = []
for k in drug_se_vector.item().keys():
    if k in answer_list:
        labels.append(1)
    else:
        labels.append(0)

print(len(answer_list) == sum(labels))
print(len(drug_se_vector.item().values()) == len(labels))

vals = []
for i in drug_se_vector.item(0).values():
    vals.append(np.array(i))

vals = np.array(vals)

vals

"""# Existing methods"""

indices = np.arange(len(vals))
(
    X_train,
    X_test,
    y_train,
    y_test,
    indices_train,
    indices_test,
) = train_test_split(vals, labels, indices, test_size=0.2)

from pandas.core.reshape.melt import lreshape
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve , roc_auc_score , f1_score , classification_report
import matplotlib.pyplot as plt
from sklearn import metrics

RFclf = RandomForestClassifier().fit(X_train, y_train)
SVclf = SVC(probability=True).fit(X_train, y_train)
LRclf = LogisticRegression().fit(X_train, y_train)
MLPclf = MLPClassifier().fit(X_train, y_train)

from tqdm import tqdm
RFauc = []
RFprc = []
SVauc = []
SVprc = []
LRauc = []
LRprc = []
MLPauc = []
MLPprc = []

for _ in tqdm(range(100)):
    indices = np.arange(len(vals))
    (
        X_train,
        X_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
    ) = train_test_split(vals, labels, indices, test_size=0.2)
    RFclf = RandomForestClassifier().fit(X_train, y_train)
    SVclf = SVC(probability=True).fit(X_train, y_train)
    LRclf = LogisticRegression().fit(X_train, y_train)
    MLPclf = MLPClassifier().fit(X_train, y_train)

    RFauc.append(metrics.roc_auc_score(y_test, RFclf.predict(X_test)))
    RFprc.append(metrics.f1_score(y_test, RFclf.predict(X_test)))
    SVauc.append(metrics.roc_auc_score(y_test, SVclf.predict(X_test)))
    SVprc.append(metrics.f1_score(y_test, SVclf.predict(X_test)))
    LRauc.append(metrics.roc_auc_score(y_test, LRclf.predict(X_test)))
    LRprc.append(metrics.f1_score(y_test, LRclf.predict(X_test)))
    MLPauc.append(metrics.roc_auc_score(y_test, MLPclf.predict(X_test)))
    MLPprc.append(metrics.f1_score(y_test, MLPclf.predict(X_test)))

df = pd.DataFrame(columns=['classifier_type', 'AUC_score', 'F1_score'])
df['classifier_type'] = ['RF']*100 + ['SV']*100 + ['LR']*100 +['MLP']*100
df['AUC_score'] = RFauc + SVauc + LRauc + MLPauc
df['F1_score'] = RFprc + SVprc + LRprc + MLPprc

import seaborn as sns

sns.catplot(x='classifier_type', y='AUC_score',
                data=df, kind="box",
            height=6, aspect=1.3).set(title='AUC score')

sns.catplot(x='classifier_type', y='F1_score',
                data=df, kind="box",
            height=6, aspect=1.3).set(title='F1 score')

for clf in [RFclf, SVclf, LRclf, MLPclf]:
    pred = clf.predict(X_test)
    print(classification_report(pred, y_test))

probs = pd.DataFrame(RFclf.predict_proba(X_test))

probs[1].loc[probs[1] > 0.5].index

thres = 0.9
indices_test[probs[1].loc[probs[1] > thres].index] # <-- candidate drugs

repurposable = []
for i in range(100):
    indices = np.arange(len(vals))
    (
        X_train,
        X_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
    ) = train_test_split(vals, labels, indices, test_size=0.2)
    RFclf = RandomForestClassifier().fit(X_train, y_train)
    probs = pd.DataFrame(RFclf.predict_proba(X_test))
    thres = 0.7
    if len(indices_test[probs[1].loc[probs[1] > thres].index]) != 0:
        repurposable.append(indices_test[probs[1].loc[probs[1] > thres].index][0]) # <-- candidate drugs

idx, count = np.unique(repurposable,return_counts=True)
result = []
for i in range(len(idx)):
    result.append((count[i], idx[i]))

import matplotlib.pyplot as plt
from sklearn import metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, RFclf.predict(X_test))
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

precision, recall, thresholds = metrics.precision_recall_curve(y_test, RFclf.predict(X_test))
auprc = metrics.average_precision_score(y_test, RFclf.predict(X_test))

plt.title('Precision-Recall Curve')
plt.plot(recall, precision, 'b', label = 'AUPRC = %0.2f' % auprc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
plt.show()

sorted(result, reverse=True)

list(drug_se_vector.item().keys())[992]



"""# Novel method"""

import networkx as nx
import matplotlib.pyplot as plt

def overlap_percentage(vector1, vector2):
    intersection = sum([i & j for i, j in zip(vector1, vector2)])
    union = sum([i | j for i, j in zip(vector1, vector2)])

    if union == 0:
        return 0  # Prevent division by zero if there are no side effects

    return intersection / union


def create_network(drug_dict, threshold=0.5):
    G = nx.Graph()

    drugs = list(drug_dict.keys())
    for i in range(len(drugs)):
        for j in range(i+1, len(drugs)):
            drug1 = drugs[i]
            drug2 = drugs[j]
            overlap = overlap_percentage(drug_dict[drug1], drug_dict[drug2])
            if overlap >= threshold:
                G.add_edge(drug1, drug2, weight=overlap)

    return G

# Example usage:
drug_dict = drug_se_vector.item()

'''for element in drug_se_vector:
    drug_dict.update(element)'''

G = create_network(drug_dict, threshold=0.0)

import random
from collections import Counter
from tqdm import tqdm

def weighted_random_walk(graph, start_node, num_steps):
    current_node = start_node
    for _ in range(num_steps):
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            # Get the weights of the edges to neighbors
            weights = [graph[current_node][neighbor].get('weight', 1) for neighbor in neighbors]
            # Normalize the weights
            total_weight = sum(weights)
            probabilities = [weight / total_weight for weight in weights]
            # Choose the next node based on the weights
            current_node = random.choices(neighbors, probabilities)[0]
        else:
            break
    return current_node


def propagation_effect(graph, start_node, num_walks, num_steps):
    end_nodes = [weighted_random_walk(graph, start_node, num_steps) for _ in range(num_walks)]
    propagation_counts = Counter(end_nodes)

    # Creating a vector of counts ordered by node labels
    nodes = sorted(graph.nodes())
    propagation_vector = [propagation_counts[node]/(num_walks * num_steps) if node in propagation_counts else 0 for node in nodes]

    return propagation_vector

effect_vectors = []
for node in tqdm(sorted(G.nodes)):
    start_node = node
    num_walks = 100
    num_steps = 5
    propagation = propagation_effect(G, start_node, num_walks, num_steps)
    effect_vectors.append(propagation)

np.save(dir_path + '/RW_result.npy', effect_vectors, allow_pickle=True)

# evecs = np.load(dir_path + '/RW_result.npy', allow_pickle=True)
evecs = effect_vectors

new_labels = []
for k in sorted(G.nodes):
    if k in answer_list:
        new_labels.append(1)
    else:
        new_labels.append(0)

print(len(answer_list) == sum(new_labels))
print(len(G.nodes) == len(new_labels))

indices = np.arange(len(evecs))
(
    X_train,
    X_test,
    y_train,
    y_test,
    indices_train,
    indices_test,
) = train_test_split(evecs, new_labels, indices, test_size=0.2)

from pandas.core.reshape.melt import lreshape
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve , roc_auc_score , f1_score , classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics

from tqdm import tqdm
import warnings

RFauc = []
RFprc = []
SVauc = []
SVprc = []
LRauc = []
LRprc = []
MLPauc = []
MLPprc = []
XGBauc = []
XGBprc = []


with warnings.catch_warnings(record=True) as caught_warnings:
    for _ in tqdm(range(100)):
        indices = np.arange(len(evecs))
        (
            X_train,
            X_test,
            y_train,
            y_test,
            indices_train,
            indices_test,
        ) = train_test_split(evecs, new_labels, indices, test_size=0.2)

        RFclf = RandomForestClassifier().fit(X_train, y_train)
        SVclf = SVC(probability=True).fit(X_train, y_train)
        LRclf = LogisticRegression().fit(X_train, y_train)
        MLPclf = MLPClassifier().fit(X_train, y_train)
        XGBclf = xgb.XGBClassifier().fit(X_train, y_train)

        RFauc.append(metrics.roc_auc_score(y_test, RFclf.predict(X_test)))
        RFprc.append(metrics.f1_score(y_test, RFclf.predict(X_test)))
        SVauc.append(metrics.roc_auc_score(y_test, SVclf.predict(X_test)))
        SVprc.append(metrics.f1_score(y_test, SVclf.predict(X_test)))
        LRauc.append(metrics.roc_auc_score(y_test, LRclf.predict(X_test)))
        LRprc.append(metrics.f1_score(y_test, LRclf.predict(X_test)))
        MLPauc.append(metrics.roc_auc_score(y_test, MLPclf.predict(X_test)))
        MLPprc.append(metrics.f1_score(y_test, MLPclf.predict(X_test)))
        XGBauc.append(metrics.roc_auc_score(y_test, XGBclf.predict(X_test)))
        XGBprc.append(metrics.f1_score(y_test, XGBclf.predict(X_test)))

df = pd.DataFrame(columns=['classifier_type', 'AUC_score', 'F1_score'])
df['classifier_type'] = ['RF']*100 + ['SV']*100 + ['LR']*100 +['MLP']*100 + ['XGB']*100
df['AUC_score'] = RFauc + SVauc + LRauc + MLPauc + XGBauc
df['F1_score'] = RFprc + SVprc + LRprc + MLPprc + XGBprc

import seaborn as sns

sns.catplot(x='classifier_type', y='AUC_score',
                data=df, kind="box",
            height=6, aspect=1.3).set(title='AUC score')

sns.catplot(x='classifier_type', y='F1_score',
                data=df, kind="box",
            height=6, aspect=1.3).set(title='F1 score')

for clf in [RFclf, SVclf, LRclf, MLPclf, XGBclf]:
    pred = clf.predict(X_test)
    print(classification_report(pred, y_test))

filtered_indices = [index for index, label in zip(indices_train, y_train) if label == 1]

len([i for i in new_labels if i == 1])

def sum_columns(binary_arrays):
    return [sum(col) for col in zip(*binary_arrays)]

print(sum_columns(vals[filtered_indices]))
print(sum_columns(vals))

vals

a = sum_columns(vals[filtered_indices])
b = sum_columns(vals)
np.where(np.array(a) == 0)

b.count(208)

