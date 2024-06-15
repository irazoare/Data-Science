import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.utils import resample
from matplotlib.colors import ListedColormap

def bootstrap_resample(X, y):
    n_samples = 250
    indices = np.arange(n_samples)
    bootstrap_indices = resample(indices, replace=True, n_samples=n_samples, random_state=42)
    return X.iloc[bootstrap_indices], y[bootstrap_indices]

def create_pipeline():
    scaler = StandardScaler()
    pca = PCA(n_components=52)  # 52 components describe 95% variance in dataset
    svm = SVC(C=1.5, gamma="scale", probability=True, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('pca', pca),
        ('svm', svm)
    ])
    
    return pipeline

relevant_genes = ["PIK3CA", "RUNX1", "CDH1", "TP53", "TBX3", "PTEN", "FOXA1", "MAP3K1", "GATA3", "AKT1", 
                  "NBL1", "DCTD", "RB1", "SF3B1", "CBFB", "OR9A2", "NCOA3", "RBMX", "MAP2K4", "TROVE2", 
                  "NADK", "CASP8", "CTSS", "ACTL6B", "LGALS1", "KRAS", "KCNN3", "FBXW7", "LRIG2", "PIK3R1", 
                  "PARP4", "ZNF28", "HLA-DRB1", "ERBB2", "ZMYM3", "RAB42", "CTCF", "ATAD2", "CDKN1B", "GRIA2", 
                  "NCOR1", "HRNR", "GPRIN2", "PAX2", "ACTG1", "AQP12A", "PIK3C3", "MYB", "IRS4", "TBL1XR1", 
                  "RPGR", "CCNI", "ARID1A", "CD3EAP", "ADAMTS6", "OR2D2", "TMEM199", "MST1", "RHBG", "ZFP36L1", 
                  "TCP11", "CASZ1", "GAL3ST1", "FRMPD2", "GPS2", "ZNF362"]

genes = pd.read_csv("../TCGA/dataset.csv")
genes = genes.rename(columns={"Unnamed: 0": "Patient_ID"})

patient_info = pd.read_csv("../TCGA/outcome.csv")
patient_info = patient_info.rename(columns={"Unnamed: 0": "Patient_ID"})

missing_subtype_patients = patient_info[patient_info["BRCA_subtype"].isnull()]["Patient_ID"]
genes = genes[~genes["Patient_ID"].isin(missing_subtype_patients)]
patient_info = patient_info.dropna()

merged_data = pd.merge(genes, patient_info, on="Patient_ID")

relevant_genes_in_data = [gene for gene in relevant_genes if gene in merged_data.columns]
X = merged_data[relevant_genes_in_data]
y = merged_data["BRCA_subtype"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=2024, stratify=y_encoded)

X_train_resampled, y_train_resampled = bootstrap_resample(X_train, y_train) #Bootstrap resampling on training set

pipeline = create_pipeline()
pipeline.fit(X_train_resampled, y_train_resampled)

# Model evaluation (test set)
def weighted_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    tn = np.diag(cm)  
    fp = cm.sum(axis=0) - np.diag(cm)
    specificities = tn / (tn + fp)
    weights = np.bincount(y_true)
    return np.average(specificities, weights=weights)

y_pred = pipeline.predict(X_test)
y_score = pipeline.predict_proba(X_test)

initial_metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=1),
    'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
    'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
    'roc_auc': roc_auc_score(label_binarize(y_test, classes=np.unique(y_encoded)), y_score, average='weighted', multi_class='ovr'),
    'specificity_weighted': weighted_specificity(y_test, y_pred)
}


# Hyperparameter tuning with gridsearch
""" param_grid = {
    'svm__C': [0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
    'svm__gamma': [0.00005, 0.005, 0.5, 0.1, 1, 2, 'scale', 'auto']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_) """


print("Initial Metrics (before CV):")
for metric, value in initial_metrics.items():
    print(f"{metric.replace('_', ' ').title()}: {value:.3f}")

# Cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
scoring = {
    'accuracy': 'accuracy',
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=1),
    'recall_weighted': make_scorer(recall_score, average='weighted'),
    'f1_weighted': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='weighted'),
    'specificity_weighted': make_scorer(weighted_specificity)
}

cv_results = cross_validate(pipeline, X, y_encoded, cv=cv, scoring=scoring, return_train_score=False)
cv_metrics = {metric: (np.mean(values), np.std(values)) for metric, values in cv_results.items() if 'test_' in metric}

print("\nCross-Validation Metrics:")
for metric, (mean, std) in cv_metrics.items():
    metric_name = metric.replace('test_', '').replace('_', ' ').title()
    print(f"{metric_name}: {mean:.3f} ± {std:.3f}") 
  
# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('SVC Confusion Matrix')
plt.tight_layout()
plt.savefig("../graphs/Confusion_Matrix_bootstrapped.png")
#plt.show()

# ROC AUC
y_test_binarized = label_binarize(y_test, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(9, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVC ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("../graphs/ROC_AUC_bootstrapped.png")
#plt.show()  

#Decision boundary plot of first two PCA components
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_resampled_scaled)
X_test_pca = pca.transform(X_test_scaled)

svm = SVC(C=1.5, gamma='scale', kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_pca, y_train_resampled)

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

unique_labels = np.unique(y_train_resampled)
colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_labels)))
cmap = ListedColormap(colors)

plt.figure(figsize=(9, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_resampled, edgecolors='k', cmap=cmap)

handles, _ = scatter.legend_elements()
labels = label_encoder.inverse_transform(unique_labels)
plt.legend(handles, labels, title="Cancer Types")

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('SVC Decision Boundary Plot')
plt.tight_layout()
plt.savefig("../graphs/Decision_Boundary_250samples.png")
plt.show()
