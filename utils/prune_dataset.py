import pickle
import os 
import pandas as pd
import os.path as osp
import glob
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from configs.general import EXPERIMENT_INFO_PATH, PRUNNING_RESULT_DIR, CLASSIFIER_MODEL_DIR, METRICS_PER_SAMPLE_DIR
from sklearn.model_selection import train_test_split



metrics_name = [
   'train-correctness_a', 'validation-correctness', 'mean-train-entropy_a',
   'std-train-entropy_a', 'mean-validation-entropy',
   'std-validation-entropy', 'train-id2m_a', 'validation-id2m',
   'mean-train-loss_a', 'std-train-loss_a', 'mean-validation-loss',
   'std-validation-loss', 'mean-train-top_proba_a',
   'std-train-top_proba_a', 'mean-validation-top_proba',
   'std-validation-top_proba', 'train-aum_a', 'validation-aum',
   'train-correctness_na', 'mean-train-entropy_na', 'std-train-entropy_na',
   'train-id2m_na', 'mean-train-loss_na', 'std-train-loss_na',
   'mean-train-top_proba_na', 'std-train-top_proba_na', 'train-aum_na'
]

def load_classifiers():
   classifiers = []
   glob_regex = osp.join(CLASSIFIER_MODEL_DIR, "*.pkl")   
   models_path = glob.glob(glob_regex)

   for path in models_path:
      with open(path, 'rb') as file:
         loaded_model = pickle.load(file)
      classifiers.append(loaded_model)
   return classifiers


def prune_based_on_classifier(experiments_index):
   os.makedirs(PRUNNING_RESULT_DIR, exist_ok=True)
   experiments = pd.read_csv(path=EXPERIMENT_INFO_PATH, index_col='index')
   target_expetiments = experiments.loc[experiments_index]

   clfs = load_classifiers()
   for _, row in target_expetiments.iterrows():
      dataset_name = row['dataset']
      valid_dataset_name = dataset_name.split('_')[0] if len(dataset_name) > 1 else dataset_name

      metrics = pd.read_pickle(osp.join(METRICS_PER_SAMPLE_DIR, f"metrics_per_sample_{row.index}"), compression='xz')
      df = pd.DataFrame()
      df['sample'] = metrics['sample']
      for i, clf in enumerate(clfs):
         X_df = metrics[metrics_name]
         pred_result = clf.predict(X_df)
         df[f"clf_{i}"] = np.array(pred_result)

      df['zero_count'] = df.apply(lambda row: (row == 0).sum(), axis=1)

      path = osp.join(PRUNNING_RESULT_DIR, f"{row.index}-{valid_dataset_name}.pkl")
      df.to_pickle(path, compression='xz')
      print(np.sum(df['agg_clf'].to_numpy() == 1) / 50000)


if __name__ == "__main__":
   experiments_index = []
   # experiments_index = ['9bf756402e8','d5a01a77f63','846ec41c22e','f732ee88030','89997809c70']
   prune_based_on_classifier(experiments_index)