import pandas as pd
import os.path as osp
from configs.general import EXPERIMENT_INFO_PATH, BASE_DIR
from utils.metrics import  load_experiments
from utils.plot import draw_stacked_bars

all_settings = []
best_settings_store = pd.DataFrame()
for noise_p in ["np=0.5", "np=0.4", "np=0.3", "np=0.2", "np=0.1"]:
  experiments = load_experiments(EXPERIMENT_INFO_PATH, index_col='index')
  experiments = experiments[(experiments['done'] == True) & ((experiments['np'] == noise_p))]
  classification_result = pd.read_csv('logs/classifier_result.csv', index_col='index')
  merged_df = pd.merge(experiments, classification_result, left_index=True, right_index=True)
  sorted_df = merged_df.sort_values(by='weighted_avg_f1', ascending=False)
  best_five_settings = sorted_df.head(5)
  # print('best_five_settings: ', best_five_settings.index)
  best_settings_store[noise_p] = best_five_settings.index.tolist()
  important_settings = best_five_settings[['model', 'dropout', 'optim', 'lr', 'lr_scheduler', 'init', 'transform']]
  all_settings.append(important_settings)

tr_map = {
    'default': 'Shallow Augmentation',
    'intermediate': 'Strong Augmentation',
}

# Map values in the 'City' column
all_settings = pd.concat(all_settings)
all_settings['transform'] = all_settings['transform'].map(tr_map)
best_settings_store.to_csv(osp.join(BASE_DIR, 'best_5_settings_normal.csv'))
# draw_stacked_bars(all_settings, title="Best classifier result component contribution")