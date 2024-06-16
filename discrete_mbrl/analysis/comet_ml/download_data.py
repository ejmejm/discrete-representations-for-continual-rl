# Source: https://docs.wandb.ai/guides/track/public-api-guide

import argparse
from collections import defaultdict

from comet_ml.api import API
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=str, default='[REDACTED]')
parser.add_argument('--project', type=str, default='discrete-representations-discrete_mbrl')
# # Options are run and table
# parser.add_argument('--sweeps', nargs='*', type=str, default=[])
# parser.add_argument('--tags', nargs='*', type=str, default=[])
parser.add_argument('--history_vars', nargs='*', type=str, default=None)
parser.add_argument('--params', nargs='*', type=str, default=None)
# parser.add_argument('--include_crashed', action='store_true')
# parser.set_defaults(include_crashed=False)


DEFAULT_METRICS = ['n_step', 'eval_policies',
        'random_state_distrib_kl_div', 'random_state_distrib_kl_div_mean',
        'goal_state_distrib_kl_div', 'goal_state_distrib_kl_div_mean',
        'explore_right_state_distrib_kl_div', 'explore_right_state_distrib_kl_div_mean']
DEFAULT_PARAMS = ['codebook_size', 'filter_size', 'eval_policy', 'env_name', 'ae_model_type']


def experiment_to_rows(experiment, metrics_names, param_names, index='step'):
  # API calls to get data
  metric_data = []
  for metric_name in metrics_names:
    new_metric_data = experiment.get_metrics(metric_name)
    metric_data.extend(new_metric_data)
  param_data = experiment.get_parameters_summary()

  # Gather parameter values
  experiment_key = experiment.id
  param_vals = {x['name']: x['valueCurrent'] \
                for x in param_data if x['name'] in param_names}
  param_vals.update({'experiment_key': experiment_key})

  # Coalesce metric data into rows
  unique_metrics = set()
  rows = defaultdict(dict)
  for entry in metric_data:
    unique_metrics.add(metric_name)
    index_val = entry[index]
    metric_name = entry['metricName']
    metric_value = entry['metricValue']

    rows[index_val][index] = index_val
    rows[index_val][metric_name] = metric_value

  rows = [dict(**row, **param_vals) for row in rows.values()]
  
  return rows


if __name__ == '__main__':
  args = parser.parse_args()

  with open('../../.comet.config', 'r') as f:
    api_key = f.read()
  api_key = api_key[api_key.find('=') + 1:]
  api = API(api_key)
  api.use_cache(True)

  print('Looking for experiments...')
  experiments = api.get_experiments(args.workspace, args.project)
  print(f'Found {len(experiments)} experiments')

  exp_keys = [exp.id for exp in experiments]

  metrics = args.history_vars if args.history_vars is not None else DEFAULT_METRICS
  params = args.params if args.params is not None else DEFAULT_PARAMS
  columns = metrics + params + ['experiment_key']

  print('Querying experiment data...')

  rows = []
  n_valid_runs = 0
  for experiment in tqdm(experiments):
    new_rows = experiment_to_rows(
      experiment, metrics, params, index='step')

    if len(new_rows) > 0:
      n_valid_runs += 1
      rows.extend(new_rows)

  print('Saving data to csv...')
  df = pd.DataFrame(rows)
  df.reset_index(drop=True, inplace=True)
  df.to_csv(f'data/{args.project}_data.csv')

  print(f'{n_valid_runs}/{len(experiments)} runs saved.')
  print(f'{len(rows)} rows saved.')