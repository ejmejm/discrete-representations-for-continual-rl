from io import BytesIO
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


def t_distrib(vals):
  """Compute the 95% confidence interval for the mean of a set of values."""
  return st.t.interval(
    alpha=0.95, df=len(vals)-1, loc=np.mean(vals), scale=st.sem(vals))

def t_ci(vals):
  """
  Compute the 95% confidence interval for the mean of a set of values,
  and return the size of one side of the interval.
  """
  return t_distrib(vals)[1] - np.mean(vals)

def bin_df(df, n_bins=10, bin_var='step', zero_start=True):
  min_val = 0 if zero_start else df[bin_var].min()
  bin_size = (df[bin_var].max() - df[bin_var].min()) / n_bins
  bins = np.arange(min_val, df[bin_var].max() + bin_size, bin_size) # define bins
  labels = bins[:-1] # define labels as bin starting values
  df[bin_var] = pd.cut(df[bin_var], bins=bins, labels=labels)

  # Identify numeric columns
  num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
  # Use dictionary comprehension to create an aggregation dictionary
  agg_dict = {col: 'mean' if col in num_cols else 'first' for col in df.columns}
  print(agg_dict)

  with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)
    df = df.groupby(['experiment_key', bin_var]).agg(agg_dict)
  df[bin_var] = pd.to_numeric(df[bin_var])

  return df.reset_index(drop=True)

# COLOR_PALETTE = [
#   '#0086DA', '#CA3720', '#2BAF2A', '#9923DC',
#   '#F5C031', '#A4A4A3', '#EC82CB'
# ]

COLOR_PALETTE = [
  '#0086DA', '#CA3720', '#2BAF2A', '#9923DC',
  '#e67e22', '#34495e', '#1abc9c'
]

DEFAULT_COLOR_IDS = {
  'vq-vae': 0,
  'discrete': 0,
  'one-hot': 0,
  'ae': 1,
  'continuous': 1,
  'vanilla ae': 1,
  'quantized': 1,
  'fta': 2,
  'fta ae': 2,
  'end-to-end': 3,
  'uniform': 5
}

def get_color_palette(classes=None, n=None):
  """
  With no arguments, return the default color palette.
  With a list of classes, return a dict mapping each class to a color.
  With an integer n, return a list of n colors.
  """
  
  if classes is not None:
    if len(classes) > len(COLOR_PALETTE):
      raise ValueError(
        f'Only {len(COLOR_PALETTE)} colors available, but {len(classes)} '
        'classes were requested.'
      )
    # First set ids for classes in DEFAULT_COLOR_IDS
    # then do the rest alphabetically
    
    color_map = {}
    for c in classes:
      if c.lower() in DEFAULT_COLOR_IDS:
        color_map[c] = DEFAULT_COLOR_IDS[c.lower()]

    remaining_ids = [i for i in range(len(COLOR_PALETTE)) \
      if i not in color_map.values()]
    for c in sorted(classes, key=lambda x: x.lower()):
      if c not in color_map:
        color_map[c] = remaining_ids.pop(0)

    color_map = {c: COLOR_PALETTE[i] for c, i in color_map.items()}
    
    return color_map

  elif n is not None:
    if n > len(COLOR_PALETTE):
      raise ValueError(
        f'Only {len(COLOR_PALETTE)} colors available, but {n} were requested.'
      )
    return COLOR_PALETTE[:n]
  
  return COLOR_PALETTE
  
def standardize_env_name(name):
  name = name.lower() \
    .replace('-rand', '') \
    .replace('-stochastic', '') \
    .replace('-v0', '') \
    .replace('-fullobs', '') \
    .replace('-6x6', '') \
    .replace('-', ' ') \
    .replace('_', ' ')
  return name.title()

def set_fig_labels(xlabel, ylabel, xsci=False, ysci=False, ax=None):
  """
  Set the labels for the x and y axes of the current figure.
  If style is 'sci', use scientific notation for the y axis.
  """
  if ax is None:
    ax = plt.gca()

  if xsci:
    ax.ticklabel_format(
      style='sci', axis='x', scilimits=(0,0), useMathText=True)
    plt.savefig(BytesIO())
    offset = ax.get_xaxis().get_offset_text()
    ax.set_xlabel(f'{xlabel} ({offset.get_text()})')
    offset.set_visible(False)
  else:
    ax.set_xlabel(xlabel)

  if ysci:
    ax.ticklabel_format(
      style='sci', axis='y', scilimits=(0,0), useMathText=True)
    plt.savefig(BytesIO())
    offset = ax.get_yaxis().get_offset_text()
    ax.set_ylabel(f'{ylabel} ({offset.get_text()})')
    offset.set_visible(False)
  else:
    ax.set_ylabel(ylabel)

def set_matplotlib_style(style='default'):
  plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False
  })

  if style == 'default':
    plt.rcParams.update({
      'figure.figsize': (5, 3),
      'axes.labelsize': 13,
      'axes.titlesize': 14,
      'axes.titlepad': 10,
      'xtick.labelsize': 11,
      'ytick.labelsize': 11,
      'lines.linewidth': 1.5
    })
  elif style == '4-row':
    # Make plots 1.8x larger when there are 4 in a row
    plt.rcParams.update({
      'figure.figsize': (5, 4),
      'axes.labelsize': 23,
      'axes.titlesize': 25,
      'axes.titlepad': 21,
      'xtick.labelsize': 20,
      'ytick.labelsize': 20,
      'lines.linewidth': 2.7
    })
  else:
    raise ValueError(f'Unknown style {style}')


def save_fig_versions(name, dir='../figures/svg/', type='svg', **kwargs):
  """
  Save the current figure with and with the legend.
  """
  plt.gca().get_legend().set_visible(False)
  plt.savefig(
    f'{os.path.join(dir, name)}_nl.{type}', bbox_inches='tight', dpi=400, **kwargs)
  
  plt.gca().get_legend().set_visible(True)
  plt.savefig(
    f'{os.path.join(dir, name)}.{type}', bbox_inches='tight', dpi=400, **kwargs)
  
  if type == 'svg':
    save_fig_versions(name, '../figures/png/', type='png', **kwargs)