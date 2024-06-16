import ast
import argparse
import copy
import os
import signal
import subprocess
import time

from comet_ml import Optimizer


# Create args
parser = argparse.ArgumentParser()

parser.add_argument('-s', '--sweep_id', default=None)
parser.add_argument('-c', '--count', type=int, default=1)
parser.add_argument('-p', '--config', type=str, nargs='*', default=None)


def run_sweep(sweep_id):
  if 'COMET_OPTIMIZER_ID' in os.environ:
    del os.environ['COMET_OPTIMIZER_ID']
  opt = Optimizer(sweep_id, verbose=0)
  config = opt.status()
  command = config['parameters']['sweep_command']['values'][0].split()
  
  environ = os.environ.copy()
  environ['COMET_OPTIMIZER_ID'] = opt.id

  # This is adapted from comet_optimize.py

  proc = subprocess.Popen(command, env=environ, stderr=subprocess.STDOUT)
  try:
    proc.wait()
    if proc.returncode != 0:
        print('There was an error running the script!')
        print('Exit code:', proc.returncode)
  except KeyboardInterrupt:
    proc.send_signal(signal.SIGINT)

    # Check that all subprocesses exit cleanly
    i = 0
    while i < 60:
      proc.poll()
      dead = proc.returncode is not None
      if dead:
        break

      i += 1
      time.sleep(1)

      # Timeout, hard-kill all the remaining subprocess
      if i >= 60:
        proc.poll()
        if proc.returncode is None:
          proc.kill()

  print()
  results = opt.status()
  for key in ['algorithm', 'status']:
    print('   ', '%s:' % key, results[key])
  if isinstance(results['endTime'], float) and \
     isinstance(results['startTime'], float):
    print(
      '   ',
      'time:',
      (results['endTime'] - results['startTime']) / 1000.0,
      'seconds',
    )

def create_sweep(config_path):
  with open(config_path, 'r') as f:
    config = ast.literal_eval(f.read())

  if 'project' in config:
    config['parameters']['sweep_project'] = config['project']
    del config['project']
  config['parameters']['sweep_command'] = config['command']
  del config['command']
  if 'name' in config:
    config['parameters']['sweep_name'] = config['name']

  final_configs = []
  config_stack = [config]
  while len(config_stack) > 0:
    tmp_config = config_stack.pop()
    for key, entry in tmp_config['parameters'].items():
      if isinstance(entry, dict) and 'dependents' in entry:
        # Make a copy of the tmp_config
        # Add one of the depent values as a value to the parameters
        # Add the copy to the stack
        n_vals = len(entry['values'])
        dependents = entry['dependents']
        for i in range(n_vals):
          new_config = copy.deepcopy(tmp_config)
          for dependent in dependents:
            new_config['parameters'][key]['values'] = [entry['values'][i]]

            new_config['parameters'][dependent] = {}
            new_config['parameters'][dependent]['type'] = \
              dependents[dependent]['type']
            new_config['parameters'][dependent]['values'] = \
              [dependents[dependent]['values'][i]]
          del new_config['parameters'][key]['dependents']
          config_stack.append(new_config)
        break
    else:
      final_configs.append(tmp_config)
  # print(len(final_configs))
  # import sys; sys.exit()

  if 'COMET_OPTIMIZER_ID' in os.environ:
    del os.environ['COMET_OPTIMIZER_ID']
  opts = []
  for config in final_configs:
    opt = Optimizer(config)
    opts.append(opt.id)
    print('Created sweep with id:\n', opt.id)
  return opts

if __name__ == '__main__':
  args = parser.parse_args()

  if args.config:
    config_ids = []
    config_names = []
    for config in args.config:
      # Get file name from path
      config_names.append(config.split('/')[-1])
      new_ids = create_sweep(config)
      config_ids.extend(new_ids)
    print('Created sweeps with ids:\n', ', '.join(config_ids))
    print('From configs:\n', ', '.join(config_names))

  if args.sweep_id is not None:
    if args.sweep_id == 'new':
      args.sweep_id = new_ids[-1]
    for _ in range(args.count):
      run_sweep(args.sweep_id)