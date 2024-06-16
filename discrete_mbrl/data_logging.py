from argparse import Namespace
import io
import logging
import os
import random
import string
import sys
import tempfile

import numpy as np

from training_helpers import process_args


wandb = None
comet_ml = None
experiment_module_name = None


def update_params(args):
  if args.wandb:
    wandb.config.update(args, allow_val_change=True)
  elif args.comet_ml:
    experiment = comet_ml.get_global_experiment()
    experiment.log_parameters(args)

def get_comet_sweep_id():
  return os.environ.get('COMET_OPTIMIZER_ID', None)

def init_experiment(project, args):
  if args and args.wandb:
    import wandb
    global wandb
    
    wandb.init(project=project, config=args, tags=args.tags,
      settings=wandb.Settings(start_method='thread'), allow_val_change=True)
    args = wandb.config

  comet_sweep_id = get_comet_sweep_id()
  if comet_sweep_id or (args and args.comet_ml):
    # import os
    # os.environ['COMET_LOGGING_FILE_LEVEL'] = 'DEBUG'
    # os.environ['COMET_LOGGING_FILE'] = './comet.log'

    import comet_ml
    global comet_ml

    global experiment_module_name # Used for capturing log output

    if experiment_module_name is None:
      from comet_ml import experiment
      experiment_module_name = experiment.__name__

    log_capture_string = io.StringIO()
    stream_handler = logging.StreamHandler(log_capture_string)
    stream_handler.setLevel(logging.ERROR)
    logger = logging.getLogger(experiment_module_name)
    logger.addHandler(stream_handler)

    if comet_sweep_id:
      opt = comet_ml.Optimizer(comet_sweep_id, verbose=0)
      project = opt.status()['parameters'].get('sweep_project')
      if project is not None:
        project = project['values'][0]

      experiment = opt.next(
        project_name=project, workspace='[REDACTED]')
      
      error_log = log_capture_string.getvalue()

      if 'run will not be logged' in error_log.lower():
        print('Error captured in experiment setup!')
        if 'was already uploaded' in error_log.lower():
          print('Creating an `ExistingExperiment` after error')
          new_experiment = comet_ml.ExistingExperiment(
            project_name=project, workspace='[REDACTED]',
            experiment_key=experiment.get_key())
        else:
          print('Creating an `OfflineExperiment` after error')
          new_experiment = comet_ml.OfflineExperiment(
            project_name=project, workspace='[REDACTED]')

        # Get parameters from original experiment
        api = comet_ml.api.API()
        api_exp = api.get_experiment_by_id(experiment.id)
        param_summary = api_exp.get_parameters_summary()
        params = {x['name']: x['valueCurrent'] for x in param_summary}
        new_experiment.params = params
        experiment = new_experiment

      comet_ml.config.set_global_experiment(experiment)
      if experiment is None:
        print('No more experiments to run in sweep!')
        sys.exit(0)
      # args are an argparse.Namespace object
      # Combine new params dict with old args
      comet_args = Namespace(**{**vars(args), **experiment.params})
      comet_args = process_args(comet_args)
      experiment.log_parameters(vars(comet_args))
      if not args.wandb:
        args = comet_args

      # Pretty print chosen args for sweep
      print('Sweep args:')
      for k, v in experiment.params.items():
        print(f'  {k}: {v}')

      if args.tags is not None:
        experiment.add_tag(args.tags)

    else:
      experiment = comet_ml.Experiment(
        project_name=project, workspace='[REDACTED]')
      error_log = log_capture_string.getvalue()

      if 'run will not be logged' in error_log.lower():
        print('Error captured in experiment setup!')
        if 'was already uploaded' in error_log.lower():
          print('Creating an `ExistingExperiment` after error')
          experiment = comet_ml.ExistingExperiment(
            project_name=project, workspace='[REDACTED]')
        else:
          print('Creating an `OfflineExperiment` after error')
          experiment = comet_ml.OfflineExperiment(
            project_name=project, workspace='[REDACTED]')

      comet_ml.config.set_global_experiment(experiment)
      experiment.log_parameters(args)
      if args.tags is not None:
        experiment.add_tag(args.tags)

    log_capture_string.close()
    
  return args

def import_logger(args):
  if args.wandb:
    import wandb
    global wandb
  elif args.comet_ml:
    import comet_ml
    global comet_ml

def log_metrics(metrics, args, prefix=None, step=None):
  if args.wandb:
    prefix = prefix + '/' if prefix else ''
    wandb.log({f'{prefix}{k}': v for k, v in metrics.items()}) #, step=step)
  if args.comet_ml:
    experiment = comet_ml.get_global_experiment()
    experiment.log_metrics(metrics, prefix=prefix, step=step)

def log_images(images, args, prefix=None, step=None):
  prefix = prefix + '/' if prefix else ''
  if args.wandb:
    formatted_imgs = {f'{prefix}{k}': [wandb.Image(img) for img in v] \
      for k, v in images.items()}
    wandb.log(formatted_imgs) #, step=step)
  if args.comet_ml:
    experiment = comet_ml.get_global_experiment()
    for k, v in images.items():
      for image in v:
        experiment.log_image(image, name=f'{prefix}{k}', step=step)

def log_figures(figures, args, prefix=None, step=None):
  prefix = prefix + '/' if prefix else ''
  if args.wandb:
    wandb.log({f'{prefix}{k}': [wandb.Image(figure) for figure in v] \
      for k, v in figures.items()}) #, step=step)
  if args.comet_ml:
    experiment = comet_ml.get_global_experiment()
    for k, v in figures.items():
      for figure in v:
        experiment.log_figure(f'{prefix}{k}', figure, step=step)

def log_videos(videos, args, prefix=None, step=None):
  prefix = prefix + '/' if prefix else ''
  if args.wandb:
    formatted_vids = {
      f'{prefix}{k}': [wandb.Video(frames, fps=4, format='gif') for frames in v] \
      for k, v in videos.items()
    }
    wandb.log(formatted_vids) #, step=step)
  if args.comet_ml:
    experiment = comet_ml.get_global_experiment()
    for k, v in videos.items():
      for frames in v:
        image = np.concatenate(frames, axis=1)
        experiment.log_image(
          image, name=f'{prefix}{k}', step=step, image_channels='first')

def finish_experiment(args):
  if args.wandb:
    wandb.finish()
  if args.comet_ml:
    experiment = comet_ml.get_global_experiment()
    experiment.end()

def track_model(model, args):
  if args.wandb:
    wandb.watch(model)

def log_np_array(arr, name, args):
  if args.wandb:
    artifact = wandb.Artifact(name, type='np_array')
    # Generate a temporary file to write the array to
    with tempfile.NamedTemporaryFile() as f:
      np.save(f, arr)
      artifact.add_file(f.name)
      wandb.log_artifact(artifact)

  if args.comet_ml:
    experiment = comet_ml.get_global_experiment()

    # Generate a random string to use as the temporary file name
    tmp_file_name = ''.join(random.choices(
      string.ascii_uppercase + string.digits, k=10))
    tmp_file_name = f'{tmp_file_name}.npy'

    # Save the array to the temporary file
    with open(tmp_file_name, 'wb') as f:
      np.save(f, arr)

    # Print size of the file in KB
    print(f'\nSize of {name}: {os.path.getsize(tmp_file_name) / 1000:.2f} KB')

    def success_callback(*args, **kwargs):
      print(f'\nSuccessfully uploaded {name}')
      print('args:', args)
      print('kwargs:', kwargs)
      # Delete the temporary file
      os.remove(tmp_file_name)

    def failure_callback(*args, **kwargs):
      print(f'\nFailed to upload {name}')
      print('args:', args)
      print('kwargs:', kwargs)
      # Delete the temporary file
      os.remove(tmp_file_name)

    experiment._log_asset(
      tmp_file_name, file_name=f'{name}.npy',
      on_asset_upload=success_callback,
      on_failed_asset_upload=failure_callback)
