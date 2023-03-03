import logging
import os
import queue
import shutil
import threading

import yaml
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example, ModelInputExample
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

from ludwig.api_annotations import DeveloperAPI, PublicAPI
from ludwig.callbacks import Callback
from ludwig.constants import TRAINER
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, TRAIN_SET_METADATA_FILE_NAME
from ludwig.types import TrainingSetMetadataDict
from ludwig.utils.data_utils import chunk_dict, flatten_dict, load_json, save_json, to_json_dict
from ludwig.utils.package_utils import LazyLoader

mlflow = LazyLoader("mlflow", globals(), "mlflow")

logger = logging.getLogger(__name__)

FLAVOR_NAME = "ludwig"


def _get_runs(experiment_id: str):
    return mlflow.tracking.client.MlflowClient().search_runs([experiment_id])


@DeveloperAPI
def get_or_create_experiment_id(experiment_name, artifact_uri: str = None):
    """Gets experiment id from mlflow."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name=experiment_name, artifact_location=artifact_uri)


@PublicAPI
class MlflowCallback(Callback):
    def __init__(self, tracking_uri=None, log_artifacts: bool = True):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = mlflow.get_tracking_uri()

        active_run = mlflow.active_run()
        if active_run is not None:
            # Use experiment already set in the current environment
            self.run = active_run
            self.experiment_id = self.run.info.experiment_id
            self.experiment_name = mlflow.get_experiment(self.experiment_id).name
            self.external_run = True
        else:
            # Will create an experiment at training time
            self.run = None
            self.experiment_id = None
            self.experiment_name = None
            self.external_run = False

        self.run_ended = False
        self.training_set_metadata = None
        self.config = None
        self.save_in_background = True
        self.save_fn = None
        self.save_thread = None
        self.log_artifacts = log_artifacts

    def get_experiment_id(self, experiment_name):
        return get_or_create_experiment_id(experiment_name)

    def on_preprocess_end(
        self,
        training_set: "Dataset",  # noqa
        validation_set: "Dataset",  # noqa
        test_set: "Dataset",  # noqa
        training_set_metadata: TrainingSetMetadataDict,
    ):
        self.training_set_metadata = training_set_metadata

    def on_hyperopt_init(self, experiment_name):
        self.experiment_id = self.get_experiment_id(experiment_name)
        self.experiment_name = experiment_name

    def on_hyperopt_trial_start(self, parameters):
        # Filter out mlflow params like tracking URI, experiment ID, etc.
        params = {k: v for k, v in parameters.items() if k != "mlflow"}
        self._log_params({"hparam": params})

        # TODO(travis): figure out a good way to support this. The problem with
        # saving artifacts in the background with hyperopt is early stopping. If
        # the scheduler decides to terminate a process, then currently there's no
        # mechanism to detect this a "flush" the queue of pending writes before
        # stopping. Should work with Ray Tune team to come up with a solution.
        self.save_in_background = False

    def on_train_init(self, base_config, experiment_name, output_directory, resume_directory, **kwargs):
        # Experiment may already have been set during hyperopt init, in
        # which case we don't want to create a new experiment / run, as
        # this should be handled by the executor.
        if self.experiment_id is None:
            mlflow.end_run()
            self.experiment_id = self.get_experiment_id(experiment_name)
            self.experiment_name = experiment_name

        active_run = mlflow.active_run()
        if active_run is not None:
            # Currently active run started by Ray Tune MLflow mixin or external run
            self.run = active_run
        else:
            run_id = None
            if resume_directory is not None:
                previous_runs = _get_runs(self.experiment_id)
                if len(previous_runs) > 0:
                    run_id = previous_runs[0].info.run_id
            if run_id is not None:
                self.run = mlflow.start_run(run_id=run_id)
            else:
                run_name = os.path.basename(output_directory)
                self.run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)

        self.log_config(base_config)

    def log_config(self, config):
        if self.log_artifacts:
            mlflow.log_dict(to_json_dict(config), "config.yaml")

    def on_train_start(self, config, **kwargs):
        self.config = config
        self._log_params({TRAINER: config[TRAINER]})

    def on_train_end(self, output_directory):
        if self.log_artifacts:
            _log_artifacts(output_directory)
        if self.run is not None and not self.external_run:
            # Only end runs managed internally to this callback
            mlflow.end_run()
            self.run_ended = True

    def on_trainer_train_setup(self, trainer, save_path, is_coordinator):
        if not is_coordinator:
            return

        # When running on a remote worker, the model metadata files will only have been
        # saved to the driver process, so re-save it here before uploading.
        training_set_metadata_path = os.path.join(save_path, TRAIN_SET_METADATA_FILE_NAME)
        if not os.path.exists(training_set_metadata_path):
            save_json(training_set_metadata_path, self.training_set_metadata)

        model_hyperparameters_path = os.path.join(save_path, MODEL_HYPERPARAMETERS_FILE_NAME)
        if not os.path.exists(model_hyperparameters_path):
            save_json(model_hyperparameters_path, self.config)

        if self.save_in_background:
            save_queue = queue.Queue()
            self.save_fn = lambda args: save_queue.put(args)
            self.save_thread = threading.Thread(target=_log_mlflow_loop, args=(save_queue, self.log_artifacts))
            self.save_thread.start()
        else:
            self.save_fn = lambda args: _log_mlflow(*args, self.log_artifacts)

    def on_eval_end(self, trainer, progress_tracker, save_path):
        self.save_fn((progress_tracker.log_metrics(), progress_tracker.steps, save_path, True))

    def on_trainer_train_teardown(self, trainer, progress_tracker, save_path, is_coordinator):
        if is_coordinator:
            self.save_fn((progress_tracker.log_metrics(), progress_tracker.steps, save_path, False))
            if self.save_thread is not None:
                self.save_thread.join()

    def on_visualize_figure(self, fig):
        # TODO: need to also include a filename for this figure
        # mlflow.log_figure(fig)
        pass

    def prepare_ray_tune(self, train_fn, tune_config, tune_callbacks):
        from ray.tune.integration.mlflow import mlflow_mixin

        return mlflow_mixin(train_fn), {
            **tune_config,
            "mlflow": {
                "experiment_id": self.experiment_id,
                "experiment_name": self.experiment_name,
                "tracking_uri": mlflow.get_tracking_uri(),
            },
        }

    def _log_params(self, params):
        flat_params = flatten_dict(params)
        for chunk in chunk_dict(flat_params, chunk_size=100):
            mlflow.log_params(chunk)

    def __setstate__(self, d):
        self.__dict__ = d
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.run and not self.run_ended:
            # Run has already been set, but may not be active due to training workers running in a separate
            # process, so resume the run
            mlflow.end_run()
            self.run = mlflow.start_run(run_id=self.run.info.run_id, experiment_id=self.run.info.experiment_id)


def _log_mlflow_loop(q: queue.Queue, log_artifacts: bool = True):
    should_continue = True
    while should_continue:
        elem = q.get()
        log_metrics, steps, save_path, should_continue = elem
        mlflow.log_metrics(log_metrics, step=steps)

        if not q.empty():
            # in other words, don't bother saving the model artifacts
            # if we're about to do it again
            continue

        if log_artifacts:
            _log_model(save_path)


def _log_mlflow(log_metrics, steps, save_path, should_continue, log_artifacts: bool = True):
    mlflow.log_metrics(log_metrics, step=steps)
    if log_artifacts:
        _log_model(save_path)


def _log_artifacts(output_directory):
    for fname in os.listdir(output_directory):
        lpath = os.path.join(output_directory, fname)
        if fname == "model":
            _log_model(lpath)
        else:
            mlflow.log_artifact(lpath)


def _log_model(lpath):
    log_saved_model(lpath)


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import ludwig

    # Ludwig is not yet available via the default conda channels, so we install it via pip
    return _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[f"ludwig=={ludwig.__version__}"],
        additional_conda_channels=None,
    )


def save_model(
    ludwig_model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
):
    """Save a Ludwig model to a path on the local file system.

    :param ludwig_model: Ludwig model (an instance of `ludwig.api.LudwigModel`_) to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pip': [
                                    'ludwig==0.4.0'
                                ]
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        train = df.drop_column("target_label")
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    """
    import ludwig

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException(f"Path '{path}' already exists")
    model_data_subpath = "model"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # Save the Ludwig model
    ludwig_model.save(model_data_path)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env) as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="ludwig.contribs.mlflow",
        data=model_data_subpath,
        env=conda_env_subpath,
    )

    schema_keys = {"name", "column", "type"}
    config = ludwig_model.config

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        ludwig_version=ludwig.__version__,
        ludwig_schema={
            "input_features": [
                {k: v for k, v in feature.items() if k in schema_keys} for feature in config["input_features"]
            ],
            "output_features": [
                {k: v for k, v in feature.items() if k in schema_keys} for feature in config["output_features"]
            ],
        },
        data=model_data_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(
    ludwig_model,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
):
    """Log a Ludwig model as an MLflow artifact for the current run.

    :param ludwig_model: Ludwig model (an instance of `ludwig.api.LudwigModel`_) to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pip': [
                                    'ludwig==0.4.0'
                                ]
                            ]
                        }
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        train = df.drop_column("target_label")
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    """
    import ludwig

    Model.log(
        artifact_path=artifact_path,
        flavor=ludwig.contribs.mlflow,
        registered_model_name=registered_model_name,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        ludwig_model=ludwig_model,
    )


def _load_model(path):
    from ludwig.api import LudwigModel

    return LudwigModel.load(path, backend="local")


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``ludwig`` flavor.
    """
    return _LudwigModelWrapper(_load_model(path))


def load_model(model_uri):
    """Load a Ludwig model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.

    :return: A Ludwig model (an instance of `ludwig.api.LudwigModel`_).
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    lgb_model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.lgb"))
    return _load_model(path=lgb_model_file_path)


class _LudwigModelWrapper:
    def __init__(self, ludwig_model):
        self.ludwig_model = ludwig_model

    def predict(self, dataframe):
        pred_df, _ = self.ludwig_model.predict(dataframe)
        return pred_df


def export_model(model_path, output_path, registered_model_name=None):
    if registered_model_name:
        if not model_path.startswith("runs:/") or output_path is not None:
            # No run specified, so in order to register the model in mlflow, we need
            # to create a new run and upload the model as an artifact first
            output_path = output_path or "model"
            log_model(
                _CopyModel(model_path),
                artifact_path=output_path,
                registered_model_name=registered_model_name,
            )
        else:
            # Registering a model from an artifact of an existing run
            mlflow.register_model(
                model_path,
                registered_model_name,
            )
    else:
        # No model name means we only want to save the model locally
        save_model(
            _CopyModel(model_path),
            path=output_path,
        )


@DeveloperAPI
def log_saved_model(lpath):
    """Log a saved Ludwig model as an MLflow artifact.

    :param lpath: Path to saved Ludwig model.
    """
    log_model(
        _CopyModel(lpath),
        artifact_path="model",
    )


class _CopyModel:
    """Get model data without requiring us to read the model weights into memory."""

    def __init__(self, lpath):
        self.lpath = lpath

    def save(self, path):
        shutil.copytree(self.lpath, path)

    @property
    def config(self):
        return load_json(os.path.join(self.lpath, MODEL_HYPERPARAMETERS_FILE_NAME))
