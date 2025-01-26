import json
import logging
import os
import time
import traceback
from pathlib import Path

import pandas as pd

from src.config import ROOT_PATH
from src.lib.evaluate import get_evaluation_method
from src.lib.experiment import ExperimentConfig
from src.lib.interface import (
    Dataset,
    FeaturesPreprocessor,
    RegressorWithPredict,
    TrainerInterface,
)
from src.lib.metrics import save_metrics
from src.lib.save import get_saving_method
from src.lib.train import Run, RunParams, get_model_key, get_raw_dataset, get_trainer



class TrainingPipeline:
    def __init__(
        self,
        experiment_config: ExperimentConfig,
        trainer_interface: TrainerInterface,
    ):
        self.experiment_config = experiment_config
        self.trainer_interface = trainer_interface

    def load_features(self, run_values: tuple[str, str, str]) -> pd.DataFrame:
        (
            zone_key,
            _,
            _,
        ) = run_values  # Assumption that the features are the same across the forecast keys and subkeys
        return pd.read_parquet(ROOT_PATH / "data/data/features" / f"{zone_key}.parquet")

    def run(self, run_values: tuple[tuple[str, str, str], pd.DataFrame]):
        (zone_key, forecast_key, forecast_subkey), features = run_values

        model_key = get_model_key(zone_key, forecast_key, forecast_subkey)

        active_run = Run(
            experiment_id=self.experiment_config.id,
            experiment_run_datetime=self.experiment_config.run_datetime,
        )
        run_params = RunParams(
            zone_key=zone_key,
            forecast_key=forecast_key,
            forecast_subkey=forecast_subkey,
            model_key=model_key,
        )
        run_dir = active_run.local_path()
        os.makedirs(run_dir, exist_ok=True)

        try:
            X, y = get_raw_dataset(features, run_params)

            trainer = get_trainer(run_params)
            dataset = trainer.split_train_eval(X, y)

            model, preprocessor = self._train(
                active_run, run_params, dataset, trainer, run_dir=run_dir
            )
            metrics, _ = self._evaluate(
                active_run,
                run_params,
                model,
                preprocessor,
                dataset,
                trainer,
                run_dir
            )
            self._save(
                active_run,
                run_params,
                model,
                trainer,
                preprocessor,
                dataset,
                metrics,
                run_dir
            )
        except Exception as e:
            self._log_text(
                run_dir=run_dir,
                content="".join(traceback.format_tb(e.__traceback__)),
                artefact_filename="exception.txt",
            )
            logging.exception("Something went wrong when running the pipeline")
        active_run.close()

    def _train(
        self,
        active_run: Run,
        run_params: RunParams,
        dataset: Dataset,
        trainer: TrainerInterface,
        run_dir: Path,
    ) -> tuple[RegressorWithPredict, FeaturesPreprocessor]:
        """
        Train a model given a mlflow run, run params, dataset and trainer.

        Logs run_params, training data range.
        """
        logging.info(f"[Train] model {run_params}")

        try:
            model, preprocessor, X_train = trainer.train(dataset)

            train_params = {
                "train_data_start": X_train.index.min()[0].to_pydatetime().isoformat(),
                "train_data_end": X_train.index.max()[0].to_pydatetime().isoformat(),
            }
            self._log_batch_params(
                run_params,
                run_dir,
                train_params,
            )
        except Exception as e:
            self._log_text(
                run_dir,
                "".join(traceback.format_tb(e.__traceback__)),
                artefact_filename="training_exception.txt",
            )
            raise e

        return model, preprocessor

    def _evaluate(
        self,
        active_run: Run,
        run_params: RunParams,
        model: RegressorWithPredict,
        preprocessor: FeaturesPreprocessor,
        dataset: Dataset,
        trainer: TrainerInterface,
        run_dir: Path,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """
        Evaluate a model given a mlflow run, run params, model, preprocessor and dataset.

        Logs evaluation data range.
        """
        logging.info(f"[Eval] model {run_params}")
        evaluator = get_evaluation_method(trainer, run_params)

        try:
            metrics, dfs = evaluator(
                run_params.zone_key,
                run_params.forecast_key,
                run_params.forecast_subkey,
                run_params.model_key,
                model,
                preprocessor,
                dataset,
                run_dir,
            )
            eval_params = {
                "eval_data_start": dataset.X_test.index.min()[0]
                .to_pydatetime()
                .isoformat(),
                "eval_data_end": dataset.X_test.index.max()[0]
                .to_pydatetime()
                .isoformat(),
            }
            self._log_batch_params(
                run_params,
                run_dir,
                eval_params,
            )

            return metrics, dfs
        except Exception as e:
            self._log_text(
                run_dir,
                "".join(traceback.format_tb(e.__traceback__)),
                artefact_filename="evaluation_exception.txt",
            )
            raise e

    def _save(
        self,
        active_run: Run,
        run_params: RunParams,
        model: RegressorWithPredict,
        trainer: TrainerInterface,
        preprocessor: FeaturesPreprocessor,
        dataset: Dataset,
        metrics:dict,
        run_dir: Path,
    ) -> None:
        """
        Save a model and its preprocessor given a mlflow run, run params, model, preprocessor and dataset.

        Logs saving time.
        """
        logging.info(f"[Save] model {run_params}")
        model_saver = get_saving_method(
            trainer,
            run_params,
        )

        try:
            save_start_time = time.time()
            model_saver(run_params.model_key, model, preprocessor, dataset, metrics, run_dir)
            save_time = time.time() - save_start_time

            metrics["saving_time"] = save_time

            save_metrics(run_dir, metrics)

        except Exception as e:
            logging.error(f"[Save] model {run_params}: Saving failed: {e}")
            self._log_text(
                run_dir,
                "".join(traceback.format_tb(e.__traceback__)),
                artefact_filename="saving_exception.txt",
            )
            raise e

    def _log_batch_params(
        self, run_params: RunParams, run_dir: Path, params: dict
    ) -> None:
        PARAMS_NAME = "params.json"
        params_to_log:dict = run_params.to_dict()
        params_to_log.update(params)
        if (run_dir / PARAMS_NAME).exists():
            with open(run_dir / PARAMS_NAME) as f:
                existing_params = json.load(f)
            params_to_log.update(existing_params)
        with open(run_dir / PARAMS_NAME, "w") as f:
            json.dump(params_to_log, f)

    def _log_text(self, run_dir: Path, content: str, artefact_filename: str) -> None:
        with open(run_dir / artefact_filename, "w") as f:
            f.write(content)
            f.close()
