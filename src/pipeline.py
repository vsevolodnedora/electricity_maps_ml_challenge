import logging
from datetime import datetime
from uuid import uuid4

from src.lib.argparser import parse_args_to_forecasts_to_train, validate_args
from src.lib.experiment import ExperimentConfig
from src.trainers import Trainers
from src.training import TrainingPipeline


def visualize_results():
    pass

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--zone_keys", default='US-CAL-CISO,US-TEX-ERCO')
    parser.add_argument("--forecast_keys", default=None)
    parser.add_argument("--forecast_subkeys", default=None)
    parser.add_argument(
        "--trainer_interface", choices=Trainers.__members__, default=None
    )
    args, _ = parser.parse_known_args()
    validate_args(args)

    forecasts_to_train = parse_args_to_forecasts_to_train(args, args.zone_keys)

    trainer_interface = (
        Trainers[args.trainer_interface] if args.trainer_interface else None
    )
    experiment_config = ExperimentConfig(
        id=str(uuid4()),
        run_datetime=datetime.now().isoformat(),
        forecasts_to_train=list(forecasts_to_train),
        trainer_interface=trainer_interface if trainer_interface else "",
    )

    pipeline = TrainingPipeline(experiment_config, trainer_interface)
    for forecast_to_train in forecasts_to_train:
        logging.info(f"Pipeline starting for {forecast_to_train}")
        features = pipeline.load_features(forecast_to_train)
        logging.info(f"Loaded features of size {features.size}")
        pipeline.run((forecast_to_train, features))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
