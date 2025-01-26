# ML Engineering Challenge

Welcome to the Electricity Maps ML Engineer technical!

First off, thanks for taking the time and putting effort into completing this challenge, we really appreciate it!

Remember, there’s never only a single solution to a problem, so don’t stress over striving for perfection. Have fun and express yourself, feel free to do stuff that is outside the box.

## Task

At Electricity Maps, our mission is to enable data-driven decarbonisation. Over the years, we’ve seen that matching electricity consumption with carbon-free electricity availability promotes further integration of fossil-free electricity sources on the grids.

One of our goals is therefore to allow our customers to use electricity when it is low-carbon. Your task will be to replicate one of the key components that enable this: __predicting the solar power production__.

In practice, we have prepared some scripts to help you retrieve the necessary data, and are providing you with a pre-built forecasting pipeline to enable you to focus on specific elements of the end-to-end lifecycle of a forecasting model. The task is split up in small tasks that we recommend tackling in succession, as explained below in the [Task Breakdown](#task-breakdown)

## Expectations

* We expect you to spend a couple of hours, max half a day, so prioritise wisely what you think is the most important.
* Your approach should scale to multiple signals (not only solar production, but also production for other modes, or storage, exchanges etc) and to multiple zones.
* Please include as many notes as possible to allow us to follow your investigation process
* Please document your thought process, assumptions and key decisions as clearly as possible. This provides a great opportunity to understand how you go about solving a real-world problem and is a chance for you to shine.

## Tools

Feel free to expand the provided project with any tools and dependencies you like, as long as they are documented and that the code can still be run once the correct dependencies have been installed.

## Deliverables
The end deliverable is a GitHub repository containing:

* A `README.md` file containing an explanation of how you solved the problem (replacing this current content). What were your assumptions? How would you improve your forecasting system if you had more time?__Ideally the README should read like a simple yet structured report on the task of forecasting solar power production in these two US zones. Focus on what you learned from the data, insightful details about your preprocessing and modelling approach, and how you see this capabiltiy being used.__
* A short description of how to run the program. Sending us the output and some evaluation metrics would also be very much appreciated.
* Your code

Don’t hesitate to ask us questions if anything is unclear.

Looking forward to your creation!

_The Electricity Maps team_

## Task breakdown

### Task 0.1 - Set up

Follow the [Set up](#set-up) instructions to get your environment ready and to obtain the required data

### Task 1.1 - Exploratory data analysis

We encourage you to use the notebook `notebooks/exploratory_data_analysis.ipynb` to run an exploratory data analysis on the provided data.

### Task 2.1 - Build a trainer

We expect you to be able to implement a trainer class, inheriting from `src/lib/interface.py:TrainerInterface`.

By including your trainer in `src/trainers.py` and updating the `src/lib/train.py:get_trainer` function, the pipeline should be able to execute the training and fail when trying to evaluate the model.

We will be curious to know:

* Why you chose a given model class.
* How did you decide on hyperparameters.
* Why did you choose to use a given preprocessor.
* If you think whether the trainer would be usable for other signals (wind power production for example).

### Task 2.2 - Build an evaluator

We expect you to fill in the `src/lib/evaluate.py:get_evaluation_method` function to return a method that can be called to evaluate your trained model.

We encourage you to think about:

* How to make the evaluation as meaningful as possible.
* What artefacts would be generated during the evaluation.
* How to ensure that the evaluation works across trainers / keys.

The pipeline should then be able to execute the training and evaluation before failing when trying to save the model.

### Task 2.3 - Saving models and artefacts

We expect you to fill in the `src/lib/save.py:get_saving_method` function to return a method that can be called to save your model and artefacts.

We are curious to learn how you can ensure that the model thus saved will be usable by downstream applications and across time. What artefacts would you need to log to ensure that an application that has access to the saved output can run the model?

The pipeline should by now run end-to-end.

### Task 2.4 - Tracking metrics

We expect you to fill the `src/lib/metrics.py:save_metrics` function.

It should be able to be executed on multiple sets of metrics and store them in a consistent fashion.

We encourage you to add metrics you find useful across the pipeline. We'll be interested to know why you deem these metrics useful.

###  Task 3.1 - Serving your models

Finally, the last part of the assignment is much more open-ended.

We'd like you to propose a way to serve the models you have trained and have selected post evaluation. It could be for example through a service, a simple command, a container exposing an endpoint.

The requirements here are deliberately loose, as we'd like to understand how you reason about such an open-ended problem and lay down your assumptions and document your decisions.

We only expect to be able to call any models you deem good enough by a simple command, and these models should return 24 hours of forecasts.

# Set up

## Install dependencies

__python version__:
First make sure that your python version matches the one specified in `.python-version`. We recommend using [pyenv](https://github.com/pyenv/pyenv) to manage your python versions.

__virtual environment__: To create a dedicated environment for this workspace, let's generate a virtual environment:
```
python -m venv .venv
```

__activating the virtual environment and provisioning the python path__:
```
source .venv/bin/activate && source .env
```


__verifications__:
Let's check the python environment
```
$ which python
.../mlengineer-challenge/.venv/bin/python
```

_Note: We do not have any windows machine and could not test the repository for it. If you're running the challenge on Windows, please let us know how we could improve the instructions_

__install packages__:
Run the installation of packages
```
pip install -r requirements.txt
```

## Retrieve the data

We are providing you with a script that should automatically download all features and targets you'll need for the challenge.

```
python data/download.py
```

You should see a log similar to `INFO:root:✅ Successfully retrieved targets for US-TEX-ERCO!` appear.

You have now features and targets for two US zones, [California (US-CAL-CISO)](https://app.electricitymaps.com/zone/US-CAL-CISO) and [Texas (US-TEX-ERCO)](https://app.electricitymaps.com/zone/US-TEX-ERCO).

The features are a collection of:
* Weather data, assembled across a grid that covers the zone geographical area. Ex: `wind_speed_100m_0` represents the wind speed at 100m of altitude for the point with index `0`. Points are roughly spread on a grid of 1 degree of longitude/latitude resolution.

| Feature name    | Signification             |
|-----------------|---------------------------|
| dswrf_avg       | downwards solar radiation |
| lcdc_instant    | low cloud cover           |
| wind_speed_100m | wind speed at 100m        |
| wind_dir_100m   | wind direction at 100m    |

* Lagged features. They provide the potential models with context around the maximum solar power production observed in previous past periods. Ex: `solar_max_14d_lag_3d`
* Time encodings. They encode seasonality components for the target time they apply to. Ex: `hour_cos`

We ask you to:

* Treat the data as confidential
* Not commit it to your GitHub repository

## Running the pipeline

We are providing you with a complete scaffolding for an end-to-end training pipeline.

You can run the pipeline with

```
python src/pipeline.py --zone_keys US-CAL-CISO,US-TEX-ERCO
```

__Required Arguments__

* --zone_keys: Comma-separated list of zone keys.
  * Example: --zone_keys US-CAL-CISO,US-TEX-ERCO

__Optional Arguments__

* --forecast_keys: Comma-separated list of forecast keys.
  * Default: None
  * Example: --forecast_keys production,...
* --forecast_subkeys: Comma-separated list of forecast subkeys.
  * Default: None
  * Example: --forecast_subkeys solar,...
* --trainer_interface: Specify the trainer interface to use.
  * Choices: [list of available Trainers.members]
  * Default: None
  * Example: --trainer_interface MySuperTrainer3000

By default, you should receive a `NotImplementedError` raised by the `get_trainer` function. That's normal! You'll fix that by going through the tasks.

## Running tests

With pytest you can simply run the tests by:

```
pytest .
```

Feel free to add your own tests!