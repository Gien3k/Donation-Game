# Advanced Donation Game Simulator

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 

This repository provides a sophisticated simulation environment for exploring evolutionary strategies within the **Donation Game**. It features an interactive web interface built with Streamlit, allowing users to easily configure, execute, monitor, and analyze simulation results across various game models.

**Acknowledgement:** The foundational concepts for the donation game models explored here, particularly those involving **Generosity and Forgiveness**, are based on the research presented by **Nathan Griffiths and Nir Oren** in their paper: *"Generosity and the Emergence of Forgiveness in the Donation Game"*. Their original Java implementation, which served as inspiration for the logic in this project, can be found here: [nathangriffiths/ECAI-Forgiveness](https://github.com/nathangriffiths/ECAI-Forgiveness).

The application leverages Celery and Redis for asynchronous task management, enabling computationally intensive simulations to run in the background without blocking the user interface. It includes user authentication, simulation history, configuration presets, and result comparison features.

This implementation was developed independently in Python, based on the concepts from the aforementioned paper and Java implementation.

## Key Features

* **Interactive Web UI:** Configure parameters, launch simulations, and view results via a user-friendly Streamlit dashboard.
* **Multiple Game Models:** Simulate different theoretical scenarios:
    * Base Model
    * Noisy Model (Action & Perception errors)
    * Generosity Model
    * Forgiveness Model
* **Efficient Simulation Core:** Utilizes NumPy and Numba for optimized agent interactions and calculations (`8.py`).
* **Asynchronous Task Execution:** Offloads simulations to background workers using Celery and Redis (`tasks.py`).
* **User Authentication:** Secure access using `streamlit-authenticator` with configuration via `config.yaml`.
* **Simulation History:** Automatically logs past simulation runs (configuration, results, code snapshot).
* **Configuration Presets:** Save and load parameter configurations for repeatable experiments.
* **Results Visualization:** Displays results in tables (Pandas) and interactive charts (Altair).
* **Comparison Tool:** Save specific runs to slots and compare their parameters and results side-by-side.
* **Containerized Deployment:** Easily deployable using Docker and Docker Compose.

## Models Implemented (`8.py`)

* **Base Model (`tick_base`):** A foundational model where observers update reputation, but donors do not update their self-image. No noise is included.
* **Noisy Model (`tick_noisy`):** Introduces action noise (`ea` - probability of unintended action) and perception noise (`ep` - probability of misinterpreting an observed action).
* **Generosity Model (`tick_generosity`):** Extends the noisy model, allowing exploration of how observers update reputation under perception errors (`g1`) and adding a chance for donors to cooperate even when their strategy dictates defection (`g2`).
* **Forgiveness Model (`tick_forgiveness`):** Implements logic where donors update their own reputation. Includes mechanisms for forgiveness influencing both the donor's action and the observer's reputation update, based on configurable strategies (`forgiveness_strategies`, `fa`, `fr`).

## Technology Stack

* **Backend:** Python
* **Simulation:** NumPy, Numba, Pandas
* **Web Framework:** Streamlit
* **Task Queue:** Celery
* **Message Broker/Backend:** Redis
* **Visualization:** Altair
* **Configuration/Validation:** Pydantic, PyYAML
* **Authentication:** Streamlit-Authenticator, Bcrypt
* **Deployment:** Docker, Docker Compose

## Running the Simulation Script (`8.py`)

While the Streamlit application provides a user interface, the core simulation logic resides in `8.py` and can be run directly from the command line. This execution path is primarily utilized by the Celery worker but can be useful for direct testing or batch processing outside the UI.

Here are the available command-line arguments for `8.py`:

* `--size <int>`: Number of agents (N). Default: `100`.
* `--pairs <int>`: Number of agent pairs interacting per generation (M). Default: `300`.
* `--generations <int>`: Number of generations to simulate. Default: `100000`.
* `--runs <int>`: Number of independent simulation runs to perform for averaging results across different parameter combinations. Default: `100`.
* `--mutation <float>`: Mutation rate applied to agent strategies after selection (probability of a strategy changing randomly). Default: `0.001`.
* `--model <int>`: Selects the simulation model to use. Default: `1` (Noisy Model).
    * `0`: Base Model (`tick_base`)
    * `1`: Noisy Model (`tick_noisy`)
    * `2`: Generosity Model (`tick_generosity`)
    * `3`: Forgiveness Model (`tick_forgiveness`)
* `--output <str>`: Filename for the output CSV containing aggregated results (mean/std deviation of rewards per parameter set). Default: `"results.csv"`.
* `--q_values <float> [<float> ...]`: One or more 'q' values representing the probability of an interaction being observed by third parties. Default: `1.0`.
* `--ea_values <float> [<float> ...]`: One or more action noise ('ea') probability values (probability of performing an action different from the intended one). Used by Noisy, Generosity, Forgiveness models. Default: `[0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]`.
* `--ep_values <float> [<float> ...]`: One or more perception noise ('ep') probability values (probability of an observer misinterpreting an action). Used by Noisy, Generosity, Forgiveness models. Default: `[0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]`. *(Note: The script iterates through pairs of `ea` and `ep` values based on their list index; ensure lists have the same length if pairing specific noise levels).*
* `--generosity`: A flag that, when present, enables specific logic within the Generosity model (used with `--model 2`).
* `-g1 <float> [<float> ...]`: One or more 'g1' parameter values, used within the Generosity model (`--model 2 --generosity`) related to reputation updates under perception errors. Default: `[0.0, 0.01, 0.02, 0.03, 0.04, 0.05]`.
* `-g2 <float> [<float> ...]`: One or more 'g2' parameter values, used within the Generosity model (`--model 2 --generosity`) related to the propensity for unintended cooperation. Default: `[0.0]`.
* `--forgiveness_action`: A flag that, when present, enables the action forgiveness mechanism in the Forgiveness model (used with `--model 3`).
* `--forgiveness_reputation`: A flag that, when present, enables the reputation forgiveness mechanism in the Forgiveness model (used with `--model 3`).

**Example Usage (direct execution):**

```bash
# Run the Noisy model for fewer generations/runs with specific q and noise values
python 8.py --model 1 --generations 50000 --runs 10 --q_values 0.9 1.0 --ea_values 0.05 0.1 --ep_values 0.05 0.1 --output noisy_results_subset.csv

# Run the Generosity model varying g1
python 8.py --model 2 --generosity --g1 0.0 0.02 0.04 --ea_values 0.1 --ep_values 0.1 --output generosity_g1_results.csv
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
