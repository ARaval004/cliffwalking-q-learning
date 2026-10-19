# CliffWalking Q-Learning
Q-learning implementation for the Gymnasium CliffWalking environment, including hyperparameter experiments and stochastic action testing.


## Requirements

- Python 3
- Gymnasium
- NumPy
- Matplotlib
- MoviePy

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/ARaval004/cliffwalking-q-learning.git
   ```

2. Navigate to the project folder:

   ```
   cd cliffwalking-q-learning
   ```

3. Install the required libraries:

   ```
   python -m pip install -r requirements.txt
   ```

## Running the Program

Run the program from the main project directory:

```
python src/cliffwalking.py
```

The program runs the baseline training, hyperparameter experiments, final agent evaluation, and stochastic environment experiments.

**Note:** Six graph windows will appear during execution. Each window must be closed manually before the program continues to the next experiment.

The generated files are saved automatically in the following folders:

- `results/` — Training and evaluation graphs (PNG)
- `videos/` — MP4 recording of the final trained agent

Evaluation statistics and experiment results are printed in the terminal.

No changes to the source code are required to run the experiments. The same Q-learning training function is used throughout, with stochastic action noise disabled by default.
