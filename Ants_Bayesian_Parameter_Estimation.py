import multiprocessing

from Ants import *

import time
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


def process_loop(dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate, exploration_decay,
                 learning_rate, discounted_return):
    global counter
    global result_dict

    try:
        sim_name = str(counter.value)
        counter.value += 1
        para_realise = Visualise(dispersion_rate=dispersion_rate,
                                 decay_rate=decay_rate,
                                 drop_amount=drop_amount,
                                 min_exploration=min_exploration,
                                 exploration_rate=exploration_rate,
                                 exploration_decay=exploration_decay,
                                 learning_rate=learning_rate,
                                 discounted_return=discounted_return,
                                 no_show=True,
                                 start_as='Play',
                                 max_steps=700000,
                                 sim_name=sim_name)
        total_food = para_realise.total_food_collected
        result_dict[sim_name] = [dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate,
                                 exploration_decay, learning_rate, discounted_return, total_food]
        return total_food
    except Exception as e:
        print(f"Error in simulation : {e}")
        return f"Error in simulation : {e}"


# Define the search space for the Bayesian optimization
space = [Real(0.01, 1.0, name='dispersion_rate'),
         Real(0.01, 1.0, name='decay_rate'),
         Real(0.01, 1.0, name='drop_amount'),
         Real(0.01, 1.0, name='min_exploration'),
         Real(0.01, 1.0, name='exploration_rate'),
         Real(0.01, 1.0, name='exploration_decay'),
         Real(0.01, 1.0, name='learning_rate'),
         Real(0.01, 1.0, name='discounted_return')]


# Define the function to be optimized using the search space
@use_named_args(space)
def objective(**params):
    return -process_loop(**params)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    para_start = time.perf_counter()

    # I know this is a good value to start at.
    x0 = [0.36061122, 0.2305835, 0.53455935, 0.79382278, 0.91876425, 0.962777704, 0.7284984445, 0.16317983]
    # Use Bayesian optimization to find the optimum parameters

    # Create shared dictionary and counter using Manager
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    counter = manager.Value('i', 0)
    # process_loop(0.36061122, 0.2305835, 0.53455935, 0.79382278, 0.91876425, 0.962777704, 0.7284984445, 0.16317983)
    res_gp = gp_minimize(objective, space, n_calls=20, random_state=0, n_jobs=-1, x0=x0, verbose=True)

    # Print the best parameters and corresponding score
    print("Best score: %f" % res_gp.fun)
    print("Best parameters: ", res_gp.x)
    para_end = time.perf_counter()
    print(f"Total Completion_Time :: {round(para_end - para_start)}")
    print(result_dict)
    df = pd.DataFrame.from_dict(result_dict, orient='index',
                                columns=['dispersion_rate', 'decay_rate', 'drop_amount', 'min_exploration',
                                         'exploration_rate', 'exploration_decay', 'learning_rate', 'discounted_return',
                                         'total_food'])
    df.to_csv('Analysis/Bayes_Params.csv', index_label='sim_name')
