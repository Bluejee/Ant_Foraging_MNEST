import multiprocessing

from Ants import *

import time
import pandas as pd

from skopt import Optimizer
from skopt.space import Real
from joblib import Parallel, delayed


def process_loop(dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate, exploration_decay,
                 learning_rate, discounted_return):
    global counter
    global result_dict

    try:
        sim_name = str(counter.value+1000)
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
                                 max_steps=10000,
                                 sim_name=sim_name)
        total_food = para_realise.total_food_collected
        result_dict[sim_name] = [dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate,
                                 exploration_decay, learning_rate, discounted_return, total_food]
        return total_food
    except Exception as e:
        print(f"Error in simulation : {e}")
        return f"Error in simulation : {e}"


def process_loop_obj(x):
    dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate, exploration_decay, learning_rate, discounted_return = x
    result = process_loop(dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate,
                          exploration_decay, learning_rate, discounted_return)
    return -result


optimizer = Optimizer(
    dimensions=[Real(0.0, 1.0),
                Real(0.0, 1.0),
                Real(0.0, 1.0),
                Real(0.0, 1.0),
                Real(0.0, 1.0),
                Real(0.0, 1.0),
                Real(0.0, 1.0),
                Real(0.0, 1.0)],
    random_state=1,
    base_estimator='gp'
)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    para_start = time.perf_counter()
    # Use Bayesian optimization to find the optimum parameters

    # Create shared dictionary and counter using Manager
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    counter = manager.Value('i', 0)

    n_jobs = os.cpu_count()-2

    for i in range(2):
        x = optimizer.ask(n_points=n_jobs)  # x is a list of n_points points
        y = Parallel(n_jobs=n_jobs)(delayed(process_loop_obj)(v) for v in x)  # evaluate points in parallel
        optimizer.tell(x, y)

    print(optimizer)
    # best_params, best_obj = optimizer.result()
    # print(best_obj, best_params)
    para_end = time.perf_counter()

    print(f"Total Completion_Time :: {round(para_end - para_start)}")

    print(result_dict)
    df = pd.DataFrame.from_dict(result_dict, orient='index',
                                columns=['dispersion_rate', 'decay_rate', 'drop_amount', 'min_exploration',
                                         'exploration_rate', 'exploration_decay', 'learning_rate', 'discounted_return',
                                         'total_food'])
    df.to_csv('Analysis/Bayes_Params.csv', index_label='sim_name')
