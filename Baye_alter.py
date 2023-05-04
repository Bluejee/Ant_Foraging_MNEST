import multiprocessing

from Ants import *

import time
import datetime
import pandas as pd

from skopt import Optimizer
from skopt.space import Real
from joblib import Parallel, delayed


def process_loop(dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate, exploration_decay,
                 learning_rate, discounted_return):
    global counter
    global result_dict
    global batch_name

    try:
        sim_name = batch_name+'/'+str(counter.value)
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
                                 max_steps=500000,
                                 sim_name=sim_name)
        total_food = para_realise.total_food_collected
        result_dict[sim_name] = [dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate,
                                 exploration_decay, learning_rate, discounted_return, total_food]
        return total_food
    except Exception as e:
        print(f"Error in simulation : {e}")
        return f"Error in simulation : {e}"


def process_loop_obj(params):
    [dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate,
     exploration_decay, learning_rate, discounted_return] = params
    result = process_loop(dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate,
                          exploration_decay, learning_rate, discounted_return)
    return -result


def printable_time(seconds):
    # Convert seconds to a timedelta object
    td = datetime.timedelta(seconds=seconds)

    # Extract the days, hours, minutes, and seconds from the timedelta object
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format the time string
    time_string = f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"

    return time_string


def now_plus_time(seconds):
    # Get the current time
    now = datetime.datetime.now()

    # Calculate the finish time
    finish_time = now + datetime.timedelta(seconds=seconds)

    # Format the finish time string
    finish_time_string = finish_time.strftime("%Y-%m-%d %H:%M:%S")

    return finish_time_string


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

    n_jobs = os.cpu_count()
    max_iterations = 20
    batch_name = 'Batch_1'

    print(f'Optimization Starting at :: {now_plus_time(0)}')
    print(f'Optimization using {n_jobs} Cores')
    print('-'*100)
    for i in range(max_iterations):
        loop_start = time.perf_counter()
        print(f'Round {i + 1} of {max_iterations}')
        x = optimizer.ask(n_points=n_jobs)  # x is a list of n_points points
        y = Parallel(n_jobs=n_jobs)(delayed(process_loop_obj)(v) for v in x)  # evaluate points in parallel
        optimizer.tell(x, y)
        df = pd.DataFrame.from_dict(result_dict, orient='index',
                                    columns=['dispersion_rate', 'decay_rate', 'drop_amount', 'min_exploration',
                                             'exploration_rate', 'exploration_decay', 'learning_rate',
                                             'discounted_return', 'total_food'])
        df.to_csv(f'Analysis/{batch_name}/Parameters_({n_jobs}cores).csv', index_label='sim_name')
        loop_end = time.perf_counter()
        run_time = loop_end - loop_start
        eta = run_time * (max_iterations - (i + 1))
        print(f'Time taken for round {i + 1} = {printable_time(run_time)}')
        print(f'Estimated Time Remaining :: {printable_time(eta)}')
        print(f'Would Probably Finish at :: {now_plus_time(eta)}')
        print('-'*100)

    # print(optimizer)
    # best_params, best_obj = optimizer.result()
    # print(best_obj, best_params)
    para_end = time.perf_counter()
    total_time = para_end - para_start
    print(f"Total Completion_Time :: {printable_time(total_time)}")
    print(f'Optimization Finished at :: {now_plus_time(0)}')

    # print(result_dict)
