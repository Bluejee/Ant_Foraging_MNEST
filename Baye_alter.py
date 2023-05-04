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
        sim_name = batch_name + '/' + str(counter.value)
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
                                 max_steps=600000,
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


if __name__ == '__main__':
    multiprocessing.freeze_support()

    ####################################################################################################################

    # Setup Initial Variables
    batch_name = 'Batch_Trial'

    # P.S. not changing the seed will keep giving the same result. So for the same world, change the seed.
    seed = 1  # A Random seed for reproducibility of optimizer.

    # Bayesian analysis works well if more cores are used.
    # It scans more point each iteration and hence comes to a better solution faster.
    n_jobs = os.cpu_count()  # Max CPUs

    # If cores are low then increase the number of iterations.
    max_iterations = 30  # No. of Optimization iterations (works well for at least 40 cores not so much for 8)

    ####################################################################################################################

    # Create a log file for Variables.
    # Check whether the specified path exists or not
    if not os.path.exists(f"Analysis/{batch_name}"):
        # Create a new directory because it does not exist
        os.makedirs(f"Analysis/{batch_name}")

    # Needs Improvement of display. Use format specifiers to make sure it looks good.
    with open(f'Analysis/{batch_name}/0_README.org', 'w') as f:
        f.write(f"* Ant Foraging Simulation\n")
        f.write(f"---\n")
        f.write(f"- Batch Name           :: {batch_name}\n")
        f.write(f"- Random Seed          :: {seed}\n")
        f.write(f"- Number of Cores      :: {n_jobs}\n")
        f.write(f"- Number of Iterations :: {max_iterations}\n")

    ####################################################################################################################

    optimizer = Optimizer(
        dimensions=[Real(0.0, 1.0),
                    Real(0.0, 1.0),
                    Real(0.0, 1.0),
                    Real(0.0, 1.0),
                    Real(0.0, 1.0),
                    Real(0.0, 1.0),
                    Real(0.0, 1.0),
                    Real(0.0, 1.0)],
        random_state=seed,
        base_estimator='gp'
    )

    ####################################################################################################################

    # Create shared dictionary and counter using Manager
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    counter = manager.Value('i', 1)

    ####################################################################################################################

    para_start = time.perf_counter()
    print(f'Optimization Starting at :: {now_plus_time(0)}')
    print(f'Optimization using {n_jobs} Cores')
    print('-' * 100)

    ####################################################################################################################

    # Use Bayesian optimization to find the optimum parameters
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
        df.to_csv(f'Analysis/{batch_name}/0_Parameters.csv', index_label='sim_name')
        loop_end = time.perf_counter()
        run_time = loop_end - loop_start
        eta = run_time * (max_iterations - (i + 1))
        print(f'Time taken for round {i + 1} = {printable_time(run_time)}')
        print(f'Estimated Time Remaining :: {printable_time(eta)}')
        print(f'Would Probably Finish at :: {now_plus_time(eta)}')
        print('-' * 100)

    ####################################################################################################################

    para_end = time.perf_counter()
    total_time = para_end - para_start
    print(f"Total Completion_Time :: {printable_time(total_time)}")
    print(f'Optimization Finished at :: {now_plus_time(0)}')
