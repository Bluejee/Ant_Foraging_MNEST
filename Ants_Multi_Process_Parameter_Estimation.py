import concurrent.futures
import time
from Ants import *
import pickle

# Open the file for reading in binary mode
with open('Parallel_Processing/parameter_dict.pickle', 'rb') as f:
    # Use pickle.load() to deserialize the data from the file
    parameter_dict = pickle.load(f)


def process_loop(index):
    start = time.perf_counter()
    realise = Visualise(dispersion_rate=parameter_dict['dispersion_rate'][index],
                        decay_rate=parameter_dict['decay_rate'][index],
                        drop_amount=parameter_dict['drop_amount'][index],
                        min_exploration=parameter_dict['min_exploration'][index],
                        exploration_rate=parameter_dict['exploration_rate'][index],
                        exploration_decay=parameter_dict['exploration_decay'][index],
                        learning_rate=parameter_dict['learning_rate'][index],
                        discounted_return=parameter_dict['discounted_return'][index],
                        no_show=True,
                        start_as='Play',
                        max_steps=10000,
                        sim_name=(f"Trial_{index}_" +
                                  "Disp_{parameter_dict['dispersion_rate'][index]}_" +
                                  "Dcy_{parameter_dict['decay_rate'][index]}"
                                  ))
    end = time.perf_counter()
    return f"Sim:: {parameter_dict['sim_name'][index]}, Completion_Time :: {round(end - start)}"


with concurrent.futures.ProcessPoolExecutor() as executor:
    results = executor.map(process_loop, range(len(parameter_dict)))

    for result in concurrent.futures.as_completed(results):
        print(result)
