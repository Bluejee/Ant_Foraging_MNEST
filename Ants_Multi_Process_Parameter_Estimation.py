import concurrent.futures
import multiprocessing
from Ants import *
import pickle
import os


def process_loop(index, parameter_dictionary):
    try:
        start = time.perf_counter()
        sim_name = (f"Trial_{index}_" +
                    f"Disp_{round(parameter_dictionary['dispersion_rate'][index], 5)}_" +
                    f"Dcy_{round(parameter_dictionary['decay_rate'][index], 5)}"
                    )
        para_realise = Visualise(dispersion_rate=parameter_dictionary['dispersion_rate'][index],
                                 decay_rate=parameter_dictionary['decay_rate'][index],
                                 drop_amount=parameter_dictionary['drop_amount'][index],
                                 min_exploration=parameter_dictionary['min_exploration'][index],
                                 exploration_rate=parameter_dictionary['exploration_rate'][index],
                                 exploration_decay=parameter_dictionary['exploration_decay'][index],
                                 learning_rate=parameter_dictionary['learning_rate'][index],
                                 discounted_return=parameter_dictionary['discounted_return'][index],
                                 no_show=True,
                                 start_as='Play',
                                 max_steps=500000,
                                 sim_name=sim_name)
        total_food = para_realise.total_food_collected
        end = time.perf_counter()
        out = f"Sim:: {sim_name}, Completion_Time :: {round(end - start)}, Total_Food :: {total_food}"
        return out
    except Exception as e:
        return f"Error in simulation {index}: {e}"


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Open the file for reading in binary mode
    with open('Parallel_Processing/parameter_dict.pickle', 'rb') as f:
        # Use pickle.load() to deserialize the data from the file
        parameter_dict = pickle.load(f)
    para_start = time.perf_counter()

    # (max_workers=int(os.cpu_count())) use this to reduce lode if needed.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_loop, range(len(parameter_dict['decay_rate'])),
                               [parameter_dict] * len(parameter_dict['decay_rate']))

        for result in results:
            print(result)

    # for i in range(len(parameter_dict['decay_rate'])):
    #     print(process_loop(i, parameter_dict))

    para_end = time.perf_counter()
    print(f"Total Completion_Time :: {round(para_end - para_start)}")
