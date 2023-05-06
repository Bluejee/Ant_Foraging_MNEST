import numpy as np
import pickle
import csv


def script_maker():
    with open('script2.bat', 'w') as f:
        for disp in [0.07, 0.08, 0.09]:
            for dcy in np.arange(0.0, 0.002, 0.0001):
                f.write(
                    f"python Ants.py --sim_name=Test_4_disp_{round(disp, 4)}_D_{round(dcy, 4)} -ns --dispersion_rate={round(disp, 4)} --decay_rate={round(dcy, 4)}\n")


def parallel_dict_maker(list_type='Grid'):
    parameter_dict = {'dispersion_rate': [],
                      'decay_rate': [],
                      'drop_amount': [],
                      'min_exploration': [],
                      'exploration_rate': [],
                      'exploration_decay': [],
                      'learning_rate': [],
                      'discounted_return': []}
    if list_type == 'Random':
        num_tests = 500
        for index in range(num_tests):
            parameter_dict['dispersion_rate'].append(np.random.random())
            parameter_dict['decay_rate'].append(np.random.random())
            parameter_dict['drop_amount'].append(np.random.random())
            parameter_dict['min_exploration'].append(np.random.random())
            parameter_dict['exploration_rate'].append(np.random.random())
            parameter_dict['exploration_decay'].append(np.random.random())
            parameter_dict['learning_rate'].append(np.random.random())
            parameter_dict['discounted_return'].append(np.random.random())
    elif list_type == 'Grid':
        for disp in np.arange(0.0, 0.11, 0.01):
            for dcy in np.arange(0.0, 0.011, 0.001):
                for drop_amount in [0.05]:
                    for min_exp in [0.05]:
                        for exp_rate in [0.9]:
                            for exp_dcy in [0.0001]:
                                for lern_rate in [0.4]:
                                    for disc_ret in [0.85]:
                                        parameter_dict['dispersion_rate'].append(disp)
                                        parameter_dict['decay_rate'].append(dcy)
                                        parameter_dict['drop_amount'].append(drop_amount)
                                        parameter_dict['min_exploration'].append(min_exp)
                                        parameter_dict['exploration_rate'].append(exp_rate)
                                        parameter_dict['exploration_decay'].append(exp_dcy)
                                        parameter_dict['learning_rate'].append(lern_rate)
                                        parameter_dict['discounted_return'].append(disc_ret)
    else:
        print('Wrong Type.')

    print(len(parameter_dict['decay_rate']))
    with open('parameter_dict.pickle', 'wb') as f:
        # Use pickle.dump() to write the dictionary to the file
        pickle.dump(parameter_dict, f)

    with open('parameter_list.csv', 'w') as f:
        # Create a writer object
        writer = csv.writer(f)

        # Write the header row
        writer.writerow(parameter_dict.keys())

        # Write the data rows
        for i in range(len(parameter_dict['decay_rate'])):
            row = [parameter_dict[name][i] for name in parameter_dict]
            writer.writerow(row)


parallel_dict_maker('Random')

# This is a parameter that works.
# python .\Ants.py --dispersion_rate=0.36061122 --decay_rate=0.2305835 --drop_amount=0.53455935
# --min_exploration=0.79382278 --exploration_rate=0.91876425 --exploration_decay=0.962777704
# --learning_rate=0.7284984445 --discounted_return=0.16317983 --sim_name=Hope_1

# python .\Ants.py --dispersion_rate=0.9942416881121815 --decay_rate=0.0 --drop_amount=0.5714313390496405
# --min_exploration=0.24635531919842119 --exploration_rate=0.9253469328911803 --exploration_decay=0.3233293461380406
# --learning_rate=0.422595187361183 --discounted_return=0.48672879956102355 --sim_name=Hope_2
