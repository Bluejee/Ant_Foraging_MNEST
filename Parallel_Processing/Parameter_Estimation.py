import numpy as np
import pickle


def script_maker():
    with open('script2.bat', 'w') as f:
        for disp in [0.07, 0.08, 0.09]:
            for dcy in np.arange(0.0, 0.002, 0.0001):
                f.write(
                    f"python Ants.py --sim_name=Test_4_disp_{round(disp, 4)}_D_{round(dcy, 4)} -ns --dispersion_rate={round(disp, 4)} --decay_rate={round(dcy, 4)}\n")


def parallel_dict_maker():
    parameter_dict = {'dispersion_rate': [],
                      'decay_rate': [],
                      'drop_amount': [],
                      'min_exploration': [],
                      'exploration_rate': [],
                      'exploration_decay': [],
                      'learning_rate': [],
                      'discounted_return': []}

    # for dcy in np.arange(0.0, 0.002, 0.0001):
    for disp in [0.07, 0.08, 0.09]:
        for dcy in [0.0017, 0.0018, 0.0015]:
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

    print(parameter_dict)
    with open('parameter_dict.pickle', 'wb') as f:
        # Use pickle.dump() to write the dictionary to the file
        pickle.dump(parameter_dict, f)


parallel_dict_maker()
