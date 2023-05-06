[dispersion_rate, decay_rate, drop_amount, min_exploration, exploration_rate, exploration_decay, learning_rate,
 discounted_return] = [0.9892177908568219, 0.00011995805387859148, 0.9721295922300442, 0.13360550143309072,
                       0.5059876186349819, 0.6175957756036495, 0.7335766393585074, 0.3842657235643588]
sim_name = 'Hope_3'
max_steps = 100000
print(f'python .\\Ants.py ' +
      f' --dispersion_rate={dispersion_rate} ' +
      f'--decay_rate={decay_rate} ' +
      f'--drop_amount={drop_amount} ' +
      f'--min_exploration={min_exploration} ' +
      f'--exploration_rate={exploration_rate} ' +
      f'--exploration_decay={exploration_decay} ' +
      f'--learning_rate={learning_rate} ' +
      f'--discounted_return={discounted_return} ' +
      f'--sim_name={sim_name}' +
      f'--max_steps={max_steps}')
