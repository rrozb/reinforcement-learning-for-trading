# from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl
from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from finrl.meta.data_processor import DataProcessor


def train(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, drl_lib, env, model_name, if_vix=True,
          **kwargs):
    # process data using unified data processor
    DP = DataProcessor(data_source,  **kwargs)
    price_array, tech_array, turbulence_array = DP.download_data(ticker_list=ticker_list, start_date=start_date, end_date=end_date, time_interval=time_interval)
    #DP.run(ticker_list,
                                                       # technical_indicator_list,
                                                       # if_vix, cache=True)

    data_config = {'price_array': price_array,
                   'tech_array': tech_array,
                   'turbulence_array': turbulence_array}

    # build environment using processed data
    env_instance = env(config=data_config)

    # read parameters and load agents
    current_working_dir = kwargs.get('current_working_dir', './' + str(model_name))



    if drl_lib == 'rllib':
        total_episodes = kwargs.get('total_episodes', 100)
        rllib_params = kwargs.get('rllib_params')

        agent_rllib = DRLAgent_rllib(env=env,
                                     price_array=price_array,
                                     tech_array=tech_array,
                                     turbulence_array=turbulence_array)

        model, model_config = agent_rllib.get_model(model_name)

        model_config['lr'] = rllib_params['lr']
        model_config['train_batch_size'] = rllib_params['train_batch_size']
        model_config['gamma'] = rllib_params['gamma']

        trained_model = agent_rllib.train_model(model=model,
                                                model_name=model_name,
                                                model_config=model_config,
                                                total_episodes=total_episodes)
        trained_model.save(current_working_dir)


    elif drl_lib == 'stable_baselines3':
        total_timesteps = kwargs.get('total_timesteps', 1e6)
        agent_params = kwargs.get('agent_params')

        agent = DRLAgent_sb3(env=env_instance)

        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(model=model,
                                          tb_log_name=model_name,
                                          total_timesteps=total_timesteps)
        print('Training finished!')
        trained_model.save(current_working_dir)
        print('Trained model saved in ' + str(current_working_dir))
    else:
        raise ValueError('DRL library input is NOT supported. Please check.')

