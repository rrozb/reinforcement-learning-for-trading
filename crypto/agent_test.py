# import DRL agents
from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from finrl.meta.data_processor import DataProcessor
# from agents.elegantrl_models import DRLAgent as DRLAgent_erl

# import data processor
from finrl.meta.data_processor import DataProcessor



def test(start_date, end_date, ticker_list, data_source, time_interval,
         technical_indicator_list, drl_lib, env, model_name, if_vix=True,
         **kwargs):
    # process data using unified data processor
    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(ticker_list,
                                                       technical_indicator_list,
                                                       if_vix, cache=True)

    np.save('./price_array.npy', price_array)
    data_config = {'price_array': price_array,
                   'tech_array': tech_array,
                   'turbulence_array': turbulence_array}
    # build environment using processed data
    env_instance = env(config=data_config)

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2 ** 7)
    current_working_dir = kwargs.get("current_working_dir", "./" + str(model_name))
    print("price_array: ", len(price_array))


    if drl_lib == "rllib":
        # load agent
        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=current_working_dir,
        )

        return episode_total_assets

    elif drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=current_working_dir
        )

        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")