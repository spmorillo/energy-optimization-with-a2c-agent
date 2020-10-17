import sys
sys.path.insert(0, "/media/hdd/Teleco/TFM/mercury/repo/lite/gride/gride_model") # Change path to Mercury location!!
import pandas as pd
from mercury import Mercury, LinkConfiguration, TransceiverConfiguration, RadioConfiguration
from rl.training.a2c_tfm.resource_manager.strategy import SDNRLAgentStrategy
from rl.mercury_models.processing_units.air.rx580.rx580_power_model import Rx580PowerModel as AirProcessingUnit
from rl.mercury_models.processing_units.immersion.rx570.rx570_power_model import Rx570PowerModel as ImmersionProcessingUnit
from rl.mercury_models.cooling_system.sota_hot_cold import AirCoolingSystem
from rl.mercury_models.cooling_system.wilo_ipl import WiloIPLPowerModel as ImmersionCoolingSystem

#CHANGE RX580 AIR MODEL TO COMPARE BOTH EDC


# Edge Data Center management configuration
hw_dvfs_mode = False
gpus_per_edc = 15
max_start_stop = 1
n_hot_standby = 0
hw_power_off = True
disp_strategy = 'emptiest_rack_fullest_pu'

# Simulation time and UE guard time
sim_time = 180
max_guard_time = 0

# Number of EDCs
n_edcs = 3
# Number of APs
n_aps = 11
# Number of UEs
n_ues = 50
# Model type
lite = True
shortcut = True

# Reports root file
model_type = 'lite' if lite else 'shortcut' if shortcut else 'classic'
sim_id = '{}_{}'.format(model_type, n_ues)
transducer_root = './res/{}/{}'.format(model_type, sim_id)

# Visualization stuff
alpha = 1
stacked = False


def prepare_ue_mobility(df, t_start, n):
    cab_ids = list(df.cab_id.unique())
    ue_mobilities = dict()
    i = 0
    for cab_id in cab_ids:
        i += 1
        if i > n:
            break
        data = df[df.cab_id == cab_id][['epoch', 'x', 'y']]
        mobility = ('history', {'t_start': t_start, 'history': data.values})
        ue_mobilities[cab_id] = mobility
    return ue_mobilities


if __name__ == '__main__':

    # Create instance of Mercury Framework
    summersim = Mercury()

    # GENERAL STUFF
    summersim.fog_model.define_rac_config(header=0, pss_period=1e-3, rrc_period=1e-3, timeout=0.2, bypass_amf=False)
    summersim.fog_model.define_fed_mgmt_config(header=0, edc_report_data=0)
    summersim.fog_model.define_network_config(header=0)

    # CROSSHAUL STUFF
    xh_link = LinkConfiguration(bandwidth=1e9, carrier_freq=0, prop_speed=2e8, penalty_delay=0, loss_prob=0, header=0,
                                att_name='fiber', att_config={'loss_factor': 0.3},
                                noise_name='thermal', noise_config=None)
    xh_tx = TransceiverConfiguration(tx_power=10, gain=0, noise_name='thermal', default_eff=10)
    summersim.fog_model.add_crosshaul_config(base_link_config=xh_link, base_trx_config=xh_tx)

    # RADIO STUFF
    dl_link = LinkConfiguration(bandwidth=100e6, carrier_freq=4.2e9, prop_speed=3e8, penalty_delay=0, loss_prob=0,
                                header=0, att_name='fspl', noise_name='thermal')
    ul_link = LinkConfiguration(bandwidth=100e6, carrier_freq=3.3e9, prop_speed=3e8, penalty_delay=0, loss_prob=0,
                                header=0, att_name='fspl', noise_name='thermal')
    dl_tx = TransceiverConfiguration(tx_power=50, gain=0, noise_name='thermal', noise_conf={'temperature': 300},
                                     mcs_table=RadioConfiguration.DL_MCS_TABLE_5G)
    ul_tx = TransceiverConfiguration(tx_power=30, gain=0, noise_name='thermal', noise_conf={'temperature': 300},
                                     mcs_table=RadioConfiguration.UL_MCS_TABLE_5G)
    summersim.fog_model.add_radio_config(base_dl_config=dl_link, base_ul_config=ul_link, base_ap_antenna=dl_tx,
                                         base_ue_antenna=ul_tx, channel_div_name='proportional')
    

    # Define SDN strategy A2C
    model_folder = sys.argv[1]
    env_id = sys.argv[2] # Path for trajectories
    
    if env_id == "base":

        # Define SDN strategy closest
        summersim.fog_model.add_core_config(amf_id='amf', sdn_controller_id='sdnc', core_location=(0, 0))

    else:
        
        summersim.fog_model.add_custom_sdn_strategy('rl_agent', SDNRLAgentStrategy)

        sdn_strategy_config = {
            'model_folder': model_folder,
            'env_id': env_id,
            'scale_dict': {"power": (944.16, 183.39),
                           "util": (656.54, 222.56),
                           "dist": (1088.39, 540.25)}
        }
        
        summersim.fog_model.add_core_config(amf_id='amf',
                                    sdn_controller_id='sdn_controller',
                                    core_location=(0, 0),
                                    sdn_strategy_name='rl_agent',
                                    sdn_strategy_config=sdn_strategy_config)

    # Define EDC dispatching strategy.
    r_manager_config = {
        'hw_dvfs_mode': hw_dvfs_mode,
        'hw_power_off': hw_power_off,
        'n_hot_standby': n_hot_standby,
        'disp_strategy_name': disp_strategy,
        'disp_strategy_config': {},
    }

    # Define services configurations
    summersim.fog_model.define_service_config(service_id='adas',  # Service ID must be unique
                                              service_u=25,  # Required Service standard std_u factor
                                              header=0,  # header of any service package
                                              generation_rate=1e6,  # data stream size in bits per second
                                              packaging_time=0.5,  # service packaging period in seconds
                                              min_closed_t=0.5,  # minimum time to stay closed for service sessions in seconds
                                              min_open_t=10,  # minimum time to stay open for service sessions in seconds
                                              service_timeout=0.2,  # service requests timeout in seconds
                                              window_size=1)  # Number of requests to be sent simultaneously with no acknowledgement
    
    # Load EDCs location
    edcs = pd.read_csv('scenarios/elements/edc_location.csv').values
    
    # Define and add air EDC
    summersim.fog_model.add_custom_pu_power_model('air_units', AirProcessingUnit)
    summersim.fog_model.add_custom_rack_node_power_model('air_cooling', AirCoolingSystem)

    summersim.fog_model.define_p_unit_config(p_unit_id='air_units',  # The ID must be unique
                                             max_u=100,  # Standard to Specific Utilization Factor
                                             max_start_stop=max_start_stop,  # Maximum number of simulatenous start/stop services
                                             dvfs_table={  # DVFS Table. At least configuration for 100% is required
                                                100: {
                                                    'memory_clock': 2000, # rx570: 1750
                                                    'core_clock': 1366 # rx570: 1284
                                                }
                                             },
                                             t_on=1,  # Time required for switching on the processing unit
                                             t_off=1,  # Time required for switching off the processing unit
                                             t_start=0.2,
                                             t_stop=0.2,
                                             t_operation=0.1,  # Time required for performing an operation
                                             pwr_model_name='air_units',
                                             pwr_model_config={'file_path_to_model_and_scalers': 'rl/mercury_models/processing_units/air/rx580/'})

    pwr_model_config = {
        'inlet_temp': 20,
        'consumption_factor': 1.0
    }

    summersim.fog_model.define_edc_rack_type(rack_id='air_cooling',
                                             pwr_model_name='air_cooling',
                                             pwr_model_config=pwr_model_config)                                          

    edc_racks_map = {'rack_1': ('air_cooling', ['air_units'] * gpus_per_edc)}
    summersim.define_standard_edc_config(edc_racks_map=edc_racks_map,
                                         resource_manager_config=r_manager_config,
                                         env_temp=298)

    summersim.add_edc(edcs[0, 0], (edcs[0, 1], edcs[0, 2]))
    summersim.add_edc(edcs[1, 0], (edcs[1, 1], edcs[1, 2]))

    #Define and add immersion EDC
    summersim.fog_model.add_custom_pu_power_model('immersion_units', ImmersionProcessingUnit)
    summersim.fog_model.add_custom_rack_node_power_model('immersion_cooling', ImmersionCoolingSystem)

    summersim.fog_model.define_p_unit_config(p_unit_id='immersion_units',  # The ID must be unique
                                             max_u=100,  # Standard to Specific Utilization Factor
                                             max_start_stop=max_start_stop,  # Maximum number of simulatenous start/stop services
                                             dvfs_table={  # DVFS Table. At least configuration for 100% is required
                                                100: {
                                                    'memory_clock': 1750, # rx570: 1750, rx580: 2000
                                                    'core_clock': 1284 # rx570: 1284, rx580: 1366
                                                }
                                             },
                                             t_on=1,  # Time required for switching on the processing unit
                                             t_off=1,  # Time required for switching off the processing unit
                                             t_start=0.2,
                                             t_stop=0.2,
                                             t_operation=0.1,  # Time required for performing an operation
                                             pwr_model_name='immersion_units',
                                             pwr_model_config={'file_path_to_model_and_scalers': 'rl/mercury_models/processing_units/immersion/rx570/'})

    pwr_model_config = {
        'delta_temp': 0.042,
        'consumption_factor': 0.03
    }

    summersim.fog_model.define_edc_rack_type(rack_id='immersion_cooling',
                                             pwr_model_name='immersion_cooling',
                                             pwr_model_config=pwr_model_config)

    edc_racks_map = {'rack_1': ('immersion_cooling', ['immersion_units'] * gpus_per_edc)}
    summersim.define_standard_edc_config(edc_racks_map=edc_racks_map,
                                         resource_manager_config=r_manager_config,
                                         env_temp=298)
    
    summersim.add_edc(edcs[2, 0], (edcs[2, 1], edcs[2, 2]))
        
    # Load and add APs
    aps = pd.read_csv('scenarios/elements/ap_location.csv').values
    for i in range(min(n_aps, aps.shape[0])):
        summersim.add_ap(aps[i, 0], (aps[i, 1], aps[i, 2]))

    summersim.fog_model.set_max_guard_time(max_guard_time)
    # Define standard Fog Node configuration
    # add UEs
    df = pd.read_csv('scenarios/elements/ue_mobility.csv', index_col=None)
    ue_mobilities = prepare_ue_mobility(df, 1212802200, n_ues)
    for ue_id, ue_mobility in ue_mobilities.items():
        summersim.add_ue(ue_id, ['adas'], ue_mobility[0], ue_mobility[1])

    #summersim.add_transducers('mysql', db='summersim20_' + sim_id)
    #summersim.add_transducers('csv')

    summersim.fog_model_quick_build(lite=lite, shortcut=shortcut)
    summersim.fog_model.initialize_coordinator()
    summersim.fog_model.start_simulation(time_interv=sim_time)
    
    """
    summersim.plot_delay(alpha)
    if not lite and not shortcut:
        summersim.plot_ul_bw(alpha)
        summersim.plot_dl_bw(alpha)
    
    summersim.plot_edc_utilization(stacked, alpha)
    summersim.plot_edc_power_demand(stacked, alpha)
    """
