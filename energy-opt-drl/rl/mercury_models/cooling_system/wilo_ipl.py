from mercury.fog_model.edge_fed.edc.rack.rack_node.rack_pwr import RackPowerModel
from mercury.fog_model.edge_fed.edc.rack.rack_node.rack_temp import RackTemperatureModel
from scipy.interpolate import interp1d



class WiloIPLPowerModel(RackPowerModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.c_p = kwargs.get('c_p', 4.18)  # TODO hacer esto más flexible para otras cosas
        self.delta_temp = kwargs.get('delta_temp', 20)
        self.consumption_factor = kwargs.get('consumption_factor', 1)

        pump_data_original = {
            "flow": [3,16,20,24,28,32],
            "P_torque": [0.50,0.75,0.78,0.80,0.80,0.79],
            "efficiency": [0.20,0.61,0.63,0.60,0.50,0.38]
        }
        pump_data = kwargs.get('pump_data', pump_data_original)
        self.min_flow = pump_data["flow"][0]
        self.max_flow = pump_data["flow"][-1]
        self.p_torque_model = interp1d(pump_data["flow"],
                                       pump_data["P_torque"],
                                       kind='quadratic')
        self.efficiency_model = interp1d(pump_data["flow"],
                                    pump_data["efficiency"],
                                    kind='quadratic')

    def compute_rack_power(self, it_power, rack_temp: float, env_temp: float) -> float:
        
        # compute flow and check if it's within model rang
        flow = it_power / (277.78 * self.c_p * self.delta_temp)
        if flow < self.min_flow:
            flow = self.min_flow
        elif flow >= self.max_flow:
            flow = self.max_flow

        # compute cooling power (Pin) with torque power (Pout) and pump efficiency
        p_torque = self.p_torque_model(flow)
        efficiency = self.efficiency_model(flow)
        return p_torque * 1000 * self.consumption_factor / efficiency
  

