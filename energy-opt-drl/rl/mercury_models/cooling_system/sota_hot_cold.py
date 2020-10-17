from mercury.fog_model.edge_fed.edc.rack.rack_node.rack_pwr import RackPowerModel
from mercury.fog_model.edge_fed.edc.rack.rack_node.rack_temp import RackTemperatureModel
from scipy.interpolate import interp1d



class AirCoolingSystem(RackPowerModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.inlet_temp = kwargs.get('inlet_temp', 24)  # TODO: Modelo de temperatura
        self.consumption_factor = kwargs.get('consumption_factor', 1.0)
    

    def compute_rack_power(self, it_power, rack_temp: float, env_temp: float) -> float:

        COP = 0.0068 * self.inlet_temp**2 + 0.0008 * self.inlet_temp + 0.458
        return self.consumption_factor * (it_power / COP)
  

