import csv
import os
import numpy as np
from mercury.fog_model.core.sdn_strategy import SDNStrategy
from .rl_agent import RLAgent

class SDNRLAgentStrategy(SDNStrategy):
    def __init__(self, aps_location, edcs_location, services_id, **kwargs):
        super(SDNRLAgentStrategy, self).__init__(aps_location, edcs_location, services_id, **kwargs)

        model_folder = kwargs.get('model_folder')
        env_id = kwargs.get('env_id')
        self.scale_dict = kwargs.get('scale_dict')

        self.trajectory_path = (model_folder+'current_trajectories/n_{}.csv').format(env_id)
        self.rl_agent = RLAgent(model_folder)
        self.last_edc = None
        
        
    def assign_edc(self, ap_id):
        """
        From a list of available Edge Data Centers, the closest to a given Access Point is chosen
        :param str ap_id: ID of the Access Point
        :return: dictionary {service_id: edc_id} with the most suitable EDC for each service
        """

        # for each service routing e.g. adas
        service_routing = {service_id: None for service_id in self.services_id}
        for service_id in self.services_id:

            # For each timestep select edc (just one ap the rest select the same)
            if ap_id == "ap_0":

                # Getting observation (state) from edc_report as P and U
                state = []
                for _, edc_report in self.edc_reports.items():
                    # Sometines at the beginning the report is None
                    if edc_report is not None:
                        state.extend([self.scale_data(edc_report.overall_std_u, "util"),
                                      self.scale_data(edc_report.power_demand, "power"),
                                      self.scale_data(edc_report.it_power, "power"),
                                      self.scale_data(edc_report.cooling_power, "power")])
                    else:
                        service_routing[service_id] = self.closest_edc(ap_id, service_id)
                        continue
                np_state = np.array(state)

                # NN chosing from a distribution the edc for this timestep
                _, policy_dist = self.rl_agent.forward(np_state)
                dist = policy_dist.detach().numpy()
                action_index = np.random.choice(dist.shape[1], p=np.squeeze(dist))
                action = "edc_{}".format(action_index)

                # Computing penalty (if the edc is available)
                penalty = not self.is_available(action, service_id)

                # Appending row to trajectory file
                state.extend([action, penalty])
                self.append_row(state)

                # Passing selected edc_id, if penalty passing closest edc instead
                service_routing[service_id] = self.closest_edc(ap_id, service_id) if penalty else action

                # The other APs that belongs to the same timestep select the same edc
                self.last_edc = service_routing[service_id]
            else:
                service_routing[service_id] = self.last_edc

        return service_routing

    def scale_data(self, data, feature):
        scale_pair = self.scale_dict[feature]
        return (data - scale_pair[0]) / (scale_pair[1] + np.finfo(np.float32).eps)

    def append_row(self, row):
        with open(self.trajectory_path, 'a') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(row)

    def closest_edc(self, ap_id, service_id):
        for edc_id, _ in self.distances[ap_id]:
            if self.is_available(edc_id, service_id):
                return edc_id
        return ValueError
