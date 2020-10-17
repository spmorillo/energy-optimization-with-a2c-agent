from tensorflow.keras.models import model_from_json
from joblib import load, dump
from sklearn.preprocessing import MinMaxScaler
from numpy import zeros as np_zeros

class Rx580PowerModel():
    def __init__(self, scaler_x, scaler_y):
        """
        Create object for Rx580 power estimation
        :param str file_path_to_model_and_scalers: File path to files containing model's weights (h5), scalers (save)
        and arch (json), extension of files not needed
        :raises FileNotFoundError
        """
        super().__init__()
        with open('model.json', 'r') as f:
            self.model = model_from_json(f.read())
        self.model.load_weights('weights.h5')
        self.scaler_x = scaler_x 
        self.scaler_y = scaler_y

    def compute_power(self, status, utilization, max_u, dvfs_index, dvfs_table):  # TODO con max_u
        res = 0
        if status:
            x = np_zeros((1, 3))
            x[0, 0] = dvfs_table[dvfs_index]['memory_clock']
            x[0, 1] = dvfs_table[dvfs_index]['core_clock']
            x[0, 2] = utilization / 20.0  # TODO esto es lo raro
            y = self.model.predict(self.scaler_x.transform(x))
            res = self.scaler_y.inverse_transform(y)[0, 0]
        return res

if __name__ == '__main__':


    dvfs_table = [{
        'memory_clock': 2000,
        'core_clock': 1366
    }]

    scaler_x = load('scaler_x.save')
    scaler_y = load('scaler_y.save')
    rx580 = Rx580PowerModel(scaler_x, scaler_y)
    print(rx580.scaler_x.data_min_)
    print(rx580.scaler_x.data_max_)
    print(rx580.scaler_y.data_min_)
    print(rx580.scaler_y.data_max_)
    print(rx580.scaler_x.scale_)
    print(rx580.scaler_y.scale_)
    print(rx580.compute_power(1,100,100,0,dvfs_table))

    print("\nnuevo\n")

    sc_x = MinMaxScaler([-1,1])
    sc_y = MinMaxScaler([-1,1])
    sc_x.fit(X=[[300,300,1],[2000,1366,5]])
    sc_y.fit(X=[[33.38],[123.15]])
    rx580 = Rx580PowerModel(sc_x, sc_y)
    print(sc_x.data_min_)
    print(sc_x.data_max_)
    print(sc_y.data_min_)
    print(sc_y.data_max_)
    print(sc_x.scale_)
    print(sc_y.scale_)
    print(rx580.compute_power(1,100,100,0,dvfs_table))

    dump(sc_x, "sc_x.joblib")
    dump(sc_y, "sc_y.joblib")
    