from GeoSim.sim import GeoSim
import numpy as np
import csv
import torch
import pandas as pd
import pickle


from wf_demo.default_load import input_dict, load_default_latent_tensor

class RealTruth:
    " Replacement for SyntheticTruth. Uses a real GeoSphere HD channel-response file instead of generating observations from GeoSim."
    #Mapping of data types to channel names in the real data file.
    CHANNEL_MAP = { 
        ('6kHz', '83ft'): ['USDA_F2_R2', 'USDP_F2_R2', 'UADA_F2_R2', 'UADP_F2_R2', 'UHRA_F2_R2', 'UHRP_F2_R2', 'UHAA_F2_R2', 'UHAP_F2_R2'],
        ('12kHz', '83ft'): ['USDA_F3_R2', 'USDP_F3_R2', 'UADA_F3_R2', 'UADP_F3_R2', 'UHRA_F3_R2', 'UHRP_F3_R2', 'UHAA_F3_R2', 'UHAP_F3_R2'],
        ('24kHz', '83ft'): ['USDA_F4_R2', 'USDP_F4_R2', 'UADA_F4_R2', 'UADP_F4_R2', 'UHRA_F4_R2', 'UHRP_F4_R2', 'UHAA_F4_R2', 'UHAP_F4_R2'],
        ('24kHz', '43ft'): ['USDA_F4_R1', 'USDP_F4_R1', 'UADA_F4_R1', 'UADP_F4_R1', 'UHRA_F4_R1', 'UHRP_F4_R1', 'UHAA_F4_R1', 'UHAP_F4_R1'],
        ('48kHz', '43ft'): ['USDA_F5_R1', 'USDP_F5_R1', 'UADA_F5_R1', 'UADP_F5_R1', 'UHRA_F5_R1', 'UHRP_F5_R1', 'UHAA_F5_R1', 'UHAP_F5_R1'],
        ('96kHz', '43ft'): ['USDA_F6_R1', 'USDP_F6_R1', 'UADA_F6_R1', 'UADP_F6_R1', 'UHRA_F6_R1', 'UHRP_F6_R1', 'UHAA_F6_R1', 'UHAP_F6_R1']
        }

    def __init__(self, udar_file, all_data_types):
        self.all_data_types = all_data_types
        #self.df = pd.read_csv(udar_file, delim_whitespace=True, comment='%')
        # Read the first line to get the header, then read the rest of the file into a DataFrame, avoid errors with comment lines
        with open(udar_file, "r") as f:
            header = f.readline().strip()
        if header.startswith("%"):
            header = header[1:].strip()
        columns = header.split()
        self.df = pd.read_csv(udar_file, sep=r"\s+", skiprows=1, names=columns)

        # Verify required columns exist
        required_cols = []
        for cols in self.CHANNEL_MAP.values():
            required_cols.extend(cols)
        
        missing = [c for c in required_cols if c not in self.df.columns]
        if len(missing):
            raise ValueError(f"Missing required channels: {missing}")

        # bookkeeping files exactly as SyntheticTruth
        with open('../data/datatyp.csv', 'w', newline='') as f:
            csv.writer(f).writerow([str(el) for el in self.all_data_types])

        with open('../data/assim_index.csv', 'w') as f:
            f.write('0\n')

    def _extract_logs_np(self, real_row):
        logs_np = []
        for datatype in self.all_data_types:
            channel_names = self.CHANNEL_MAP[datatype]
            values = [float(real_row[ch]) for ch in channel_names]
            logs_np.append(values)

        return np.asarray(logs_np)

    def acquire_data(self, keys):
        # DISTINGUISH drilling position
        bit_col = keys['bit_pos'][0][1]
        # Synthetic is every 10 m
        # Real data is every 1 m
        real_idx = bit_col * 10

        if real_idx >= len(self.df):
            raise IndexError(
                f"Real data index {real_idx} exceeds "
                f"file length {len(self.df)}"
            )

        real_row = self.df.iloc[real_idx]
        logs_np = self._extract_logs_np(real_row)
        # Same output format as SyntheticTruth

        data = {}
        var = {}

        for count, datatype in enumerate(self.all_data_types):
            data[datatype] = [logs_np[count, :]]
            # Keep current synthetic variance model
            # until real uncertainty is defined

            var[datatype] = [['ABS', [(0.1 * np.max(np.abs(val))) ** 2 for val in logs_np[count, :]]]]

        df_data = pd.DataFrame(data, columns=self.all_data_types, index=[0])
        df_data.index.name = 'tvd'

        df_data.to_pickle('../data/data.pkl')

        df_var = pd.DataFrame(var, columns=self.all_data_types, index=[0])
        df_var.index.name = 'tvd'

        df_var.to_csv('../data/var.csv', index=True)

        with open('../data/var.pkl', 'wb') as f:
            pickle.dump(df_var, f)

        with open('../data/assim_index.csv', 'w') as f:
            f.write('0\n')

        with open('../data/datatyp.csv', 'w', newline='') as f:
            csv.writer(f).writerow([str(el) for el in self.all_data_types])

class SyntheticTruth:
    def __init__(self, latent_truth_vector, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # todo make this simulator with less stuff included
        # this is simulator of the true data
        self.simulator = GeoSim(input_dict)

        self.simulator.l_prim = [0]
        self.simulator.all_data_types = input_dict['datatype']

        self.latent_synthetic_truth = latent_truth_vector
        # load_default_latent_tensor().to(device))

        # intialize
        l = open('../data/datatyp.csv', 'w', newline='')
        writer5 = csv.writer(l)
        writer5.writerow([str(el) for el in self.simulator.all_data_types])
        l.close()

        k = open('../data/assim_index.csv', 'w', newline='')
        for c, _ in enumerate([0]):
            k.writelines(str(c) + '\n')
        k.close()

    def acquire_data(self, keys):
        # todo fix empty index vector
        # this is the old index vector size:
        print(f"size=(1, keys['bit_pos'][0][1]) {(1, keys['bit_pos'][0][1])}")
        print(f"fill_value=keys['bit_pos'][0][0] {keys['bit_pos'][0][0]}")
        # the size of the index vector per our convention should cover up to bit_pos [1]
        # this means that it should be position plus one
        # todo we need to update the convension in the simulator
        index_vector = torch.full(size=(1, keys['bit_pos'][0][1]+1),
                                  fill_value=keys['bit_pos'][0][0],
                                  dtype=torch.long
                                  ).to(self.device)

        logs = self.simulator.NNmodel.forward(self.latent_synthetic_truth, index_vector, output_transien_results=False)

        # here we need to use the bit_pos (but was bit_pos-1)
        if self.simulator.all_data_types == ['point']:
            logs_np = logs.cpu().detach().numpy()[0,keys['bit_pos'][0][1],:]
        else:
            logs_np = logs.cpu().detach().numpy()[0,keys['bit_pos'][0][1],:,-8:]
        # todo describe which logs are used in the paper

        # bookkeeping
        k = open('../data/assim_index.csv', 'w', newline='')
        # writer4 = csv.writer(k)
        l = open('../data/datatyp.csv', 'w', newline='')
        writer5 = csv.writer(l)

        # build a pandas dataframe with the data.
        # The tvd is the index and the tuple (freq,dist) is the columns

        data = {}
        var = {}
        for count, di in enumerate(self.simulator.all_data_types):
            if self.simulator.all_data_types == ['point']:
                data[di] = [logs_np]
                var[di] = [['ABS', [(0.001*np.mean(val))**2 for val in logs_np]]]
            else:
                freq, dist = di
                data[(freq, dist)] = [logs_np[count, :]]
                # var[(freq, dist)] = [[['REL', 10] if abs(el) > abs(0.1*np.mean(values)) else ['ABS', (0.1*np.mean(values))**2] for el in val] for val in values]
                var[(freq, dist)] = [['ABS' ,[(0.1 * np.max(np.abs(val))) ** 2 for val in logs_np[count, :]]]]
                # var[(freq, dist)] = [[['REL', 0.1] for val in logs_np[count, :]]]

        df = pd.DataFrame(data, columns=self.simulator.all_data_types, index=[0])
        df.index.name = 'tvd'
        # df.to_csv('data.csv',index=True)
        df.to_pickle('../data/data.pkl')

        df = pd.DataFrame(var, columns=self.simulator.all_data_types, index=[0])
        df.index.name = 'tvd'
        df.to_csv('../data/var.csv', index=True)
        with open('../data/var.pkl', 'wb') as f:
            pickle.dump(df, f)

        # filt = [i*10 for i in range(50)]
        for c, _ in enumerate([0]):
            # if c in filt:
            k.writelines(str(c) + '\n')
        k.close()

        writer5.writerow([str(el) for el in self.simulator.all_data_types])
        l.close()


