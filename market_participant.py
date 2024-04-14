import numpy as np
import json
import argparse
import da_offers as da
# import rt_offers as rt

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

########################################################## Input Data ###########################################################
#################################################################################################################################
if __name__ == '__main__':
    # Add argument parser for three required input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('time_step', type=int, help='Integer time step tracking the progress of the\
                        simulated market.')
    parser.add_argument('market_file', help='path to json formatted dictionary with market \
                        information.')
    parser.add_argument('resource_file', help='path to json formatted dictionary with resource \
                        information.')

    args = parser.parse_args()

    # Parse json inputs into python dictionaries
    time_step = args.time_step
    with open(args.market_file, 'r') as f:
        market_info = json.load(f)
    with open(args.resource_file, 'r') as f:
        resource_info = json.load(f)

# Extract resources rid
rid = resource_info['rid']
# Extract battery location
bus_value = resource_info['bus']
# Extract initial energy (between 0 and storage_capacity)
initial_energy = resource_info['status'][rid]['soc']
# Extract market type
market_type = market_info['market_type']
# Extract the forecast price: length = 36, type = list
prices_forecast = market_info['previous'][market_type]['prices']['EN'][bus_value]
total_period = len(prices_forecast)

########################################################## Make Offers ##########################################################
#################################################################################################################################
if 'DAM' in market_type:
    prices_forecast = market_info['previous'][market_type]['prices']['EN'][bus_value]
    required_times = [t for t in market_info['timestamps']]

    # Make the offer curves and unload into arrays
    offer_dis, offer_ch = da.da_offers(prices_forecast,bus_value,initial_energy)

    charge_mc = offer_ch['BinStart'].values
    charge_mq = offer_ch['ChValue'].values
    charge_t = offer_ch['Period'].values

    discharge_mc = offer_dis['Price'].values
    discharge_mq = offer_dis['DisValue'].values
    discharge_t = offer_dis['Period'].values

    # Convert the offer curves to timestamp:offer_value dictionaries
    block_ch_mc = {}
    block_ch_mq = {}
    for t in range(total_period):
        list_charge_mc = []
        list_charge_mq = []
        for i in range(len(charge_t)):
            if charge_t[i] == t+1:
                list_charge_mc.append(charge_mc[i])
                list_charge_mq.append(charge_mq[i])
        if len(list_charge_mc) == 0:
            list_charge_mc.append(-100)
            list_charge_mq.append(10)
        block_ch_mc[required_times[t]] = list_charge_mc
        block_ch_mq[required_times[t]] = list_charge_mq

    block_dc_mc = {}
    block_dc_mq = {}
    block_soc_mq = {}
    block_soc_mc = {}
    for t in range(total_period):
        list_discharge_mc = []
        list_discharge_mq = []
        for i in range(len(discharge_t)):
            if discharge_t[i] == t+1:
                list_discharge_mc.append(discharge_mc[i])
                list_discharge_mq.append(discharge_mq[i])
        if len(list_discharge_mc) == 0:
            list_discharge_mc.append(100)
            list_discharge_mq.append(10)
        block_dc_mc[required_times[t]] = list_discharge_mc
        block_dc_mq[required_times[t]] = list_discharge_mq
        block_soc_mc[required_times[t]] = 0
        block_soc_mq[required_times[t]] = 0

    reg = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp']
    zero_arr = np.zeros(len(required_times))
    rgu_dict = {}
    for r in reg:
        rgu_dict[r] = {}
        for t in required_times:
            rgu_dict[r][t] = 0

    max_dict = {}
    for mx in ['chmax', 'dcmax']:
        max_dict[mx] = {}
        for t in required_times:
            max_dict[mx][t] = 125

    constants = {}
    constants['soc_begin'] = 128
    constants['init_en'] = 0
    constants['init_status'] = 0
    constants['ramp_dn'] = 9999
    constants['ramp_up'] = 9999
    constants['socmax'] = 608
    constants['socmin'] = 128
    constants['eff_ch'] = 0.892
    constants['eff_dc'] = 1.0
    constants['soc_end'] = 128
    constants['bid_soc'] = False

    # Pacakge the dictionaries into an output formatted dictionary
    # rid = 'R00229'
    offer_out_dict = {rid:{}}
    offer_out_dict[rid] = {"block_ch_mc":block_ch_mc, "block_ch_mq":block_ch_mq, "block_dc_mc":block_dc_mc, "block_dc_mq":block_dc_mq, "block_soc_mc":block_soc_mc, "block_soc_mq":block_soc_mq}
    offer_out_dict[rid].update(rgu_dict)
    offer_out_dict[rid].update(max_dict)
    offer_out_dict[rid].update(constants)

    # Save as json file in the current directory with name offer_{time_step}.json
    with open(f'offer_{time_step}.json', 'w') as f:
        json.dump(offer_out_dict, f, indent=4, cls=NpEncoder)

########################################### Real-time Market ############################################
elif 'RTM' in market_type:
    required_times = [t for t in market_info['timestamps']]
    block_soc_mq = {}
    for i, soc_mq in enumerate(required_times):
        block_soc_mq[required_times[i]] = 1
    block_soc_mc = {}
    for i, soc_mc in enumerate(required_times):
        block_soc_mc[required_times[i]] = 100

    # Convert the offer curves to timestamp:offer_value dictionaries
    charge_mc = np.zeros(36)
    charge_mq = np.zeros(36)
    discharge_mc = np.zeros(36)
    discharge_mq = np.zeros(36)
    block_ch_mc = {}
    for i, cost in enumerate(charge_mc):
        block_ch_mc[required_times[i]] = 0

    block_ch_mq = {}
    for i, power in enumerate(charge_mq):
        block_ch_mq[required_times[i]] = 0 

    block_dc_mc = {}

    for i, cost in enumerate(discharge_mc):
        block_dc_mc[required_times[i]] = 0

    block_dc_mq = {}
    for i, power in enumerate(discharge_mq):
        block_dc_mq[required_times[i]] = 0 

    reg = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp']
    zero_arr = np.zeros(len(required_times))
    rgu_dict = {}
    for r in reg:
        rgu_dict[r] = {}
        for t in required_times:
            rgu_dict[r][t] = 0

    max_dict = {}
    for mx in ['chmax', 'dcmax']:
        max_dict[mx] = {}
        for t in required_times:
            max_dict[mx][t] = 125

    constants = {}
    constants['soc_begin'] = 128
    constants['init_en'] = 0
    constants['init_status'] = 0
    constants['ramp_dn'] = 9999
    constants['ramp_up'] = 9999
    constants['socmax'] = 608
    constants['socmin'] = 128
    constants['eff_ch'] = 0.892
    constants['eff_dc'] = 1.0
    constants['soc_end'] = 128
    constants['bid_soc'] = True

    # Pacakge the dictionaries into an output formatted dictionary
    # rid = 'R00229'
    offer_out_dict = {rid:{}}
    offer_out_dict[rid] = {"block_ch_mc":block_ch_mc, "block_ch_mq":block_ch_mq, "block_dc_mc":block_dc_mc, "block_dc_mq":block_dc_mq, "block_soc_mc":block_soc_mc, "block_soc_mq":block_soc_mq}
    offer_out_dict[rid].update(rgu_dict)
    offer_out_dict[rid].update(max_dict)
    offer_out_dict[rid].update(constants)
    # Save as json file in the current directory with name offer_{time_step}.json
    with open(f'offer_{time_step}.json', 'w') as f:
        json.dump(offer_out_dict, f, indent=4, cls=NpEncoder)
