import numpy as np
import json
import argparse
import datetime
from itertools import accumulate
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
    
######################################################### Default offers ########################################################
normal_charge_mc = 2
normal_charge_mq = 100

normal_discharge_mc = 30
normal_discharge_mq = 100

extreme_charge_mc = 1
extreme_charge_mq = 10

extreme_discharge_mc = 50
extreme_discharge_mq = 10

########################################################## Input Data ###########################################################
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

########################################### Day-ahead Market ############################################
if 'DAM' in market_type:
    prices_forecast = market_info['previous'][market_type]['prices']['EN'][bus_value]
    required_times = [t for t in market_info['timestamps']]
    price_dict = {required_times[i]:prices_forecast[i] for i in range(len(required_times))}
    # Writing prices to a local JSON file
    file_path = "da_prices.json"
    with open(file_path, "w") as file:
        json.dump(price_dict, file)
    prices = np.array(prices_forecast)

    # Make the offer curves and unload into arrays
    offer_dis, offer_ch = da.da_offers(prices_forecast,initial_energy,bus_value,rid)

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
            list_charge_mc.append(extreme_charge_mc)
            list_charge_mq.append(extreme_charge_mq)
        block_ch_mc[required_times[t]] = list_charge_mc
        block_ch_mq[required_times[t]] = list_charge_mq

    block_dc_mc = {}
    block_dc_mq = {}
    for t in range(total_period):
        list_discharge_mc = []
        list_discharge_mq = []
        for i in range(len(discharge_t)):
            if discharge_t[i] == t+1:
                list_discharge_mc.append(discharge_mc[i])
                list_discharge_mq.append(discharge_mq[i])
        if len(list_discharge_mc) == 0:
            list_discharge_mc.append(extreme_discharge_mc)
            list_discharge_mq.append(extreme_discharge_mc)
        block_dc_mc[required_times[t]] = list_discharge_mc
        block_dc_mq[required_times[t]] = list_discharge_mq

    block_soc_mq = {}
    block_soc_mc = {}
    for t in range(total_period):
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
    price_path = "da_prices.json"
    with open(price_path, "r") as file:
        prices = json.load(file)
        dam_times = [key for key in prices.keys()]
        prices = [value for value in prices.values()]
    # Read in information from the resource
    en_schedule_list = [z[0] for z in resource_info["ledger"][rid]["EN"].values()]
    sch_time = [z for z in resource_info["ledger"][rid]["EN"].keys()]
    initial_soc = resource_info["status"][rid]["soc"]
    marginal_quantities, marginal_prices = zip(*en_schedule_list)
    adjusted_marginal_quantities = [initial_soc] + list(marginal_quantities)
    # Calculating the cumulative sum of adjusted marginal quantities
    cumulative_soc = list(accumulate(adjusted_marginal_quantities))
    #print("cumulative soc is", cumulative_soc)
    # Generating a new list of cumulative soc and marginal prices
    soc_price_list = list(zip(cumulative_soc, marginal_prices))
    #print(soc_price_list)
    required_times = [t for t in market_info['timestamps']]
    # Convert the offer curves to timestamp:offer_value dictionaries
    #soc_mq =soc_price_list.values
    soc_mq =cumulative_soc
    soc_mc =prices
    time_soc = np.zeros(len(required_times))
    price_soc = np.zeros(len(required_times))
    soc = np.zeros(len(required_times))
    ti =0
    idx_price = 0
    for i in range(len(required_times)):
        # real-time datetime format
        dt1 = datetime.datetime.strptime(required_times[i], '%Y%m%d%H%M')
        # resource json file datetime format
        dt2 = datetime.datetime.strptime(sch_time[ti], '%Y%m%d%H%M')
        dt3 = datetime.datetime.strptime(dam_times[idx_price], '%Y%m%d%H%M')
        while dt1 > dt3 and dt1.hour != dt3.hour:
            idx_price += 1
            dt3 = datetime.datetime.strptime(dam_times[idx_price], '%Y%m%d%H%M')
        if idx_price == len(dam_times)-1:
            price_soc[i] = 0
        else:
            if marginal_quantities[ti]>0:
                multiplier = 1
            else:
                multiplier = -1
            price_soc[i] = prices[idx_price+1]*multiplier

        while dt1 > dt2 and dt1.hour != dt2.hour:
            ti += 1
            dt2 = datetime.datetime.strptime(sch_time[ti], '%Y%m%d%H%M')

        if dt1.hour == dt2.hour:
            soc[i] = soc_mq[ti+1]
        elif ti == 0:
            soc[i] = initial_soc
        else:
            soc[i] = soc_mq[ti]

    block_soc_mq = {}
    for i, soc_mq in enumerate(required_times):
        block_soc_mq[required_times[i]] = float(soc[i])
    block_soc_mc = {}
    for i, soc_mc in enumerate(required_times):
        block_soc_mc[required_times[i]] = float(price_soc[i])

    # Convert the offer curves to timestamp:offer_value dictionaries
    charge_mc = np.zeros(total_period)
    charge_mq = np.zeros(total_period)
    discharge_mc = np.zeros(total_period)
    discharge_mq = np.zeros(total_period)

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
    offer_out_dict = {rid:{}}
    offer_out_dict[rid] = {"block_ch_mc":block_ch_mc, "block_ch_mq":block_ch_mq, "block_dc_mc":block_dc_mc, "block_dc_mq":block_dc_mq, "block_soc_mc":block_soc_mc, "block_soc_mq":block_soc_mq}
    offer_out_dict[rid].update(rgu_dict)
    offer_out_dict[rid].update(max_dict)
    offer_out_dict[rid].update(constants)
    # Save as json file in the current directory with name offer_{time_step}.json
    with open(f'offer_{time_step}.json', 'w') as f:
        json.dump(offer_out_dict, f, indent=4, cls=NpEncoder)
