from pyomo.environ import *
import numpy as np
import pandas as pd

def da_offers(prices_forecast,initial_energy,bus_value,rid):

    #################################################### Adjustable Parameters #######################################################
    if rid == 'R00233' and bus_value == 'CIPV':
        replacement_cost =  3.5e+5
        CVar_risk_weight = 0.5
        periodicity_constraint = True
    elif rid == 'R00235' and bus_value == 'NEVP':
        replacement_cost =  4e+5
        CVar_risk_weight = 0.8
        periodicity_constraint = True
    elif rid == 'R00236' and bus_value == 'NEVP':
        replacement_cost =  4.5e+5
        CVar_risk_weight = 0.2
        periodicity_constraint = False
    else:
        replacement_cost =  4.5e+5
        CVar_risk_weight = 0.5
        periodicity_constraint = True

    # number of newly generated scenarios
    num_scenario= 24

    # number of segments for degradation cost
    num_segments = 60          

    # chance constraint satisfication rate
    sigma_val = 0.95

    #################################################### Scenario Generation ########################################################
    # Read historical price data from .csv file
    historical_price = pd.read_csv('full_year_lmps.csv')
    n_days = 364
    chosen_place = np.reshape(np.array(historical_price[bus_value]), (n_days, 24))

    def euclidean_distance(curve1, curve2):
        return np.sqrt(np.sum((np.array(curve1) - np.array(curve2)) ** 2))

    def find_closest_days(predicted_curve, historical_data, num_scenario):
        distances = []
        for index, daily_prices in enumerate(historical_data):
            distance = euclidean_distance(predicted_curve, daily_prices)
            distances.append((index, distance))
        # Sort the list of tuples by the second item (distance) and return the first num_days elements
        closest_days = sorted(distances, key=lambda x: x[1])[:num_scenario]
        return closest_days

    closest_days = find_closest_days(prices_forecast[0:24], chosen_place, num_scenario)
    scenarios = []
    reciprocal_distances = []
    for day, distance in closest_days:
        scenarios.append(chosen_place[day].tolist())
        reciprocal_distances.append(1/distance)
    scenarios.append(prices_forecast[0:24])

    # The weight of each generated scenario is inversely proportional to the distance from the scenario to prices_forecast
    sum_reciprocal_distances = sum(reciprocal_distances)
    prob_instance = [d / sum_reciprocal_distances/2 for d in reciprocal_distances]
    # The weight of the prices_forecast is set to 0.5
    prob_instance.append(0.5)
    # Conver the list of scenarios to match num_instance = len(prices[0]) and num_periods = len(prices)
    transposed_scenarios = [list(row) for row in zip(*scenarios)]
    prices = transposed_scenarios
    # Convert the dictionary to a DataFrame
    df_prc = pd.DataFrame(prices)

    price_flatten = df_prc.values.flatten()
    single_row_df = pd.DataFrame([price_flatten])

    hist_values, bin_edges = pd.cut(single_row_df.values.flatten(), bins=10, retbins=True)
    # Get the counts for each bin
    hist_counts = hist_values.value_counts().sort_index()
    # Prepare data for Excel
    bin_data = {'Bin Start': bin_edges[:-1], 'Bin End': bin_edges[1:], 'Frequency': hist_counts}
    # Convert to DataFrame
    bin_df = pd.DataFrame(bin_data)

    ######################################################## Battery Parameters #####################################################
    num_instance = len(prices[0])  # Update number of periods based on the length of price
    num_periods = len(prices)  # Update number of periods based on the length of price
    SoC_min = 0
    SoC_max = 1
    b_1 = 5.24e-4
    b_2 = 2.03
    eta = 0.946             # Round-trip efficiency
    Crep =  replacement_cost/eta  # Replacement cost of battery
    # Crep =  (1.5e+5)* (1/eta)  # Replacement cost of bettery
    eta_ch = 0.892             # charging efficiency
    eta_dis = 1             # discharging efficiency
    storage_capacity = 640  # Example value, adjust as needed
    duration = 4            # maximum duration for battery to charge and discharge completely
    max_power = 125  # Maximum power is normalized to 1
    seg_power = 1 # Normalized

    initial_SOC = initial_energy/storage_capacity
    max_SoC = 0.95
    min_SoC = 0.2
    initial_segment = int(initial_SOC * num_segments) 
    # Initial state of charge (SoC) for each segment for discharge
    initial_soc_vector_disch = [1 if s <= initial_segment else 0 for s in range(1, num_segments + 1)]  # Adjusted initial SoC vector

    # Initial state of charge (SoC) for each segment for charge
    initial_soc_vector_ch = [1 if s > (num_segments-initial_segment) else 0 for s in range(1, num_segments + 1)]  # Adjusted initial SoC vector

    if 1 in initial_soc_vector_disch:
        # Find the index of the first occurrence of 0 after 1 in initial_soc_vector_disch
        first_zero_index_disch = initial_soc_vector_disch.index(0, initial_soc_vector_disch.index(1))
        # Update the value at that index
        initial_soc_vector_disch[first_zero_index_disch] = initial_SOC* num_segments - int(initial_SOC* num_segments)

    # Check if 1 exists in initial_soc_vector_ch before searching for 0
    if 1 in initial_soc_vector_ch:
        # Find the index of the last occurrence of 0 after 1 in initial_soc_vector_ch
        last_zero_index_ch = len(initial_soc_vector_ch) - initial_soc_vector_ch[::-1].index(0) - 1
        # Update the value at that index
        initial_soc_vector_ch[last_zero_index_ch] = initial_SOC* num_segments - int(initial_SOC* num_segments)

    ################################################### Degradation Penalty Function ################################################
    # Caculation of piecewise linear costs 
    # Define the phi function
    def phi(del_val):
        return b_1 * (del_val ** b_2)

    # Function to calculate segment costs
    def calculate_segment_costs(J):
        c = np.zeros(J)
        for j in range(1, J + 1):
            c[j-1] = J * (phi(j / J) - phi((j - 1) / J))
        return c

    # Calculate charging and discharging cost for each segment
    costs_charge_segment_norm = calculate_segment_costs(num_segments)
    costs_discharge_segment_norm = calculate_segment_costs(num_segments)

    costs_charge_segment =  [x * Crep for x in costs_charge_segment_norm]
    costs_discharge_segment = [x * Crep for x in costs_discharge_segment_norm]

    #################################################### Profit Optimization Model ##################################################
    # Create a model
    model = ConcreteModel()

    # Define indices
    model.instance = RangeSet(1, num_instance)
    model.segments = RangeSet(1, num_segments)
    model.periods = RangeSet(1, num_periods)

    # Define CVaR variables
    model.thet_var = Var()
    model.phi_instance = Var(model.instance, within=NonNegativeReals)

    #################### DISCHARGE VARIABLES ########################
    # Variables at segment level for discharge
    model.segment_charge_power_disch = Var(model.segments, model.periods, model.instance, bounds=(0, seg_power))
    model.segment_discharge_power_disch = Var(model.segments, model.periods, model.instance, bounds=(0, seg_power))
    model.SoC_disch = Var(model.segments, model.periods, model.instance, bounds=(SoC_min, SoC_max))

    # Define charge and discharge power variables at aggregate level for discharge
    model.charge_power_disch = Var(model.periods, model.instance, bounds=(0, max_power))
    model.discharge_power_disch = Var(model.periods, model.instance, bounds=(0, max_power))
    model.SoC_aggregate_disch = Var(model.periods, model.instance, bounds=(min_SoC*storage_capacity, max_SoC*storage_capacity))
    model.v_aggregate_disch = Var(model.periods, model.instance, within=Binary)

    ########################## CHARGE VARIABLES ############################
    # Variables at segment level for charge
    model.segment_charge_power_ch = Var(model.segments, model.periods, model.instance, bounds=(0, seg_power))
    model.segment_discharge_power_ch = Var(model.segments, model.periods, model.instance, bounds=(0, seg_power))
    model.SoC_ch = Var(model.segments, model.periods, model.instance, bounds=(SoC_min, SoC_max))

    # Define charge and discharge power variables at aggregate level for charge
    model.charge_power_ch = Var(model.periods, model.instance, bounds=(0, max_power))
    model.discharge_power_ch = Var(model.periods, model.instance, bounds=(0, max_power))
    model.SoC_aggregate_ch = Var(model.periods, model.instance, bounds=(min_SoC*storage_capacity, max_SoC*storage_capacity))
    model.v_aggregate_ch = Var(model.periods, model.instance, within=Binary)

    ################################ MODEL FORMULATION ##############################
    # Objective function
    def objective_rule(model):
        revenue = sum(prob_instance[i-1]*sum(prices[t-1][i-1] * (model.discharge_power_disch[t,i] - model.charge_power_ch[t,i])
                    for t in model.periods)for i in model.instance)  
        cost = sum(prob_instance[i-1]*sum(sum(costs_discharge_segment[s-1] * model.segment_discharge_power_disch[s, t, i] +
                    costs_charge_segment[s-1] * model.segment_charge_power_ch[s, t, i]
                    for s in model.segments) for t in model.periods)for i in model.instance)  
        auxiliary_cost = sum(prob_instance[i-1]*sum(sum(
                    costs_discharge_segment[s-1] * model.segment_discharge_power_ch[s, t, i] +
                    costs_charge_segment[s-1] * model.segment_charge_power_disch[s, t, i] + 
                    costs_discharge_segment[s-1] * model.segment_discharge_power_disch[s, t, i] +
                    costs_charge_segment[s-1] * model.segment_charge_power_ch[s, t, i]
                    for s in model.segments)*(num_periods+1-t)/num_periods 
                    for t in model.periods)for i in model.instance)  
        CVaR_term = CVar_risk_weight*(model.thet_var - (1/(1-sigma_val))*sum(prob_instance[i-1]*model.phi_instance[i] for i in model.instance))
        return revenue - cost - 0.0001 * auxiliary_cost + CVaR_term

    model.profit = Objective(rule=objective_rule, sense=maximize)

    ################################### CVaR Constraints ################################
    def cvar_constriant(model, i):
        revenue = sum(prices[t-1][i-1] * (model.discharge_power_disch[t,i] - model.charge_power_ch[t,i])
                    for t in model.periods) 
        cost = sum(sum(costs_discharge_segment[s-1] * model.segment_discharge_power_disch[s, t, i] +
                    costs_charge_segment[s-1] * model.segment_charge_power_ch[s, t, i]
                    for s in model.segments) for t in model.periods) 
        return model.thet_var - revenue + cost - model.phi_instance[i] <= 0

    model.cvar_constriant = Constraint(model.instance, rule=cvar_constriant)

    ################################### CONSTRAINTS DISCHARGE ################################
    # State of Charge evolution
    def soc_evolution_rule_disch(model, s, t, i):
        if t == 1:
            return model.SoC_disch[s, t, i] == initial_soc_vector_disch[s-1] + model.segment_charge_power_disch[s, t, i]*eta_ch - model.segment_discharge_power_disch[s, t, i]/eta_dis
        else:
            return model.SoC_disch[s, t, i] == model.SoC_disch[s, t-1, i] + model.segment_charge_power_disch[s, t, i]*eta_ch - model.segment_discharge_power_disch[s, t, i]/eta_dis

    model.soc_evolution_constraint_disch = Constraint(model.segments, model.periods, model.instance, rule=soc_evolution_rule_disch)

    # Dispatch power is constrained by storage rate
    def total_power_constraint_disch_1(model, t, i):
        # charge_constraint = (1/num_segments)*storage_capacity*sum(model.segment_charge_power_disch[s, t] for s in model.segments) == model.charge_power_disch[t]
        return (1/num_segments)*storage_capacity * sum(model.segment_charge_power_disch[s, t, i] for s in model.segments) == model.charge_power_disch[t, i]

    def total_power_constraint_disch_2(model, t, i):
        # discharge_constraint = (1/num_segments)*storage_capacity*sum(model.segment_discharge_power_disch[s, t] for s in model.segments) == model.discharge_power_disch[t]
        return (1/num_segments)*storage_capacity * sum(model.segment_discharge_power_disch[s, t, i] for s in model.segments) == model.discharge_power_disch[t, i]

    model.total_power_constraint_disch_1 = Constraint(model.periods, model.instance, rule=total_power_constraint_disch_1)
    model.total_power_constraint_disch_2 = Constraint(model.periods, model.instance, rule=total_power_constraint_disch_2)

    # SoC at the aggregate level is contrained by max Soc
    def soc_aggregate_constraint_disch(model, t, i):
        return ((1/num_segments)*storage_capacity*sum(model.SoC_disch[s, t, i] for s in model.segments) == model.SoC_aggregate_disch[t, i])

    model.soc_aggregate_constraint_disch = Constraint(model.periods, model.instance, rule=soc_aggregate_constraint_disch)

    # Constraint ensuring SoC at the last time period is same as initial SoC
    def final_soc_constraint_rule_disch(model, s, i):
        return model.SoC_aggregate_disch[num_periods, i] == initial_SOC*storage_capacity 
    
    if periodicity_constraint:
        model.final_soc_constraint_disch = Constraint(model.segments, model.instance, rule=final_soc_constraint_rule_disch)

    ######################################### CONSTRAINTS CHARGE ########################################
    # State of Charge evolution
    def soc_evolution_rule_ch(model, s, t, i):
        if t == 1:
            return model.SoC_ch[s, t, i] == initial_soc_vector_ch[s-1] + model.segment_charge_power_ch[s, t, i]*eta_ch - model.segment_discharge_power_ch[s, t, i]/eta_dis
        else:
            return model.SoC_ch[s, t, i] == model.SoC_ch[s, t-1, i] + model.segment_charge_power_ch[s, t, i]*eta_ch - model.segment_discharge_power_ch[s, t, i]/eta_dis

    model.soc_evolution_constraint_ch = Constraint(model.segments, model.periods, model.instance, rule=soc_evolution_rule_ch)

    # Dispatch power is constrained by storage rate
    def total_power_constraint_ch_1(model, t, i):
        return (1/num_segments)*storage_capacity * sum(model.segment_charge_power_ch[s, t, i] for s in model.segments) == model.charge_power_ch[t, i]

    def total_power_constraint_ch_2(model, t, i):
        return (1/num_segments)*storage_capacity * sum(model.segment_discharge_power_ch[s, t, i] for s in model.segments) == model.discharge_power_ch[t, i]

    model.total_power_constraint_ch_1 = Constraint(model.periods, model.instance, rule=total_power_constraint_ch_1)
    model.total_power_constraint_ch_2 = Constraint(model.periods, model.instance, rule=total_power_constraint_ch_2)

    # SoC at the aggregate level is contrained by max Soc
    def soc_aggregate_constraint_ch(model, t, i):
        return ((1/num_segments)*storage_capacity * sum(model.SoC_ch[s, t, i] for s in model.segments) == model.SoC_aggregate_ch[t, i])

    model.soc_aggregate_constraint_ch = Constraint(model.periods, model.instance, rule=soc_aggregate_constraint_ch)

    # Constraint ensuring SoC at the last time period is same as initial SoC
    def final_soc_constraint_rule_ch(model, s, i):
        return model.SoC_aggregate_ch[num_periods, i] == initial_SOC*storage_capacity 
    
    if periodicity_constraint:
        model.final_soc_constraint_ch = Constraint(model.segments, model.instance, rule=final_soc_constraint_rule_ch)

    #################### Link charge part and discharge part ##########################
    def link_charge_power_in_two_parts(model, t, i):
        return model.charge_power_disch[t, i] == model.charge_power_ch[t, i]

    def link_discharge_power_in_two_parts(model, t, i):
        return  model.discharge_power_disch[t, i] == model.discharge_power_ch[t, i]

    model.link_charge_power_in_two_parts = Constraint(model.periods, model.instance, rule=link_charge_power_in_two_parts)
    model.link_discharge_power_in_two_parts = Constraint(model.periods, model.instance, rule=link_discharge_power_in_two_parts)

    ####################  Aggregated binary variables to control simutanously charging and discharging ######################
    def v_aggregate_constraint_ch_1(model, s, t, i):
        return model.segment_charge_power_ch[s, t, i] <= model.v_aggregate_ch[t, i]

    def v_aggregate_constraint_ch_2(model, s, t, i):
        return model.segment_discharge_power_ch[s, t, i] <= (1 - model.v_aggregate_ch[t, i]) 

    def v_aggregate_constraint_disch_1(model, s, t, i):
        return model.segment_charge_power_disch[s, t, i] <= model.v_aggregate_disch[t, i]

    def v_aggregate_constraint_disch_2(model, s, t, i):
        return model.segment_discharge_power_disch[s, t, i] <= (1 - model.v_aggregate_disch[t, i]) 

    model.v_aggregate_constraint_ch_1 = Constraint(model.segments, model.periods, model.instance, rule=v_aggregate_constraint_ch_1)
    model.v_aggregate_constraint_ch_2 = Constraint(model.segments, model.periods, model.instance, rule=v_aggregate_constraint_ch_2)
    model.v_aggregate_constraint_disch_1 = Constraint(model.segments, model.periods, model.instance, rule=v_aggregate_constraint_disch_1)
    model.v_aggregate_constraint_disch_2 = Constraint(model.segments, model.periods, model.instance, rule=v_aggregate_constraint_disch_2)

    ####################################### Solve the model ##############################################
    solver = SolverFactory('gurobi')
    results = solver.solve(model)

    # Check the results
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        # print("Solution is optimal.")
        optimization_status = 1
        # Create empty dictionaries to store variable values
        variables = {
            "Segment": [],
            "Period": [],
            "Instance": [],
            "Discharge_Power_Disch": [],
            "SoC_Disch": [],
            "Charge_Power_Ch": [],
            "SoC_Ch": [],
        }

        # Fill in dictionaries with variable values
        for s in model.segments:
            for t in model.periods:
                for i in range(1, num_instance + 1):
                    variables["Segment"].append(s)
                    variables["Period"].append(t)
                    variables["Instance"].append(i)
                    variables["Discharge_Power_Disch"].append(value(model.segment_discharge_power_disch[s, t, i]))
                    variables["SoC_Disch"].append(value(model.SoC_disch[s, t, i]))
                    variables["Charge_Power_Ch"].append(value(model.segment_charge_power_ch[s, t, i]))
                    variables["SoC_Ch"].append(value(model.SoC_ch[s, t, i]))

        ########################################### Post analysis for offer curve generation ############################################
        prob_matrix = [prob_instance for _ in range(num_periods)]

        values_list_dis = []
        for i in model.discharge_power_disch:
            values_list_dis.append(model.discharge_power_disch[i].value)

        values_list_ch = []
        for i in model.discharge_power_disch:
            values_list_ch.append(model.charge_power_ch[i].value)

        # Creating a numpy array
        dis_values_array = np.array(values_list_dis)
        ch_values_array = np.array(values_list_ch)

        # For example, you can create a DataFrame from values_array
        df_dis = pd.DataFrame(dis_values_array)
        df_ch = pd.DataFrame(ch_values_array)

        df_dis_value = pd.DataFrame(df_dis)
        df_ch_value = pd.DataFrame(df_ch)

        reshaped_dis_value = np.array(df_dis_value).reshape(24, num_scenario+1)
        reshaped_ch_value = np.array(df_ch_value).reshape(24, num_scenario+1)

        df_reshaped_dis_value = pd.DataFrame(reshaped_dis_value)
        df_reshaped_ch_value = pd.DataFrame(reshaped_ch_value)

        assigned_data = []

        # Iterate through each price value and its index
        for i in range(len(prices)):
            for j in range(len(prices[0])):
                price_value = prices[i][j]
                dis_value = df_reshaped_dis_value[j][i]  # Assuming dis_values is aligned with prices
                ch_value = df_reshaped_ch_value[j][i]    # Assuming ch_values is aligned with prices
                weight_fac = prob_matrix[i][j]

                # Find the bin category for the price value
                category = pd.cut([price_value], bins=bin_edges, labels=False).item()
                max_category = pd.cut([np.max(prices)], bins=bin_edges, labels=False).item()
                bin_start = bin_edges[category]
                
                # Correctly access the end of the bin
                if category < len(bin_edges) - 1:
                    bin_end = bin_edges[category + 1]
                else:
                    # Handle the last bin scenario by ensuring bin_end is set correctly
                    bin_end = bin_start + (bin_edges[1] - bin_edges[0])  # Assuming uniform bin widths

                assigned_data.append({
                    'Price': price_value,
                    'DisValue': dis_value,
                    'ChValue': ch_value,
                    'BinStart': bin_start,
                    'BinEnd': bin_end,
                    'weight': weight_fac
                })

        # Convert assigned_data to a DataFrame
        df_assigned_data = pd.DataFrame(assigned_data)
        sorted_aggregated_df = pd.DataFrame()

        # Process chunks of (num_scenario+1) rows at a time
        for chunk_id, start_row in enumerate(range(0, df_assigned_data.shape[0], num_scenario+1), 1):
            # Extracting (num_scenario+1) rows at a time and columns 'DisValue' to 'BinEnd'
            chunk_df = df_assigned_data.iloc[start_row:start_row+num_instance, 1:num_scenario+2]
            
            if (chunk_df['DisValue'] == 0).all() and (chunk_df['ChValue'] == 0).all():
                # Collapse rows into one row with the minimum of BinStart and maximum of BinEnd
                min_bin_start = chunk_df['BinStart'].min()
                max_bin_end = chunk_df['BinEnd'].max()
                collapsed_row = pd.DataFrame({
                    'DisValue': [0],
                    'ChValue': [0],
                    'BinStart': [min_bin_start],
                    'BinEnd': [max_bin_end],
                    'Period': [chunk_id],
                    'weight': [0]
                })
                # Append the collapsed row to the aggregated DataFrame
                sorted_aggregated_df = pd.concat([sorted_aggregated_df, collapsed_row], ignore_index=True)
            else:
                # Sort the chunk
                sorted_chunk = chunk_df.sort_values(by=['BinStart', 'BinEnd'])
                
                # Remove duplicates from the sorted chunk
                unique_sorted_chunk = sorted_chunk.drop_duplicates()
                
                # Append a column with the chunk identifier to the unique sorted chunk
                unique_sorted_chunk['Period'] = chunk_id
                
                # Append the unique and sorted chunk to the aggregated DataFrame
                sorted_aggregated_df = pd.concat([sorted_aggregated_df, unique_sorted_chunk], ignore_index=True)
                
        collapsed_rows = []

        # Initialize variables to track consecutive rows with zero DisValue and zero ChValue
        is_collapse = False
        min_bin_start = None
        max_bin_end = None

        # Iterate over the DataFrame
        for chunk_id in sorted_aggregated_df['Period'].unique():
            # Filter rows with the current Period
            chunk_df = sorted_aggregated_df[sorted_aggregated_df['Period'] == chunk_id]
            
            # Initialize variables to track consecutive rows with zero DisValue and zero ChValue
            is_collapse = False
            min_bin_start = None
            max_bin_end = None
            
            # Iterate over the filtered DataFrame
            for index, row in chunk_df.iterrows():
                # If DisValue and ChValue are both zero
                if row['DisValue'] == 0 and row['ChValue'] == 0:
                    # If it's not already collapsing, start a new collapse
                    if not is_collapse:
                        is_collapse = True
                        min_bin_start = row['BinStart']
                    # Update max_bin_end for consecutive rows
                    max_bin_end = row['BinEnd']
                else:
                    # If already collapsing and found a row with non-zero values,
                    # collapse the consecutive rows and add to the collapsed rows list
                    if is_collapse:
                        collapsed_rows.append({
                            'DisValue': 0,
                            'ChValue': 0,
                            'BinStart': min_bin_start,
                            'BinEnd': max_bin_end,
                            'Period': chunk_id,
                            'weight': 0
                        })
                        is_collapse = False  # Reset collapse flag

                    # Add rows with non-zero values directly to the collapsed rows list
                    collapsed_rows.append(row.to_dict())

            # If the last chunk was being collapsed, add the collapsed row
            if is_collapse:
                collapsed_rows.append({
                    'DisValue': 0,
                    'ChValue': 0,
                    'BinStart': min_bin_start,
                    'BinEnd': max_bin_end,
                    'Period': chunk_id,
                    'weight': weight_fac
                })

        # Convert the list of collapsed rows to a DataFrame
        collapsed_df = pd.DataFrame(collapsed_rows)

        # Function to calculate weighted sum
        def weighted_sum(group, values, weight_col):
            weights = group[weight_col]
            return (group[values] * weights).sum() / weights.sum() if weights.sum() != 0 else 0

        # Add a column for preserving original order
        collapsed_df['original_index'] = collapsed_df.index

        # List to store processed dataframes
        processed_data = []

        # Process each period sequentially
        for _, period_group in collapsed_df.groupby('Period', sort=False):
            collapsed = period_group.groupby(['BinStart', 'BinEnd'], as_index=False).apply(
                lambda g: pd.Series({
                    'DisValue': weighted_sum(g, 'DisValue', 'weight'),
                    'ChValue': weighted_sum(g, 'ChValue', 'weight'),
                    'weight': g['weight'].sum(),  # Summing weights in case needed for further processing
                    'Period': g['Period'].iloc[0],  # Directly take the period value
                    'original_index': g['original_index'].min()  # Take the minimum index to preserve order
                })
            )
            processed_data.append(collapsed)

        # Concatenate all processed data frames and sort by the original index
        final_df = pd.concat(processed_data)
        final_df.sort_values('original_index', inplace=True)
        final_df.drop('original_index', axis=1, inplace=True)  # Drop the helper column

        # Filter DataFrame for DisValue and ChValue separately
        disvalue_df = final_df[final_df['DisValue'] != 0]
        chvalue_df = final_df[final_df['ChValue'] != 0]

        # Remove the weight column from both DataFrames
        disvalue_df = disvalue_df.drop(columns=['weight'])
        chvalue_df = chvalue_df.drop(columns=['weight'])

        # Remove DisValue column from chvalue_df and ChValue column from disvalue_df
        disvalue_df = disvalue_df.drop(columns=['ChValue'])
        chvalue_df = chvalue_df.drop(columns=['DisValue'])

        # check if the charging and discharging offer is still valid after the drop
        modified_disvalue_df = pd.DataFrame()
        modified_chvalue_df = pd.DataFrame()

        if len(disvalue_df) == 0 or np.sum(disvalue_df['DisValue']) <= 0:
            valid_discharge = 0
        else:
            valid_discharge = 1

        if len(chvalue_df) == 0 or np.sum(chvalue_df['ChValue']) <= 0:
            valid_charge = 0
        else:
            valid_charge = 1
            
        ####################################### Final offer curve modification for Discharge offer curve ################################
        if valid_discharge == 1:
            
            # Iterate through each unique period
            for period in disvalue_df['Period'].unique():
                # Get the rows for the current period
                period_df = disvalue_df[disvalue_df['Period'] == period].copy()  # Make a copy to avoid SettingWithCopyWarning
                
                # If there is only one row in the period, append it to the modified DataFrame and continue to the next period
                if len(period_df) == 1:
                    modified_disvalue_df = pd.concat([modified_disvalue_df, period_df])
                    continue
                
                # Initialize a new column to store the modified DisValue
                period_df['Modified_DisValue'] = np.nan
                
                # Fill the DisValue from the first row for each period
                period_df.at[period_df.index[0], 'Modified_DisValue'] = period_df.iloc[0]['DisValue']
                
                # Compare each DisValue with the DisValues of the previous row and the first row
                for i in reversed(range(1, len(period_df))):
                    current_disvalue = period_df.iloc[i]['DisValue']
                    prev_disvalue = period_df.iloc[i-1]['DisValue']
                    first_disvalue = period_df.iloc[0]['DisValue']
                    
                    # If DisValue is greater than or equal to both the previous row and the first row, keep the original DisValue
                    if current_disvalue > prev_disvalue and current_disvalue > first_disvalue:
                        period_df.at[period_df.index[i], 'Modified_DisValue'] = current_disvalue
                    else:
                        period_df = period_df.drop(period_df.index[i])
                    # Reset the index after removing rows
                    period_df = period_df.reset_index(drop=True)
                
                for i in range(1, len(period_df)):
                    modified_disvalue = period_df.iloc[i]['DisValue'] - period_df.iloc[i-1]['DisValue']
                    period_df.at[period_df.index[i], 'DisValue'] = modified_disvalue
                
                # Append the modified period DataFrame to the modified DataFrame
                modified_disvalue_df = pd.concat([modified_disvalue_df, period_df])

            modified_disvalue_df = modified_disvalue_df.drop(columns=['Modified_DisValue'])
            modified_disvalue_df = modified_disvalue_df.drop(columns=['BinStart'])
            modified_disvalue_df = modified_disvalue_df.rename(columns={'BinEnd': 'Price'})

        ####################################### Final offer curve modification for Charge offer curve ###################################
        if valid_charge == 1:
            
            chvalue_df.drop(columns=['BinEnd'], inplace=True)

            # Sort the DataFrame in descending order of "BinStart" for each period
            chvalue_df = chvalue_df.groupby('Period', group_keys=False).apply(lambda x: x.sort_values(by='BinStart', ascending=False))

            # Iterate through each unique period
            for period in chvalue_df['Period'].unique():
                # Get the rows for the current period
                period_df = chvalue_df[chvalue_df['Period'] == period].copy()  # Make a copy to avoid SettingWithCopyWarning
                
                # If there is only one row in the period, append it to the modified DataFrame and continue to the next period
                if len(period_df) == 1:
                    modified_chvalue_df = pd.concat([modified_chvalue_df, period_df])
                    continue
                
                # Initialize a new column to store the modified ChValue
                period_df['Modified_ChValue'] = np.nan
                
                # Assign the last row's ChValue to its Modified_ChValue
                period_df.at[period_df.index[-1], 'Modified_ChValue'] = period_df.iloc[-1]['ChValue']


                # Compare each ChValue with the ChValues of the previous row and the first row
                for i in (range(0, len(period_df)-1)):
                    current_chvalue = period_df.iloc[i]['ChValue']
                    next_chvalue = period_df.iloc[i+1]['ChValue']
                    
                    # If ChValue is greater than or equal to both the previous row and the first row, keep the original ChValue
                    if current_chvalue > next_chvalue :
                        period_df.at[period_df.index[i], 'Modified_ChValue'] = np.nan
                    else :
                        period_df.at[period_df.index[i], 'Modified_ChValue'] = 0

                # Remove rows where Modified_ChValue is NaN
                period_df = period_df.dropna(subset=['Modified_ChValue'])
                
                # Calculate the modified ChValue by subtracting the ChValue of the current row from the ChValue of the previous row
                for i in reversed(range(1, len(period_df))):
                    modified_chvalue = period_df.iloc[i]['ChValue'] - period_df.iloc[i-1]['ChValue']
                    period_df.at[period_df.index[i], 'ChValue'] = modified_chvalue
                
                # Reset index before concatenating
                period_df.reset_index(drop=True, inplace=True)

                # Append the modified period DataFrame to the modified DataFrame
                modified_chvalue_df = pd.concat([modified_chvalue_df, period_df])

            modified_chvalue_df = modified_chvalue_df.drop(columns=['Modified_ChValue'])

    # elif results.solver.termination_condition == TerminationCondition.infeasible:
    #     print("Model is infeasible.")
    else:
        # print("Solver Status:", results.solver.status)
        optimization_status = 0
        valid_discharge = 0
        valid_charge = 0
        modified_chvalue_df = pd.DataFrame()
        modified_disvalue_df = pd.DataFrame()

    return optimization_status, valid_discharge, valid_charge, modified_disvalue_df, modified_chvalue_df
