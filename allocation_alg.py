# Zikai Li

from dataclasses import dataclass
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import pulp


@dataclass
class AllocationResult:
    allocations_detailed: pd.DataFrame  # account, period, step, allocation, request
    allocations_summary: pd.DataFrame   
    weights: List[float]
    total_weighted_allocated: float
    capacity_used: List[float]
    capacities: List[float]
    utilization: List[float]
    step1_allocated: float
    step2_allocated: float


def compute_demand_based_weights(requests):
    """Compute differential pricing weights based on demand patterns.

    The period with lowest total demand is the reference period (weight is 1). 
    Others are scaled based on their demand's ratio to the reference period.
    """
    total_demand_per_period = np.sum(requests, axis=0)
    min_demand = np.min(total_demand_per_period)
    
    if min_demand <= 0:
        raise ValueError("All periods must have positive total demand")
    
    weights = total_demand_per_period / min_demand
    return weights

def allocate_resource(
    account_ids: List[str],
    cur: List[float],
    requests: Union[List[List[float]], np.ndarray],
    capacities: Union[List[float], np.ndarray],
    weights: Optional[Union[List[float], np.ndarray]] = None,
    auto_weights: bool = True,
    two_step: bool = True,
    convergence_threshold: float = 0.01,
    demand_gap_threshold: float = 0.1,
    max_iterations: int = 10
) -> AllocationResult:
    """
    Allocate resources to multiple accounts 
    across multiple periods with differential pricing.

    parameters
    ----------
    account_ids : list of account identifiers
    cur : revenue / performance figures (non-negative)
    requests : 2D array-like of shape (n_accounts, n_periods)
        requests[i][t] = amount requested by account i in period t
    capacities : 1D array-like of length n_periods
        capacities[t] = total capacity available in period t
    weights : 1D array-like of length n_periods, optional parameters:
        weights[t] = pricing weight for period t
        If None and auto_weights=True, computed from demand patterns
    auto_weights : bool, default True
        If True and weights=None, automatically compute weights from demand
    two_step : bool, default True
        If True, perform second step to allocate leftover capacity
    convergence_threshold : float, default 0.01
        Convergence threshold as proportion of capacity (e.g., 0.01 = 1%)
        Step 2 stops when leftover capacity < threshold * capacity for all periods
    demand_gap_threshold : float, default 0.1
        Minimum unmet demand to consider an account for Step 2 allocation
        Accounts with smaller gaps are excluded from Step 2 entitlement constraints
    max_iterations : int, default 10
        Maximum number of iterations for Step 2 allocation to prevent infinite loops
    """
    # convert to np
    requests = np.array(requests)
    capacities = np.array(capacities)
    n_accounts, n_periods = requests.shape

    # some checks on the inputs
    assert len(account_ids) == n_accounts, "Number of accounts must match request rows"
    assert len(cur) == n_accounts, "Number of cur values must match number of accounts"
    assert len(capacities) == n_periods, "Number of capacity values must match request columns"

    # calculate weights if they are not provided
    if weights is None:
        if auto_weights:
            weights = compute_demand_based_weights(requests)
        else:
            weights = np.ones(n_periods)
    else:
        weights = np.array(weights)
        assert len(weights) == n_periods, "Number of weights must match number of periods"

    # currency shares (for entitlements)
    total_cur = sum(cur)
    if total_cur == 0:
        shares = [1.0 / n_accounts] * n_accounts
    else:
        shares = [x / total_cur for x in cur]

    # ========== step 1 allocation ==========
    print("========== step 1 ==========")
    
    # compute weighted total request per account and entitlement caps
    W = [] 
    u_raw = []
    total_weighted_capacity = np.sum(weights * capacities)
    
    for i in range(n_accounts):
        W_i = np.sum(weights * requests[i])
        if W_i > 0:
            cap_weighted = shares[i] * total_weighted_capacity
            u_i = cap_weighted / W_i
        else:
            u_i = 0.0
        W.append(W_i)
        u_raw.append(u_i)

    u_cap = [min(1.0, max(0.0, u)) for u in u_raw]

    # lp for step 1
    prob1 = pulp.LpProblem("step1_allocation", pulp.LpMaximize)
    lambdas = [pulp.LpVariable(f"lambda_{i}", lowBound=0, upBound=u_cap[i]) \
        for i in range(n_accounts)]

    # objective: max total weighted allocation
    prob1 += pulp.lpSum(W[i] * lambdas[i] for i in range(n_accounts))

    # capacity constraints for each period
    for t in range(n_periods):
        prob1 += pulp.lpSum(requests[i, t] * lambdas[i] \
            for i in range(n_accounts)) <= capacities[t], f"Period_{t}_Capacity"
    
    # fairness constraints: accounts with identical entitlements 
    # and requests should get identical lambdas
    for i in range(n_accounts):
        for j in range(i+1, n_accounts):
            # check if accounts i and j are identical 
            # (same entitlement and same requests)
            if (cur[i] == cur[j] and 
                all(abs(requests[i, t] - requests[j, t]) < 1e-6 for t in range(n_periods))):
                prob1 += lambdas[i] == lambdas[j], f"Fairness_{i}_{j}"

    # solve 
    status1 = prob1.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status1] != 'Optimal':
        raise RuntimeError(f"Step 1 LP not optimal: {pulp.LpStatus[status1]}")

    lambda_vals = [l.value() for l in lambdas]
    
    # compute step 1 allocations
    step1_allocations = np.zeros((n_accounts, n_periods))
    for i in range(n_accounts):
        for t in range(n_periods):
            step1_allocations[i, t] = lambda_vals[i] * requests[i, t]

    step1_capacity_used = [np.sum(step1_allocations[:, t]) for t in range(n_periods)]
    step1_weighted_allocated = sum(lambda_vals[i] * W[i] for i in range(n_accounts))
    
    print(f"Step 1 capacity used: {[f'{u:.1f}' for u in step1_capacity_used]}")

    # ========== step 2: allocate leftover capacity ==========
    step2_allocations = np.zeros((n_accounts, n_periods))
    step2_weighted_allocated = 0.0
    
    if two_step:
        print("========== step 2 ==========")

        # init current allocations with step 1 results
        current_allocations = step1_allocations.copy()
        current_capacity_used = step1_capacity_used.copy()
        
        step2_iteration = 0
        
        while step2_iteration < max_iterations:
            step2_iteration += 1
            
            # periods with leftover capacity (over than specified threshold)
            leftover_periods = []
            remaining_capacity = []
            
            for t in range(n_periods):
                remaining = capacities[t] - current_capacity_used[t]
                if remaining > convergence_threshold * capacities[t]:  
                    leftover_periods.append(t)
                    remaining_capacity.append(remaining)
            
            if not leftover_periods:
                print(f"Iteration {step2_iteration}:")
                print(f"Leftover capacity is less than {convergence_threshold:.1%} in all periods. Step 2 completed")
                break
            
            # check if there's excess demand in any leftover period
            # use demand_gap_threshold to ignore very small unmet demands
            has_excess_demand = False
            accounts_with_excess_demand = set()
            
            for t in leftover_periods:
                for i in range(n_accounts):
                    unmet_demand = requests[i, t] - current_allocations[i, t]
                    if unmet_demand > demand_gap_threshold:
                        has_excess_demand = True
                        accounts_with_excess_demand.add(i)
            
            if not has_excess_demand:
                print(f"Iteration {step2_iteration}: \
                    No (nontrivial) excess demand in periods with \
                        leftover capacity. Step 2 completed.")
                break

            print(f"Iteration {step2_iteration}:")
            print(f"Accounts with (nontrivial) excess demand: {list(accounts_with_excess_demand)}")

            # now let's define LP for this iteration
            prob2 = pulp.LpProblem(f"Step2_Iteration{step2_iteration}", pulp.LpMaximize)
            
            # variables: additional allocation b_{it} 
            # for periods with leftovers only
            b_vars = {}
            for i in range(n_accounts):
                for t in leftover_periods:
                    b_vars[(i, t)] = pulp.LpVariable(
                        f"b_{i}_{t}_iter{step2_iteration}", lowBound=0)
            
            # objective is again to maximize total weighted additional allocation
            prob2 += pulp.lpSum(weights[t] * b_vars[(i, t)] 
                               for i in range(n_accounts) for t in leftover_periods)
            
            # constraints
            for t in leftover_periods:
                # capacity constraint: additional allocations cannot exceed remaining capacity
                prob2 += pulp.lpSum(b_vars[(i, t)] for i in range(n_accounts)) <= \
                    remaining_capacity[leftover_periods.index(t)], f"Remaining_Capacity_{t}"
                
                # request constraint: total allocation cannot exceed request
                for i in range(n_accounts):
                    prob2 += current_allocations[i, t] + b_vars[(i, t)] <= \
                        requests[i, t], f"Request_Limit_{i}_{t}"
            
            # fairness constraints again: 
            # identical accounts should get identical additional allocations
            for i in range(n_accounts):
                for j in range(i+1, n_accounts):
                    # check if accounts i and j are identical AND both have excess demand
                    if (i in accounts_with_excess_demand and 
                        j in accounts_with_excess_demand and
                        cur[i] == cur[j] and
                        all(abs(requests[i, t] - requests[j, t]) < 1e-6
                            for t in range(n_periods)) and
                        all(abs(current_allocations[i, t] - current_allocations[j, t]) < 1e-6
                            for t in range(n_periods))):
                        # ensure identical additional allocations for each period
                        for t in leftover_periods:
                            prob2 += b_vars[(i, t)] == b_vars[(j, t)], \
                                f"Step2_Fairness_{i}_{j}_{t}"
            
            # step 2 entitlement constraint: 
            # only allocate to accounts with nontrivial unmet demand
            total_weighted_remaining = sum(weights[t] * remaining_capacity[
                leftover_periods.index(t)] for t in leftover_periods)
            
            if len(accounts_with_excess_demand) <= 1:
                pass
            else:
                # multiple accounts with nontrivial unmet demand: 
                # apply sharing among them proportional to entitlements
                total_share_with_demand = sum(shares[i] for i in accounts_with_excess_demand)
                
                for i in accounts_with_excess_demand:
                    additional_weighted = pulp.lpSum(weights[t] * b_vars[(i, t)] \
                        for t in leftover_periods)
                    # distribute among accounts with excess demand
                    adjusted_share = shares[i] / total_share_with_demand \
                        if total_share_with_demand > 0 else 0
                    step2_entitlement = adjusted_share * total_weighted_remaining
                    prob2 += additional_weighted <= step2_entitlement, f"Step2_Entitlement_{i}"
            

            status2 = prob2.solve(pulp.PULP_CBC_CMD(msg=False))
            if pulp.LpStatus[status2] != 'Optimal':
                print(f"Iteration {step2_iteration}: LP not optimal: \
                    {pulp.LpStatus[status2]}. Stopping Step 2.")
                break
            
            # extract additional allocations from this iteration
            iteration_allocations = np.zeros((n_accounts, n_periods))
            iteration_weighted_allocated = 0.0
            allocation_made = False
            
            for i in range(n_accounts):
                for t in leftover_periods:
                    additional = b_vars[(i, t)].value() or 0.0
                    if additional > 1e-6:
                        iteration_allocations[i, t] = additional
                        step2_allocations[i, t] += additional
                        current_allocations[i, t] += additional
                        allocation_made = True
                        iteration_weighted_allocated += weights[t] * additional
            
            step2_weighted_allocated += iteration_weighted_allocated
            
            # update capacity usage
            for t in range(n_periods):
                current_capacity_used[t] = np.sum(current_allocations[:, t])
                        
            # check if we can call it a day
            if not allocation_made or iteration_weighted_allocated < 1e-6:
                print(f"Iteration {step2_iteration}: \
                    Minimal allocation made. Step 2 completed.")
                break
        
        if step2_iteration >= max_iterations:
            print(f"Step 2 reached maximum iterations ({max_iterations}).")
        
        print(f"Step 2 completed after {step2_iteration} iterations.")
    
    else:
        print("Two-step allocation disabled. Using Step 1 results only.")

    # ========== combine results ==========
    if two_step:
        final_allocations = current_allocations
        final_capacity_used = current_capacity_used
    else:
        final_allocations = step1_allocations
        final_capacity_used = step1_capacity_used
    
    utilization = [final_capacity_used[t] / capacities[t] 
                   if capacities[t] > 0 else 0 for t in range(n_periods)]
    total_weighted_allocated = step1_weighted_allocated + step2_weighted_allocated

    # create df to store results
    detailed_data = []
    for i in range(n_accounts):
        for t in range(n_periods):
            # step 1 allocation
            detailed_data.append({
                'account_id': account_ids[i],
                'period': t + 1,
                'step': 1,
                'allocation': step1_allocations[i, t],
                'request': requests[i, t]
            })
            
            # step 2 allocation 
            # combined across all iterations
            if two_step and step2_allocations[i, t] > 1e-6:
                detailed_data.append({
                    'account_id': account_ids[i],
                    'period': t + 1,
                    'step': 2,
                    'allocation': step2_allocations[i, t],
                    'request': requests[i, t]
                })
    
    allocations_detailed = pd.DataFrame(detailed_data)

    # cerate summary df
    summary_data = {
        'account_id': account_ids,
        'cur': cur,
        'entitlement_share': shares,
        'lambda': lambda_vals,
        'u_cap': u_cap,
        'weighted_request': W,
        'weighted_allocated': [sum(weights[t] * final_allocations[i, t] 
                                   for t in range(n_periods)) 
                               for i in range(n_accounts)],
    }
    
    # add period-specific columns
    for t in range(n_periods):
        summary_data[f'req_period_{t+1}'] = requests[:, t]
        summary_data[f'alloc_period_{t+1}'] = final_allocations[:, t]
        summary_data[f'step1_alloc_period_{t+1}'] = step1_allocations[:, t]
        summary_data[f'step2_alloc_period_{t+1}'] = step2_allocations[:, t]
    
    allocations_summary = pd.DataFrame(summary_data)

    return AllocationResult(
        allocations_detailed=allocations_detailed,
        allocations_summary=allocations_summary,
        weights=weights.tolist(),
        total_weighted_allocated=total_weighted_allocated,
        capacity_used=final_capacity_used,
        capacities=capacities.tolist(),
        utilization=utilization,
        step1_allocated=step1_weighted_allocated,
        step2_allocated=step2_weighted_allocated,
    )