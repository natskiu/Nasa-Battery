def remaining_cycles(dataset=new_cycles, indices=discharge_indices,threshold = 0.7):
    '''
    input
    ---------
    threshold: int, the battery is considered to have reached its end-of-life if the capacity is lower than initial_capacity*threshold
    output
    --------
    remaining_cycles_list: list, a list that 
    '''
    #capacity_list = []
    initial_capacity = (dataset[0,1][3][0,0][6]).flatten().tolist()[0]
    cutoff = initial_capacity*threshold
    capacity = initial_capacity
    i = 0
    while capacity > cutoff:
        critical_cycle = i-1
        capacity = (dataset[0,indices[i]][3][0,0][6]).flatten().tolist()[0]
        i += 1

    remaining_cycles_list = [(critical_cycle - j) for j in range(len(indices))]
    return critical_cycle, (dataset[0,indices[critical_cycle]][3][0,0][6]).flatten().tolist()[0], remaining_cycles_list