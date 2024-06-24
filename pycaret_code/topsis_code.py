import numpy as np
import pandas as pd

def topsis(weights, impacts, data):
    # print("")

    # Validate weights and impacts
    if len(weights) != len(impacts):
        raise ValueError("Number of weights and impacts should be the same.")
    if not all(impact in ['+', '-'] for impact in impacts):
        raise ValueError("Impacts must be either '+' or '-' for all criteria.")
    
    # Normalizing the data
    norm_data = data / np.linalg.norm(data, axis=0)
    # print(norm_data)
    # Weighted normalized decision matrix
    weighted_norm_data = norm_data * np.array(weights)
    # print(weighted_norm_data)
    
    # Determine positive and negative ideal solutions based on impacts
    ideal_best = np.max(weighted_norm_data * (np.array(impacts) == '+'), axis=0)
    ideal_worst = np.min(weighted_norm_data * (np.array(impacts) == '-'), axis=0)
    ideal_best_val=[]
    for i in range(len(ideal_best)):
        ideal_best_val.append(ideal_best[i]+ideal_worst[i])
    ideal_best_val=np.transpose(ideal_best_val)
    # print(ideal_best_val)
    ideal_best = np.max(weighted_norm_data * (np.array(impacts) == '-'), axis=0)
    ideal_worst = np.min(weighted_norm_data * (np.array(impacts) == '+'), axis=0)
    ideal_worst_val=[]
    for i in range(len(ideal_best)):
        ideal_worst_val.append(ideal_best[i]+ideal_worst[i])
    ideal_worst_val=np.transpose(ideal_worst_val)
    # print(ideal_worst_val)
    # Calculate the Euclidean distances to positive and negative ideal solutions
    positive_distances = np.sqrt(np.sum((weighted_norm_data - ideal_best_val) ** 2, axis=1))
    negative_distances = np.sqrt(np.sum((weighted_norm_data - ideal_worst_val) ** 2, axis=1))
    
    # Calculate the performance scores
    performance_scores = negative_distances / (positive_distances + negative_distances)
    
    # Determine the rank
    rank = pd.Series(performance_scores, index=data.index).rank(ascending=False)
    # print(data["file_index"])
    

    result= pd.DataFrame({
        "Performance Score": performance_scores,
        "Rank": rank

    })
    
    # print(data['file_index'])
    result=pd.concat([result,data.iloc[:,len(data.columns)-1]],axis=1)
    result=result.sort_values(by='Rank')
    print("result of topsis: ")
    print(result)
    # print(data.iloc[0,2])
    return (result.index[0],result.iloc[0,len(result.columns)-1])
    # return 1

# Example usage:
# weights = [0.25,0.25,0.25,0.25]  # Weights for each criterion
# impacts = ['-','+','+','+']      # Impacts for each criterion ('+' for benefit, '-' for cost)
# data = pd.DataFrame({
#     'price': [250,200,300,275,225],
#     'storage': [16,16,32,32,16],
#     'camera': [12,8,16,8,16],
#     'looks': [5,3,4,4,2],
    
# }, index=['mOBILE1', 'B','c','D','e'])

# result = topsis(weights, impacts, data)
# print(result)
