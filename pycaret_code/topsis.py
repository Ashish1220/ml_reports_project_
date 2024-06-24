import os
import numpy as np
import pandas as pd
from pycaret_code.topsis_code import topsis
def apply_topsis(weights,impacts,ncols):
    temp=[]
    for i in weights:
         temp.append(int(i))
    weights=temp 
    files_in_directory = os.listdir("results")
    l=list()
    final_topsis=pd.DataFrame([])
    file=1
    
    weights.append(int(0))
    impacts.append('+')
    for i in files_in_directory:            
            df=pd.read_csv("results\\"+str(i))
            to_add=df.iloc[:,-ncols:]
            # print([file]*len(to_add))
            to_add["file_index"]= [file]*len(to_add)
            file=file+1
            # print(to_add)
            index=topsis(weights,impacts,to_add)
            # index=1
            # print(index)
            # print(to_add)
            final_topsis=pd.concat([final_topsis,to_add.iloc[index[0],:]],axis=1)

    # print("Final topsis:")
    # print(final_topsis.transpose())
    final_index=topsis(weights,impacts,final_topsis.transpose())

    print(final_index)
    # print(final_index[1])
    # temp=-1
    
  
     

    # def search_series_in_dataframe(series, dataframe):
    
    #     indices = []
    #     for index, row in dataframe.iterrows():
    #         if series.equals(row):
    #             indices.append(index)
    #     return indices

 
    # table_index=1
    # print(type(topsis_final_value))

    t=1
    
    # print(final_index[1])
    for i in files_in_directory:
        
        if(t==final_index[1]):
            print("matched!!: "+str(i))
            df=pd.read_csv("results\\"+str(i))
            return [df.columns,df.iloc[final_index[0],:],i]
        t=t+1
            # df=df.iloc[:,-ncols:]
    #         ind=search_series_in_dataframe(topsis_final_value,df)
    #         if(not len(ind)==0):
    #             return {"file_name":i,"index":ind}            
    #         table_index=table_index+1 

