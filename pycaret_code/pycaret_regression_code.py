import pandas as pd
import matplotlib.pyplot as plt
from pycaret.regression import *

def tune_the_model(model_name,data,target_col,preprocessing_pipeline):
    df=pd.read_csv("data_bases\\"+data)
    print(preprocessing_pipeline)
    setup(data=df,target=target_col,preprocess=preprocessing_pipeline)
    selected_model=create_model(model_name,fold=5)
    tuned_model=tune_model(selected_model,n_iter=20,choose_better = True)
    return (tuned_model.get_params())

def create_reg_report_raw(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable, train_size = train_size, data_split_shuffle = shuffle, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    
    df.to_csv(r"results\\"+str(new_filename))
    



def create_reg_report_normalized(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,normalize=True,normalize_method="zscore",train_size = train_size, data_split_shuffle = shuffle, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+"Normalized_"+str(new_filename))
    


def create_reg_report_transformation(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,transformation=True,transformation_method="yeo-johnson",train_size = train_size, data_split_shuffle = shuffle, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+"Transformed_"+str(new_filename))
    


def create_reg_report_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    
    setup(data=data, target=target_variable,pca=True,pca_method="linear",train_size = train_size, data_split_shuffle = shuffle, verbose=verb)
    
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    print("saved")
    df.to_csv(r"results\\"+"pca_"+str(new_filename))
    


def create_reg_report_normalized_transformation(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,normalize=True,transformation=True,normalize_method="zscore",transformation_method="yeo-johnson",train_size = train_size, data_split_shuffle = shuffle, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+"norm+transformed_"+str(new_filename))
    


def create_reg_report_normalized_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,normalize=True,pca=True,normalize_method="zscore",pca_method="linear",train_size = train_size, data_split_shuffle = shuffle, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+"norm+pca"+str(new_filename))
    


def create_reg_report_transformed_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,transformation=True,pca=True,transformation_method="yeo-johnson",pca_method="linear",train_size = train_size, data_split_shuffle = shuffle, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+"transformed+pca"+str(new_filename))
    


def create_reg_report_all(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    create_reg_report_raw(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    create_reg_report_normalized(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    create_reg_report_transformation(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    create_reg_report_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    create_reg_report_normalized_transformation(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    create_reg_report_normalized_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    create_reg_report_transformed_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle)


def create_report(preprocessing_params,target_variable,scores,verb,new_filename):
    data=pd.read_csv('data_bases/'+str(new_filename))
    print(data)
    report=pd.DataFrame({},index=scores)
    train_size=0.7
    shuffle=False
    print(preprocessing_params)
    

    create_reg_report_raw(data,target_variable,scores,verb,new_filename,train_size,shuffle)

    if(str(1) in preprocessing_params):
        create_reg_report_normalized(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    if(str(2) in preprocessing_params):
        create_reg_report_transformation(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    if(str(3) in preprocessing_params):
        create_reg_report_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    if(str(4) in preprocessing_params):
        create_reg_report_all(data,target_variable,scores,verb,new_filename,train_size,shuffle)            
    
        
