import pandas as pd
import matplotlib.pyplot as plt
from pycaret.classification import *



def create_reg_report_raw(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable, train_size = train_size, data_split_shuffle = shuffle, data_split_stratify = False, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+str(new_filename))
    



def create_reg_report_normalized(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,normalize=True,normalize_method="zscore",train_size = train_size, data_split_shuffle = shuffle, data_split_stratify = False, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+"Normalized_"+str(new_filename))
    


def create_reg_report_transformation(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,transformation=True,transformation_method="yeo-johnson",train_size = train_size, data_split_shuffle = shuffle, data_split_stratify = False, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+"Transformed_"+str(new_filename))
    


def create_reg_report_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,pca=True,pca_method="linear",train_size = train_size, data_split_shuffle = shuffle, data_split_stratify = False ,verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+"pca_"+str(new_filename))
    


def create_reg_report_normalized_transformation(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,normalize=True,transformation=True,normalize_method="zscore",transformation_method="yeo-johnson",train_size = train_size, data_split_shuffle = shuffle, data_split_stratify = False, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+"norm+transformed_"+str(new_filename))
    


def create_reg_report_normalized_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,normalize=True,pca=True,normalize_method="zscore",pca_method="linear",train_size = train_size, data_split_shuffle = shuffle, data_split_stratify = False, verbose=verb)
    compare_models()
    df=pull()
    df=df.loc[:,scores]
    df.to_csv(r"results\\"+"norm+pca"+str(new_filename))
    


def create_reg_report_transformed_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle):
    setup(data=data, target=target_variable,transformation=True,pca=True,transformation_method="yeo-johnson",pca_method="linear",train_size = train_size, data_split_stratify = False, verbose=verb)
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
    

    create_reg_report_raw(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    print(preprocessing_params)
    if(str(1) in preprocessing_params):
        create_reg_report_normalized(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    if(str(2) in preprocessing_params):
        create_reg_report_transformation(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    if(str(3) in preprocessing_params):
        create_reg_report_pca(data,target_variable,scores,verb,new_filename,train_size,shuffle)
    if(str(4) in preprocessing_params):
        create_reg_report_all(data,target_variable,scores,verb,new_filename,train_size,shuffle)            
    
        
