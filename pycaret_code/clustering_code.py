import pandas as pd

from pycaret.clustering import *


# def highlight_max(s):
#    is_max = s == s.max()
#    return ['background-color: pink' if v else '' for v in is_max]

# def highlight_min(s):
#    is_min = s == s.min()
#    return ['background-color: white' if v else '' for v in is_min]


def create_report(algo,scores,s,e,verb,new_filename):
  data=pd.read_csv('data_bases/'+str(new_filename))
  e=e+1
  report=pd.DataFrame({},index=scores)
  
  setup(data,verbose=verb)
  for i in range(s,e):
    create_model(algo,num_clusters=i)
    model=pull()
    model=model.transpose().loc[scores,:]
    model.columns=["Clusters="+str(i)]
    report=pd.concat([report,model],axis=1)
  print(report)
  print("okrf")

  setup(data,normalize=True,normalize_method="zscore",verbose=verb)
  for i in range(s,e):
    create_model(algo,num_clusters=i)
    model=pull()
    model=model.transpose().loc[scores,:]
    model.columns=["Clusters="+str(i)+"(normalized)"]
    report=pd.concat([report,model],axis=1)
  # print(report)

  setup(data,transformation=True,transformation_method="yeo-johnson",verbose=verb)
  for i in range(s,e):
    create_model(algo,num_clusters=i)
    model=pull()
    model=model.transpose().loc[scores,:]
    model.columns=["Clusters="+str(i)+"(transformed)"]
    report=pd.concat([report,model],axis=1)
  # print(report)

  setup(data,pca=True,pca_method="linear",verbose=verb)
  for i in range(s,e):
    create_model(algo,num_clusters=i)
    model=pull()
    model=model.transpose().loc[scores,:]
    model.columns=["Clusters="+str(i)+"(pca)"]
    report=pd.concat([report,model],axis=1)

  setup(data,normalize=True,normalize_method="zscore",transformation=True,transformation_method="yeo-johnson",verbose=verb)
  for i in range(s,e):
    create_model(algo,num_clusters=i)
    model=pull()
    model=model.transpose().loc[scores,:]
    model.columns=["Clusters="+str(i)+"(norm+trans)"]
    report=pd.concat([report,model],axis=1)


  setup(data,normalize=True,normalize_method="zscore",transformation=True,transformation_method="yeo-johnson",pca=True,pca_method="linear",verbose=verb)
  for i in range(s,e):
    tuned_model=create_model(algo,num_clusters=i)
    model=pull()
    model=model.transpose().loc[scores,:]
    model.columns=["Clusters="+str(i)+"(norm+trans+pca)"]
    report=pd.concat([report,model],axis=1)
    
  # plot_model(tuned_model,'tsne')
  # plt.savefig("tsne_plot.png")
  # plt.show()
  report=report.transpose()


  # report=report.apply(highlight_max, axis=0)
  # report=report.apply(highlight_min, axis=0)
  # return report
  report.to_csv(r"results/"+str(algo)+r"_report_"+str(new_filename)+r".csv")

# scores=["Silhouette","Calinski-Harabasz","Davies-Bouldin"]
# data=pd.read_csv('data_bases/'+str(sys.argv[1]))
# methods_to_use=["kmeans","hclust","ap","meanshift","dbscan","sc"]
# starting_num_clusters=2
# ending_num_clusters=4
# verb=False
# file="demo"

# for i in methods_to_use:
#   create_report(str(i),data,scores,starting_num_clusters,ending_num_clusters,verb,file)