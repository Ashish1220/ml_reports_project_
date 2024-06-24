import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_mail import Mail, Message
from pycaret_code.clustering_code import create_report as cr_clustering
from pycaret_code.pycaret_regression_code import create_report as cr_regression
from pycaret_code.pycaret_regression_code import tune_the_model as tune_regression_model
from pycaret_code.pycaret_visualisation_code import create_3d_plot
from pycaret_code.pycaret_classification_code import create_report as cr_classification
from pycaret_code.topsis import apply_topsis
import pandas as pd


ALLOWED_EXTENSIONS = set(['csv'])




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['SECRET_KEY'] = "tsfyguaistyatuis589566875623568956"
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'mlreports32700@gmail.com'
app.config['MAIL_PASSWORD'] = 'repd vkcu tvxc istb' #repd vkcu tvxc istb
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

@app.route('/',methods=['GET'])

def main_page():
    if request.method=="GET":
        return render_template("index.html")

@app.route('/clustering',methods=['GET', 'POST'])
def get_clustering_reports():
    if request.method == 'POST':
        min_clusters=request.form["min_clusters"]
        max_clusters=request.form["max_clusters"]
        methods_to_use=request.form.getlist('checkbox')
        scores=request.form.getlist('checkbox2')
        file = request.files['data']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename).replace(".csv","")
            new_filename = (filename+str(str(datetime.now()).replace(" ","").replace(".","").replace("-","").replace(":","")))+".csv"
            save_location = os.path.join(r'c:\Users\ASUS\Desktop\pycaret_assignment\data_bases',new_filename)
            file.save(save_location)
                        
            verb=False
            # for i in methods_to_use:
                # cr_clustering(str(i),scores,int(min_clusters),int(max_clusters),verb,new_filename)

            files_in_directory = os.listdir("results")
            
            df_list=list()
            head_title=list()
            ncol=-1
            for i in files_in_directory:
                df_list=[df_list,((pd.read_csv("results\\"+str(i))).to_html(classes='data'))]
                ncol=len(scores)    
                head_title=[head_title,str(i)]
            return  render_template("sent_classification.html",tables=df_list, titles=head_title,zip=zip,ncol=ncol-1,data=new_filename,target_col=None)
  
        else:
            return("did not get any file")
    else:
        return render_template('index_clustering.html')

@app.route('/regression',methods=['GET', 'POST'])
def get_regression_reports():
    if request.method == 'POST':
        files_in_directory = os.listdir("results")
        for i in files_in_directory:
            os.remove("results\\"+str(i))
        
        file = request.files['data']
        target_variable=request.form["target_variable"]
        scores=request.form.getlist('checkbox1')
        preprocessing_params=request.form.getlist('checkbox2')
        verb=False
        scores.insert(0,"Model")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename).replace(".csv","")
            new_filename = (filename+str(str(datetime.now()).replace(" ","").replace(".","").replace("-","").replace(":","")))+".csv"
            save_location = os.path.join(r'c:\Users\ASUS\Desktop\pycaret_assignment\data_bases',new_filename)
            file.save(save_location)
            
            cr_regression(preprocessing_params,target_variable,scores,verb,new_filename)
            
            
            
            files_in_directory = os.listdir("results")
            
            df_list=list()
            head_title=list()
            ncol=-1
            
            for i in files_in_directory:
                df_list=[df_list,((pd.read_csv("results\\"+str(i))).to_html(classes='data'))]
                ncol=len(scores)
                colunms=pd.read_csv("results\\"+str(i)).columns    
                head_title=[head_title,str(i)]
            return  render_template("sent.html",tables=df_list, titles=head_title,zip=zip,ncol=ncol-1,data=new_filename,target_col=target_variable,colunms=colunms)
  

            
            return("uploaded! Wait you will recieve your email soon!")
        else:
            return("did not get any file")

    else:
        return render_template('index_regression.html')
    
@app.route('/visualization',methods=['GET','POST'])
def get_visualisation():
    if request.method=="POST":
        client_mail_id=request.form["email"]
        file = request.files['data']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename).replace(".csv","")
            new_filename = (filename+str(str(datetime.now()).replace(" ","").replace(".","").replace("-","").replace(":","")))+".csv"
            save_location = os.path.join(r'c:\Users\ASUS\Desktop\pycaret_assignment\data_bases',new_filename)
            file.save(save_location)
            msg = Message(subject='REGRESSION REPORTS!', sender='noreply-mlreports32700@gmail.com', recipients=[client_mail_id])
            msg.body = "HERE ARE YOUR REPORTS FOR THE DATA-SET:"+(str(filename)).capitalize()+" !!"
            
            # create_3d_plot(new_filename)
            
            files_in_directory = os.listdir("results")
            
            for i in files_in_directory:
                with app.open_resource("results\\"+str(i)) as fp:  
                    msg.attach(str(i), "application/csv", fp.read())
                     
            for i in files_in_directory:
                os.remove("results\\"+str(i)) 
            mail.send(msg)
            return("uploaded! Wait you will recieve your email soon!")
        else:
            return("did not get any file")

    else:
        return ("Working on this!!")
    
@app.route('/classification',methods=['GET','POST'])
def get_classification_reports():
    if request.method=="POST":
        files_in_directory = os.listdir("results")
        for i in files_in_directory:
            os.remove("results\\"+str(i))
        file = request.files['data']
        target_variable=request.form["target_variable"]
        scores=request.form.getlist('checkbox1')
        preprocessing_params=request.form.getlist('checkbox2')
        verb=False
        scores.insert(0,"Model")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename).replace(".csv","")
            new_filename = (filename+str(str(datetime.now()).replace(" ","").replace(".","").replace("-","").replace(":","")))+".csv"
            save_location = os.path.join(r'c:\Users\ASUS\Desktop\pycaret_assignment\data_bases',new_filename)
            file.save(save_location)
            
            cr_classification(preprocessing_params,target_variable,scores,verb,new_filename)
            files_in_directory = os.listdir("results")
            
            df_list=list()
            head_title=list()
            ncol=-1
            for i in files_in_directory:
                df_list=[df_list,((pd.read_csv("results\\"+str(i))).to_html(classes='data'))]
                ncol=len(scores)
                colunms=pd.read_csv("results\\"+str(i)).columns    
                head_title=[head_title,str(i)]
            return  render_template("sent.html",tables=df_list, titles=head_title,zip=zip,ncol=ncol-1,data=new_filename,target_col=target_variable,colunms=colunms)           
           
        else:
            return("did not get any file")

    else:
        return render_template("index_classification.html")

@app.route('/sent',methods=['GET','POST'])
def get_topsis_results():
   
    if request.method=="POST":
        weights=request.form.getlist("numberInput")
        impacts=request.form.getlist("operatorInput")
        data=request.form["data"]
        target_col=request.form["target_col"]
        topsis_results=apply_topsis(weights,impacts,len(weights))
        return render_template("topsis_result.html",topsis_results=topsis_results,data=data,target_col=target_col,len=len,range=range)
    else:
        return render_template("sent.html")

@app.route('/trained_model',methods=['GET','POST'])
def get_trained_model():
    
    # files_in_directory = os.listdir("results")
    # for i in files_in_directory:
    #     os.remove("results\\"+str(i)) 

    if request.method=="POST":
        selected_model=request.form["selected_model_name"]
        selected__full_model=request.form["selected_model_full_name"]
        data=request.form["data"]
        target_col=request.form["target_col"]
        preprocessing_string=request.form["preprocessing_technique"]
        checks_for_preprocessing=[0,0,0,0]
        if("norm" in preprocessing_string):
            checks_for_preprocessing[1]=1
        if("transformed" in preprocessing_string):
            checks_for_preprocessing[2]=1
        if("pca" in preprocessing_string):
            checks_for_preprocessing[3]=1
        if(checks_for_preprocessing[1]==1 or checks_for_preprocessing[2]==1 or checks_for_preprocessing[3]==1):
            checks_for_preprocessing[0]=0
        else:
            checks_for_preprocessing[0]=1

        preprocessing_pipeline=list()

        if(checks_for_preprocessing[0]==0):
            if(checks_for_preprocessing[1]==1):
                preprocessing_pipeline.append("normalize")
            if(checks_for_preprocessing[2]==1):
                preprocessing_pipeline.append("scale")
            if(checks_for_preprocessing[3]==1):
                preprocessing_pipeline.append("pca")




        best_hyperparameters=tune_regression_model(selected_model,data,target_col,preprocessing_pipeline)
        return render_template("trained_model.html",selected_model=selected_model,data=data,target_col=target_col,best_hyperparameters=best_hyperparameters,selected_full_model=selected__full_model)
    

        

    else:
        return "404"
    
@app.route('/get_email',methods=['POST'])
def get_email():
    if request.method=='POST':
                client_mail_id=request.form["email"]
                filename = request.form['data']
                msg = Message(subject='ML REPORTS!', sender='noreply-mlreports32700@gmail.com', recipients=[client_mail_id])
                msg.body = "HERE ARE YOUR REPORTS FOR THE DATA-SET:"+(str(filename)).capitalize()+" !!"
                directory = 'results'
                files_in_directory = os.listdir(directory)                

                for i in files_in_directory:   
                    with app.open_resource("results\\"+str(i)) as fp:  
                        msg.attach(str(i)[0:-24], "application/csv", fp.read())
                mail.send(msg)
                return "mail sent"
    else: 
        return "sdf"



if __name__ == '__main__':
    
    app.run(debug=True)
    
