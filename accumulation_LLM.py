# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC We want to understand the accumulation risks under below scenario:
# MAGIC ### a. Correlation between model
# MAGIC ### b. Correlation between data
# MAGIC ### c. Correlation between fine-tuned foundation model 

# COMMAND ----------

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sns
import phik
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize']=(8,6)

# COMMAND ----------


def load_errors(data_type, model_names):
    #loop over all the models and train them
    dt_list=[]
    for model_name in model_names:
        error_df=pd.read_csv( "result/"+ data_type +"/"+ model_name+ "/errors.csv")
        dt_list.append(error_df)
    error_df=pd.concat(dt_list, axis=1)
    error_df.to_csv("result/"+data_type+"/all_models_errors.csv",index=False)
    return error_df

def corr_plot(data_type,model_names):
    error_df= load_errors(data_type, model_names)
    # Extracting the shortened model names by keeping only the string after "/"
    shortened_model_names = [name.split("/")[-1] for name in model_names]
    error_df.columns=shortened_model_names
    error_df.mean().plot()
    plt.xlabel("Model")
    plt.ylabel("Error frequency")
    plt.xticks(rotation=45)
    plt.savefig('result/'+data_type+'/accuracy_models.pdf')
    #correlation of errors
    corr=error_df.phik_matrix()
    #mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(8,6))
    sns.heatmap(corr,cmap="Blues",vmin=0, vmax=1)
    plt.xticks(rotation=45, ha='right')
    plt.savefig('result/'+data_type+'/corr_errors.pdf', bbox_inches='tight')
    return error_df.mean()

# COMMAND ----------

model_names=["mistralai/Mistral-7B-v0.3","meta-llama/Meta-Llama-3-8B", "Qwen/Qwen2-7B", "NousResearch/Llama-2-7b-hf","CohereForAI/aya-23-8B", "tiiuae/falcon-7b","bigscience/bloom-7b1","microsoft/phi-2"]

# COMMAND ----------

data_type ="zeroshot/twitter-financial-news-sentiment"
error1=corr_plot(data_type,model_names)

# COMMAND ----------

data_type ="AdamCodd/emotion-balanced"
error2=corr_plot(data_type,model_names)

# COMMAND ----------

data_type ="fancyzhx/ag_news"
error3=corr_plot(data_type,model_names)

# COMMAND ----------

data_type ="takala/financial_phrasebank"
error4=corr_plot(data_type,model_names)

# COMMAND ----------

data_type ="ma2za/many_emotions"
error5=corr_plot(data_type,model_names)

# COMMAND ----------

df=pd.concat([error4,error1, error2, error5,error3], axis=1)
df.columns=["financial_phrasebank","twitter-financial-news-sentiment","emotion-balanced","many_emotions","ag_news"]
df=df[["financial_phrasebank","twitter-financial-news-sentiment","emotion-balanced","ag_news"]]

# COMMAND ----------

plt.rcParams['figure.figsize']=(6,4)
df.plot.line()
plt.xlabel("Finetuned model")
plt.ylabel("Error frequency")
plt.xticks(rotation=45, ha='right')
plt.savefig('result/performance_funtue_LLM.pdf')

# COMMAND ----------

corr2=np.abs(df.corr())
plt.figure(figsize=(6,5))
plt.rcParams['savefig.bbox'] = 'tight'
sns.heatmap(corr2,cmap="Blues",vmin=0, vmax=1)
plt.xticks(rotation=45, ha='right')
plt.savefig('result/corr_performance_funtue_LLM.pdf')

# COMMAND ----------

# MAGIC %md
# MAGIC ### b. Correlation between data

# COMMAND ----------

from skimage.util import random_noise
def add_noise(var):
    X_test_noisy = np.empty_like(X_test)
    for i, img in enumerate(X_test):
        noisy_img = random_noise(img, mode='gaussian', mean=0, var=var, clip=True)
        X_test_noisy[i] = noisy_img
        dump(X_test_noisy, 'data/'+data_type+'_small/X_test_noisy_'+str(var)+'.joblib')

# COMMAND ----------

# MAGIC %md
# MAGIC ##1. Add Gaussian noise gradually, track model errors change

# COMMAND ----------

#add noise
noises=[0.001,0.01,0.05,0.1]
for no in noises:
    add_noise(no)

# COMMAND ----------

noises=[0.001,0.01,0.05,0.1]
no=noises[0]
X_test_noisy=load('data/'+data_type+'_small/X_test_noisy_'+str(no)+'.joblib')

# COMMAND ----------

y_test=load('data/'+data_type+'_small/y_train.joblib')
y_test[0]

# COMMAND ----------

data_type="CIFAR10"
X_test=load('data/'+data_type+'_small/X_test.joblib')
# Create a figure and axes object
fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # 1 row, 3 columns
# Plot each image on its corresponding axis
axes[0].imshow(X_test[0])
axes[0].set_title('Original image')
axes[0].axis('off')  # Disable axis ticks and labels

noises=[0.001,0.01,0.05,0.1]
counter=1
for no in noises:
    X_test_noisy=load('data/'+data_type+'_small/X_test_noisy_'+str(no)+'.joblib')
    axes[counter].imshow(X_test_noisy[0])
    axes[counter].set_title('noise='+str(no))
    axes[counter].axis('off')
    counter+=1


# Show the plot
plt.tight_layout()  # Adjust layout to prevent overlap of subplots
plt.savefig('plots/img_comparison.pdf')
plt.show()

# COMMAND ----------

data_type="CIFAR10"
X_test=load('data/'+data_type+'_small/X_test.joblib')
plt.figure(figsize=(10,6))
plt.imshow(X_test[0])
plt.savefig('plots/img_original.pdf')

# COMMAND ----------

plt.figure(figsize=(10,6))
plt.imshow(X_test_noisy[0])
plt.savefig('plots/img_noisy.pdf')

# COMMAND ----------

y_test

# COMMAND ----------

# list of al the models we would train
# model_sklearn=["logistic_regression","random_forest", "xgb","lgb"]
# model_nn=["NN1","NN2","CNN1","CNN2"]
# model_names=model_sklearn+model_nn
data_type="CIFAR10"
for no in noises:
    X_test_noisy=load('data/'+data_type+'_small/X_test_noisy_'+str(no)+'.joblib')
    generate_test_loss(data_type, X_test_noisy,y_test,model_nn,model_sklearn,no)

# COMMAND ----------

# MAGIC %md
# MAGIC ## a+1: Correlation between models on drifted cifar10

# COMMAND ----------

no=0.01
df_noise = load_test_loss('CIFAR10',version=no)
corr2=df_noise[model_fully_trained].phik_matrix()
plt.figure(figsize=(10,8))
sns.heatmap(corr2,cmap="Blues",vmin=0, vmax=1)
plt.xticks(rotation=45, ha='right')
# Save the plot as PNG
plt.savefig('plots/correlation_fully_trained'+str(no)+'.pdf')

# COMMAND ----------

avg_error_ori=df.mean()
avg_error_ori[model_fully_trained]

# COMMAND ----------

result=dict()
avg_error_noise={}
for no in noises:
    df_noise = load_test_loss('CIFAR10', version=no)
    corr_same_data=[]
    avg_error_model=[]
    for m in model_fully_trained:
        corr_same_data.append(scipy.stats.pearsonr(df[m],df_noise[m]).statistic)
        avg_error_model.append(df_noise[m].mean())
    result[str(no)]=corr_same_data
    avg_error_noise[str(no)]=avg_error_model

# COMMAND ----------

df_result=pd.DataFrame.from_dict(result)
df_result.index=model_fully_trained
df_result

# COMMAND ----------

df_avg_error=pd.DataFrame.from_dict(avg_error_noise)
df_avg_error.index= model_fully_trained
df_avg_error=pd.concat([pd.DataFrame(avg_error_ori[model_fully_trained]),df_avg_error],axis=1)
df_avg_error#error frequency

# COMMAND ----------

plt.rcParams['figure.figsize']=(10,6)
df_avg_error.T.plot.line()
plt.xlabel("Noise")
plt.ylabel("Error Frequency")
plt.savefig('plots/error_freq_with_noise.pdf')

# COMMAND ----------

# MAGIC %md
# MAGIC ### b+1. Correlation of model errors for noisy data and original data

# COMMAND ----------

plt.rcParams['figure.figsize']=(10,6)
df_result.T.plot.line()
plt.xlabel("Noise")
plt.ylabel("Correlation")
plt.savefig('plots/correlation_with_noise.pdf')

# COMMAND ----------

scipy.stats.pearsonr(result_df.iloc[0,:],result_df.iloc[1,:]).statistic

# COMMAND ----------

# MAGIC %md
# MAGIC ### c. Correlation between fine-tuned foundation model 

# COMMAND ----------


#loop over all the models and train them
dt_list=[]
data_names=[]
for data_type in data_names:
    error_df=pd.read_csv( "result/"+data_type+"/all_models_errors.csv")
    avg_error_df=error_df.mean()
    dt_list.append(avg_error_df)
df=pd.concat(dt_list, axis=1)
agg_errors.columns=data_names
df.to_csv("result/all_data_errors.csv",index=False)
df

# COMMAND ----------

plt.rcParams['figure.figsize']=(10,6)
df.plot.line()
plt.xlabel("Finetuned model")
plt.ylabel("Error frequency")
plt.savefig('plots/correlation_performance_funtue.pdf')

# COMMAND ----------

corr1=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr1,cmap="Blues",vmin=0, vmax=1);
plt.xticks(rotation=45, ha='right')
# Save the plot as PNG
plt.savefig('plots/correlation_finetuned_data.pdf')
corr1

# COMMAND ----------


