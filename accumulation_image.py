# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC We want to understand the accumulation risks under below scenario:
# MAGIC ### a. Correlation between model
# MAGIC ### b. Correlation between data
# MAGIC ### c. Correlation between fine-tuned foundation model 

# COMMAND ----------

# MAGIC %sh
# MAGIC git remote add origin https://github.com/YuanyuanLi96/Corr_ML.git

# COMMAND ----------

#!pip uninstall tensorflow
!pip install tensorflow==2.15.0

# COMMAND ----------

!pip install tensorflow_datasets==4.9.3

# COMMAND ----------

import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# COMMAND ----------

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import phik
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50,ResNet101, VGG16,VGG19,DenseNet121,DenseNet169, MobileNet,MobileNetV2
from tensorflow.keras.models import load_model
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize']=(8,6)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
###define models
def get_model(model_name,input_dim=(32, 32, 3),output_dim=10):
    model=None
    base_model=None
    if model_name=="NN1":
        # ANN (Artificial Neural Network)
        model = models.Sequential([
            layers.Flatten(input_shape=input_dim),
            layers.Dense(64, activation='relu'),
            layers.Dense(output_dim, activation='softmax')
        ])

    if model_name=="NN2":
        # ANN (Artificial Neural Network)
        model = models.Sequential([
            layers.Flatten(input_shape=input_dim),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            #layers.Dropout(0.5),
            layers.Dense(output_dim, activation='softmax')
        ])

    if model_name=="CNN1":
        # CNN (Convolutional Neural Network)
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            #layers.Dropout(0.5),
            layers.Dense(output_dim, activation='softmax')
        ])

    if model_name=="CNN2":
        # CNN (Convolutional Neural Network)
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            #layers.Dropout(0.5),
            layers.Dense(output_dim, activation='softmax')
        ])
    
    if model_name=="random_forest":
        model=RandomForestClassifier()
    if model_name=="logistic_regression":
        model=LogisticRegression(max_iter = 10000)
    if model_name=="xgb":
        model=xgb.XGBClassifier()
    if model_name=="lgb":
        model=lgb.LGBMClassifier()

    keras_models=["resnet50","resnet101", "vgg16","vgg19","densenet121","densenet169", "mobilenet","mobilenetV2"]
    if model_name in keras_models:
        if model_name == "densenet121":
            # Load pre-trained DenseNet121 model
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_dim)
        if model_name == "densenet169":
            base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=input_dim)
        if model_name == "mobilenet":
            base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_dim)
        if model_name == "mobilenetV2":
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_dim)
        if model_name=="resnet50":
            # ResNet (Residual Neural Network)
            base_model = ResNet50(weights='imagenet', include_top=False,input_shape=input_dim)
        if model_name=="resnet101":
            # ResNet (Residual Neural Network)
            base_model = ResNet101(weights='imagenet', include_top=False, input_shape=input_dim)
        if model_name=="vgg19":
            # VGG (VGGNet)
            base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_dim)
        if model_name=="vgg16":
            # VGG (VGGNet)
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_dim)

        base_model.trainable = False   
        # Unfreeze some top layers for fine-tuning
        # for layer in base_model.layers[-1:]:
        #     layer.trainable = True
        # Add custom top layers
        global_avg_pooling = layers.GlobalAveragePooling2D()(base_model.output)
        dense_1 = layers.Dense(64, activation='relu')(global_avg_pooling)
        output_layer = layers.Dense(output_dim, activation='softmax')(dense_1)
        
        # Create the full model
        model = models.Model(inputs=base_model.input, outputs=output_layer)
    
    return model



def train_model_nn(X_train, y_train, data_type, model_names,input_dim=(32, 32, 3)):
    folder_path = "parameters_model_full/" + data_type + "/"
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
    #loop over all the models and train them
    
    for model_name in model_names:
        print("train the model: "+ model_name)
        output_dim= len(np.unique(y_train))
        model=get_model(model_name,input_dim=input_dim,output_dim=output_dim)
        model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
        model.fit(X_train,y_train, epochs=10,verbose=0)
        filename_model = folder_path +"param_" + model_name  + data_type+ ".keras" 
        model.save(filename_model)

def train_fit_nn(X_train, y_train, X_test, y_test, model_names,input_dim=(32, 32, 3)):
    #loop over all the models and train them
    test_error={}
    for model_name in model_names:
        print("train the model: "+ model_name)
        model=get_model(model_name,input_dim)
        model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
        model.fit(X_train,y_train, epochs=10,verbose=0)
        # Evaluate the model
        _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        test_error[model_name] = 1 - test_accuracy
    #return average test error
    return test_error

def predict_errors_nn(data_type, model_names, X_test,y_test,version=0,file_name="nn_models_",error_folder="test_loss/"):
    folder_path = "parameters_model_full/" + data_type + "/"
    #loop over all the models and train them
    errors={}
    for model_name in model_names:
        print("run the model: "+ model_name)
        filename_model = folder_path +"param_" +model_name  + data_type+ ".keras" 
        model=load_model(filename_model)
        y_pred =np.argmax(model.predict(X_test), axis=1)
        errors[model_name]= (y_pred != y_test).astype(int)
        #print(y_pred.shape, y_test.flatten().shape,errors[model_name].shape)
    error_df=pd.DataFrame(errors)
    if not os.path.exists(error_folder):
        # Create the folder if it doesn't exist
        os.makedirs(error_folder)
    error_df.to_csv(error_folder+file_name+ data_type + "_"+ str(0)+".csv",index=False)


# COMMAND ----------

from keras.datasets import cifar10, cifar100, mnist,fashion_mnist
from joblib import dump
from joblib import load
import numpy as np
import scipy
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import sys
import csv
import pandas as pd
import tensorflow as tf
import os
from skimage.transform import resize
import tensorflow_datasets as tfds
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(data_type, data_path='data', train_sample_size=2000, test_sample_size=500):
    """
    Preprocess data for various datasets (CIFAR-10, CIFAR-100, MNIST, FASHION, Oxford-IIIT Pets, EuroSAT).

    Parameters:
    - data_type (str): Dataset name ("CIFAR10", "CIFAR100", "MNIST", "FASHION", "PETS", "EUROSAT").
    - data_path (str): Base directory to save the processed datasets
    - train_sample_size(int): number of samples to select from the training set
    - test_sample_size(int): number of samples to select from the test set

    Saves preprocessed datasets as joblib files in the respective data folders.
    """
    folder_path = os.path.join(data_path, f'{data_type}_small')
    os.makedirs(folder_path, exist_ok=True)

    # Load and preprocess datasets
    if data_type == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, y_train = _sample_data(x_train, y_train, train_sample_size)
        x_test, y_test = _sample_data(x_test, y_test, test_sample_size)
    elif data_type == "CIFAR100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train, y_train = _sample_data(x_train, y_train, train_sample_size)
        x_test, y_test = _sample_data(x_test, y_test, test_sample_size)
    elif data_type == "MNIST":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)
        x_train, y_train = _sample_data(x_train, y_train, train_sample_size)
        x_test, y_test = _sample_data(x_test, y_test, test_sample_size)
    elif data_type == "FASHION":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)
        x_train, y_train = _sample_data(x_train, y_train, train_sample_size)
        x_test, y_test = _sample_data(x_test, y_test, test_sample_size)
    elif data_type == "PETS":
        dataset = tfds.load("oxford_iiit_pet",as_supervised=True)
        x_train, y_train = _process_tfds(dataset["train"], train_sample_size)
        x_test, y_test = _process_tfds(dataset["test"], test_sample_size)
    elif data_type == "EUROSAT":
        dataset = tfds.load('eurosat', split='train', as_supervised=True)
        dataset = dataset.shuffle(buffer_size=10000, seed=42)  # Shuffle dataset with a fixed seed
        train_size = int(0.8 * len(dataset))  # Use 80% for training, 20% for testing
        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        x_train, y_train = _process_tfds(train_dataset, train_sample_size)
        x_test, y_test = _process_tfds(test_dataset, test_sample_size)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    # Normalize the data (only once)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert grayscale to RGB if necessary
    if x_train.shape[-1] == 1:  # Convert grayscale to RGB
        x_train = np.repeat(x_train, 3, axis=-1)
        x_test = np.repeat(x_test, 3, axis=-1)

    # Resize to 32x32 if needed
    if x_train.shape[1:3] != (32, 32):  # Resize to 32x32
        x_train = np.array([resize(img, (32, 32)) for img in x_train])
        x_test = np.array([resize(img, (32, 32)) for img in x_test])

    # Flatten labels
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    # Save the data
    try:
        dump(x_train, os.path.join(folder_path, 'X_train.joblib'))
        dump(y_train, os.path.join(folder_path, 'y_train.joblib'))
        dump(x_test, os.path.join(folder_path, 'X_test.joblib'))
        dump(y_test, os.path.join(folder_path, 'y_test.joblib'))
    except Exception as e:
        print(f"Error saving data: {e}")
        return

    print(f"{data_type} data processed and saved to {folder_path}.")


def _sample_data(x, y, sample_size):
    """Helper function to sample data with a fixed seed."""
    np.random.seed(42)
    indices = np.random.choice(len(x), sample_size, replace=False)
    return x[indices], y[indices]


def _process_tfds(dataset, sample_size):
    """Helper function to process tensorflow datasets."""
    images, labels = [], []
    for image, label in dataset.take(sample_size):  # Unpack directly
        image = tf.image.resize(image, (32, 32)).numpy()
        images.append(image)
        labels.append(label.numpy())
    return np.array(images), np.array(labels)


def train_model_sklearn(X_train, y_train, data_type, model_names):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    folder_path = "parameters_model_full/" + data_type + "/"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
    #loop over all the models and train them
    
    for model_name in model_names:
        print("train the model: "+ model_name)
        model=get_model(model_name)
        model.fit(X_train_flat,y_train)
        #np.savetxt(f,[model_names[count]],fmt = "%s")
        #np.savetxt(f,[acc],fmt = "%f")
        filename_model = folder_path +"param_" + model_name  + data_type+".joblib"
        dump(model,filename_model)
        
def predict_errors_sklearn(data_type, model_names,X_test,y_test,version=0,file_name="sklearn_models_",error_folder="test_loss/"):
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    folder_path = "parameters_model_full/" + data_type + "/"
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
    #loop over all the models and train them
    errors={}
    for model_name in model_names:
        print("run the model: "+ model_name)
        filename_model = folder_path +"param_" + model_name  + data_type+".joblib"
        model=load(filename_model)
        y_pred = model.predict(X_test_flat)
        errors[model_name]= (y_pred != y_test).astype(int)
        #print(y_pred.shape, y_test.flatten().shape,errors[model_name].shape)
    error_df=pd.DataFrame(errors)
    if not os.path.exists(error_folder):
        # Create the folder if it doesn't exist
        os.makedirs(error_folder)
    error_df.to_csv(error_folder+file_name + data_type +"_"+ str(version)+".csv",index=False)

def generate_test_loss(data_type, X_test,y_test,model_nn,model_sklearn,version=0):
    predict_errors_nn(data_type, model_nn,X_test, y_test,version)   
    predict_errors_sklearn(data_type, model_sklearn,X_test, y_test,version)

def load_test_loss(data_type, version=0):
    test_error=pd.read_csv( "test_loss/sklearn_models_"+ data_type +"_"+ str(version)+ ".csv")
    test_error_nn=pd.read_csv( "test_loss/nn_models_"+ data_type + "_"+ str(version)+".csv")
    return pd.concat([test_error, test_error_nn], axis=1)

# COMMAND ----------

# list of al the models we would train
model_sklearn=["logistic_regression","random_forest", "xgb"]
model_nn_full=["NN1","NN2","CNN1","CNN2"]
model_nn_tune=["resnet50","resnet101", "vgg16","vgg19","densenet121","densenet169", "mobilenet","mobilenetV2"]
model_fully_trained=model_sklearn+model_nn_full
model_nn=model_nn_full+model_nn_tune
model_names=model_sklearn+model_nn

# COMMAND ----------

data_type ="PETS"
preprocess_data(data_type)
X_train = load('data/'+data_type+'_small/X_train.joblib')
y_train = load('data/'+data_type+'_small/y_train.joblib')
y_test = load('data/'+data_type+'_small/y_test.joblib')
X_test=load('data/'+data_type+'_small/X_test.joblib')

# COMMAND ----------

import matplotlib.pyplot as plt

# Assuming X_train is loaded as a NumPy array
# Load the first image from X_train
image = X_train[5]

# Display the image
plt.imshow(image)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train all models on cifar10

# COMMAND ----------

data_type ="CIFAR10"
#preprocess_data(data_type)
X_train = load('data/'+data_type+'_small/X_train.joblib')
y_train = load('data/'+data_type+'_small/y_train.joblib')
y_test = load('data/'+data_type+'_small/y_test.joblib')
X_test=load('data/'+data_type+'_small/X_test.joblib')

# COMMAND ----------

train_model_sklearn(X_train, y_train, "CIFAR10",model_sklearn)

# COMMAND ----------

train_model_nn(X_train, y_train, "CIFAR10",model_nn_full)

# COMMAND ----------

train_model_nn(X_train, y_train, "CIFAR10",model_nn_tune)

# COMMAND ----------

generate_test_loss("CIFAR10", X_test,y_test,model_nn,model_sklearn)

# COMMAND ----------

df = load_test_loss("CIFAR10")
plt.figure(figsize=(8,6))
df.mean().plot()
plt.xlabel("Model")
plt.ylabel("Error frequency")
#plt.savefig('plots/accuracy_models.pdf')

# COMMAND ----------

df.mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation among fully trained models

# COMMAND ----------

model_fully_trained

# COMMAND ----------

df_selected=df[model_fully_trained]
df_selected.columns=['logistic_regression',
 'random_forest',
 'xgboost',
 'NN1',
 'NN2',
 'CNN1',
 'CNN2']
corr1=df_selected.phik_matrix()
corr1
plt.figure(figsize=(8,6))
sns.heatmap(corr1,cmap="Blues",vmin=0, vmax=1);
plt.xticks(rotation=45, ha='right')
# Save the plot as PNG
plt.savefig('plots/correlation_fully_trained_image.pdf')

# COMMAND ----------

corr1

# COMMAND ----------

significance_overview = df_selected.significance_matrix()
significance_overview

# COMMAND ----------

avg_error_ori=df.mean()
avg_error_ori[model_fully_trained]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation among finetuned model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Finetune on MNIST

# COMMAND ----------

def fintune_pipeline(data_type,model_nn=None, model_sklearn=None):
    preprocess_data(data_type)
    X_train = load('data/'+data_type+'_small/X_train.joblib')
    y_train = load('data/'+data_type+'_small/y_train.joblib')
    y_test = load('data/'+data_type+'_small/y_test.joblib')
    X_test=load('data/'+data_type+'_small/X_test.joblib')
    raw_error=None
    if model_sklearn is not None:
        train_model_sklearn(X_train, y_train, data_type,model_sklearn)
        predict_errors_sklearn(data_type, model_sklearn,X_test, y_test)
        raw_error=pd.read_csv( "test_loss/sklearn_models_"+ data_type + "_"+ str(0)+".csv")
    if model_nn is not None:
        train_model_nn(X_train, y_train, data_type, model_nn)
        predict_errors_nn(data_type, model_nn,X_test, y_test)
        nn_error=pd.read_csv( "test_loss/nn_models_"+ data_type + "_"+ str(0)+".csv")
        raw_error=pd.concat([raw_error,nn_error], axis=1)
    return raw_error

# COMMAND ----------

data_type="MNIST"
raw_error_mnist=fintune_pipeline(data_type,model_nn_tune)

# COMMAND ----------

raw_error_mnist=pd.read_csv( "test_loss/nn_models_"+ "MNIST" + "_"+ str(0)+".csv")

# COMMAND ----------

test_error_cifar= load_test_loss("CIFAR10")
avg_error_cifar10=test_error_cifar.mean()[model_nn_tune]
avg_error_cifar10

# COMMAND ----------

avg_error_mnist=raw_error_mnist.mean()[model_nn_tune]
#correlation of accuracy due to same foundation model--very strong!
scipy.stats.pearsonr(avg_error_mnist.values,avg_error_cifar10.values).statistic

# COMMAND ----------

#correlation of raw errors--very weak！
(scipy.stats.pearsonr(raw_error_mnist["resnet50"],test_error_cifar["resnet50"]).statistic,
scipy.stats.pearsonr(raw_error_mnist["resnet101"],test_error_cifar["resnet101"]).statistic,
scipy.stats.pearsonr(raw_error_mnist["vgg16"],test_error_cifar["vgg16"]).statistic,
scipy.stats.pearsonr(raw_error_mnist["vgg19"],test_error_cifar["vgg19"]).statistic)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Finetune on Fashion dataset

# COMMAND ----------

raw_error_fashion=fintune_pipeline("FASHION",model_nn_tune)

# COMMAND ----------

raw_error_fashion=pd.read_csv( "test_loss/nn_models_"+ "FASHION" + "_"+ str(0)+".csv")

# COMMAND ----------

avg_error_fashion=raw_error_fashion.mean()[model_nn_tune]
scipy.stats.pearsonr(avg_error_fashion.values,avg_error_cifar10.values).statistic

# COMMAND ----------

scipy.stats.pearsonr(avg_error_mnist.values,avg_error_fashion.values).statistic

# COMMAND ----------

# MAGIC %md
# MAGIC ### Finetune on cifar100

# COMMAND ----------

raw_error_cifar100=fintune_pipeline("CIFAR100",model_nn_tune,output_dim=100)

# COMMAND ----------

raw_error_cifar100=pd.read_csv( "test_loss/nn_models_"+ "CIFAR100" + "_"+ str(0)+".csv")

# COMMAND ----------

avg_error_cifar100=raw_error_cifar100.mean()[model_nn_tune]
scipy.stats.pearsonr(avg_error_cifar100.values,avg_error_cifar10.values).statistic

# COMMAND ----------

# MAGIC %md
# MAGIC ### Finetune on EUROSAT

# COMMAND ----------

dataset = tfds.load('eurosat')

# COMMAND ----------

dataset

# COMMAND ----------

data_type="EUROSAT"
raw_error_eurosat=fintune_pipeline(data_type,model_nn_tune)

# COMMAND ----------

raw_error_eurosat=pd.read_csv( "test_loss/nn_models_"+ "EUROSAT" + "_"+ str(0)+".csv")

# COMMAND ----------

avg_error_eurosat=raw_error_eurosat.mean()[model_nn_tune]

# COMMAND ----------

agg_errors=pd.concat([pd.DataFrame(avg_error_cifar10),pd.DataFrame(avg_error_eurosat),pd.DataFrame(avg_error_fashion),pd.DataFrame(avg_error_mnist)], axis=1)

# COMMAND ----------

agg_errors.columns=["CIFAR10","EUROSAT","FASHION","MNIST"]

# COMMAND ----------

plt.rcParams['figure.figsize']=(6,4)
agg_errors.plot.line()
plt.xlabel("Finetuned model")
plt.ylabel("Error frequency")
plt.xticks(rotation=45, ha='right')
#plt.savefig('plots/correlation_performance_funtue.pdf')

# COMMAND ----------

corr1=agg_errors.corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr1,cmap="Blues",vmin=0, vmax=1);
plt.xticks(rotation=45, ha='right')
# Save the plot as PNG
#plt.savefig('plots/correlation_finetuned_data.pdf')
corr1

# COMMAND ----------


