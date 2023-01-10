from sklearn.datasets import fetch_openml
from fastapi.testclient import TestClient
from main import app
from customer.payment_utils import Client
from json import loads, dumps
from numpy import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from io import BytesIO
from time import sleep
import requests

test_app = TestClient(app)
_client_name = "bob"
_port = "8082"

def state_init():
    st.session_state.session_id=''
    st.session_state.payment_preimage=''
    st.session_state.itercount=0 # Keep track of the number of compute iters
    st.session_state.label_itercount=0 # Keep track of the number of compute iters
    st.session_state.train_itercount=0 # Keep track of the number of compute iters
    st.session_state.accuracies = [] # List for tracking accuracy versus iterations
    st.session_state.chosen_indices = []
    st.session_state.data_labels=[]
    st.session_state.X = pd.DataFrame()
    st.session_state.current_index=42
    st.session_state.label_response_dict = {}
    st.session_state.train_response_dict = {}
    st.session_state.rows = 28
    st.session_state.columns = 28
    st.session_state.raw_df = pd.DataFrame()
    st.session_state.features_df = pd.DataFrame()
    st.session_state.validation_df = pd.DataFrame()
    st.session_state.label_checkpoint_df = pd.DataFrame()
    st.session_state.raw_display_format = "Image"
    st.session_state.label_decision = "label"
    st.session_state.label_uncertainty = 1.0
    st.session_state.current_label=0
    st.session_state.label_select_decision = "Label"
    st.session_state.label_decision_index=0
    st.session_state.label_uncertainties = []



def session_valid(session_id:str, payment_preimage:str)->bool:
    """ Check for session validity """
    #response = test_app.post("/session_info/"+session_id+"/"+payment_preimage)
    response = requests.get(url='http://localhost:8000/session_info/'+session_id+'/'+payment_preimage)
    if response.status_code==200:
        return response.json()["valid_session"]
    return False

#@st.cache
def display_title():
    """ title related formalities """

    st.set_page_config(layout='wide')
    st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
    st.title('ALsats - Active Learning (for a few) sats')

def sidebar_utils():
    with st.sidebar:
        st.session_state.session_id = st.text_input("Session ID",st.session_state.session_id)
        st.session_state.payment_preimage = st.text_input("Payment Hash",st.session_state.payment_preimage)
        if session_valid(st.session_state.session_id,st.session_state.payment_preimage):
            features_dataset = st.file_uploader("Features dataset")
            if features_dataset:
                st.session_state.features_df = pd.read_csv(features_dataset)
            raw_dataset = st.file_uploader("Raw dataset")
            if raw_dataset:
                st.session_state.raw_df = pd.read_csv(raw_dataset)
            st.session_state.raw_display_format = st.selectbox("Raw data display format",options=["Image","DataFrame"])
            validation_dataset = st.file_uploader("Validation dataset - features + labels")
            if validation_dataset:
                st.session_state.validation_df = pd.read_csv(validation_dataset)
            label_checkpoint_dataset = st.file_uploader("Checkpointed label dataset - features + labels")
            if label_checkpoint_dataset:
                st.session_state.label_checkpoint_df = pd.read_csv(label_checkpoint_dataset)
            st.session_state.rows = int(st.number_input("Number of rows in feature image",value=28))
            st.session_state.cols = int(st.number_input("Number of cols in feature image",value=28))

def label(session_id:str,payment_preimage:str,x_label:list):
    """ Sends a label decision (yes/no) request to alsats server """
    
    label_response_dict={}
    data = dumps({"algorithm":"rf","x_label":x_label})
    headers = {"preimage":payment_preimage}
    response = test_app.post("/label/"+session_id,headers=headers,data=data)
    if response.status_code==200:
        label_response_dict = response.json()

    st.session_state.label_response_dict = label_response_dict

def train(session_id:str,payment_preimage:str,x_train:list,y_train:list):
    """ Sends a train request to alsats server """
    train_response_dict={}
    data = dumps({"algorithm":"rf","x_train":x_train,"y_train":y_train})
    headers = {"preimage":payment_preimage}
    response = test_app.post("/train/"+session_id,headers=headers,data=data)
    if response.status_code==200:
        train_response_dict = response.json()
    st.session_state.train_response_dict = train_response_dict

def calc_accuracy(chosen_indices:list,data_labels:list):
    """ 
    Calculate accuracy on labels labeled so far
    """
    print(f'Selected label = {st.session_state.current_label}')
    if len(chosen_indices)<=1 or len(data_labels)<=1:
        return 0.0
    print(f'Len chosen indices {len(chosen_indices)}')
    print(f'len data labels {len(data_labels)}')
    gbc = GradientBoostingClassifier()
    X = st.session_state.X[chosen_indices,:]
    y = np.array(data_labels)
    gbc.fit(X,y)
    validation_matrix = st.session_state.validation_df.to_numpy()
    y_refs = validation_matrix[:,-1]
    y_preds = gbc.predict(validation_matrix[:,:-1])
    return accuracy_score(y_refs, y_preds)

def label_decision_change():
    pass


def label_change():
    """ Actions taken When the user selects a label """
    st.session_state.data_labels.append(st.session_state.current_label)
    st.session_state.chosen_indices.append(st.session_state.current_index)
    st.session_state.accuracies.append(calc_accuracy(st.session_state.chosen_indices,st.session_state.data_labels))
    x_train = st.session_state.X[st.session_state.current_index].astype(float).tolist()
    train(st.session_state.session_id,st.session_state.payment_preimage,[x_train],[st.session_state.current_label])


def convert_labels_to_df():
    """ Convert existing labels and features to dataframe and then csv to make it ready for download """
    label_df = pd.DataFrame(st.session_state.X[st.session_state.chosen_indices,:])
    label_df['label'] = st.session_state.data_labels
    return label_df.to_csv(index=False).encode('utf-8')

def download_model(session_id:str,preimage:str):
    """ Save and download a valid model """
    download_result={}
    try:
        response = test_app.get("/download/"+session_id+"/"+preimage)
        download_result = response.json()
    except Exception as e:
        print("Unable to download file.")
        return None

    if "download_payload" in download_result.keys():
        # Read contents of file into stream as bytes
        # return bytes
        url = download_result["download_payload"]
        print(url)
        response = requests.get(url, stream=True)
        if response.status_code==200:
            contents = response.content
            return contents
        else:
            st.write("File not available to download")
    else:
        st.write("File not available to download")
    return None

def get_current_index()->int:
    """ Sets current index, checks if the X values are already present in the checkpoint dataset """

    # Check if label checkpoint df has values else set random index and return
    if len(st.session_state.label_checkpoint_df)>0:
        X_checkpoint = st.session_state.label_checkpoint_df.to_numpy()
        X_checkpoint = X_checkpoint[:,:-1]
    else:
        return random.randint(0,st.session_state.X.shape[0])
        
    checked_indices=[]
    iterations=0
    # Check only up to a max number of points
    while iterations<=len(st.session_state.X):

        iterations+=1
        #first select randomly
        current_index = random.randint(0,st.session_state.X.shape[0])

        #check if selection has already been tested
        if current_index not in checked_indices:
            checked_indices.append(current_index)
        # if selection has already been tested don't bother going through because its in the labeled dataset, so start from top
        else:
            continue

        # Now check if the randomly selected value exists in the checkpoint matrix
        candidate_x = st.session_state.X[current_index]
        # find all indices in the checkpoint dataframe that correspond to the candidate X value
        idx_list = np.where((X_checkpoint==candidate_x).all(axis=1))[0]
        # if match exists ==> point has already been labeled ==> len of list > 0 so start from top
        if len(idx_list)>0:
            continue
        return current_index
    
    return random.randint(0,st.session_state.X.shape[0])


def display_training_data():
    """ Based on user data selection, display as image or x-y graph on the left of display"""
    img_cont = st.empty()
    # Display raw data as image
    st.session_state.X = st.session_state.features_df.to_numpy()
    st.session_state.current_index = get_current_index()
    if st.session_state.raw_display_format == "Image":
        img = st.session_state.X[st.session_state.current_index].reshape((st.session_state.rows,st.session_state.cols))
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)
        plt.savefig('./image.jpg',dpi=50,format='jpg')
        img_cont.image('./image.jpg',width=7,caption='Input image',clamp=True,use_column_width='auto')

def display_label_train_options():
    # When a new image is displayed, under the hood a label decision request is sent, the result
    # is displayed along with a selection box
    # The value selected in the selection box is set in session state as the last labeled value.
    # When the selection is done, this example is sent to train via a training request.
    # A new id is then chosen at random from the dataset and the process continues.
    x_label = st.session_state.X[st.session_state.current_index].astype(float).tolist()
    label(st.session_state.session_id,st.session_state.payment_preimage,[x_label])  
    if bool(st.session_state.label_response_dict):
        st.session_state.label_decision = st.session_state.label_response_dict["decision"]
        st.session_state.label_uncertainty = st.session_state.label_response_dict["uncertainty"][0]
        if 'label_uncertainties' not in st.session_state:
            st.session_state.label_uncertainties=[]
            st.session_state.label_uncertainties.append(st.session_state.label_uncertainty)
        st.session_state.label_itercount+=1
    else:
        st.session_state.label_decision = "label"
        st.session_state.label_uncertainty = 1.0
        st.session_state.label_uncertainties.append(st.session_state.label_uncertainty)

    st.metric("Active Learner Decision","Label"if st.session_state.label_decision=="label" else "Do Not Label")
    st.session_state.label_select_decision = st.selectbox("Label?",options=["Label","Skip"],index=0)
    if st.session_state.label_select_decision == "Label": 
        st.selectbox("Label data point",options=list(range(0,10)),on_change=label_change,key='current_label')
        if bool(st.session_state.train_response_dict):
            st.session_state.train_itercount+=1


def display_test_accuracy():
    """ Display test set accuracy per iteration """
    plot_cont = st.empty()
    fig,axes = plt.subplots(figsize=(3,2))
    iters = list( range( 0,st.session_state.label_itercount ) )
    min_iters = min(len(iters),len(st.session_state.accuracies))
    axes.scatter(iters[:min_iters],st.session_state.accuracies[:min_iters])
    axes.set_title('Validation set accuracy v Label iterations',fontsize=8.0)
    axes.set_xlabel('Label Iterations',fontsize=8.0)
    axes.set_ylabel('Accuracy',fontsize=8.0)
    axes.set_ylim(0,1.02)
    axes.set_yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=8.0)
    #axes.text(1,95,"Target Accuracy",fontsize=8)
    #axes.hlines(0.9,0,i,linestyles='dashed',label='Target accuracy')
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=200)
    plot_cont.image(buf, width=300)
    plt.close(fig)     

def streamlit_display():
    """
    Streamlit Demos Display related text+logic
    """

    # Display title related items and other markdown elements
    display_title()

    # Init session state
    if 'label_itercount' not in st.session_state:
        state_init()
    if st.session_state.label_itercount==0:    
        sidebar_utils()
    if st.session_state.raw_df.empty or st.session_state.features_df.empty or st.session_state.validation_df.empty:
        return
    

    with st.container():
        col11 , col12, col13, col14,col15 = st.columns(5)
        col11.empty()
        col12.empty()
        col13.empty()
        col14.empty()
        col15.empty()
        with col12:
            col12.metric("Number of label requests in session",st.session_state.label_itercount)
        with col13:
            col13.metric("Number of model trainings in session",st.session_state.train_itercount)
        with col14:
            if "remaining_iterations" in st.session_state.train_response_dict.keys():
                remaining_iterations = st.session_state.train_response_dict["remaining_iterations"][0]
            else:
                remaining_iterations = '--'
            col14.metric("Number of labels+trains remaning in session",remaining_iterations)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        st.empty()
    with col2:
        display_training_data()
    with col3:
        display_label_train_options()
    with col4:
        display_test_accuracy()
    with col5:
        st.empty()
    st.write("")
    st.write("")
    with st.container():
        col20,col21,col22,col23,col24,col25,col26,col27 = st.columns(8)
        col20.empty()
        col21.empty()
        col22.empty()
        col23.empty()
        col24.empty()
        col25.empty()
        col26.empty()
        col27.empty()
        with col22:
            sats_spent = st.session_state.label_itercount+st.session_state.train_itercount
            col22.metric("Sats spent in session",sats_spent)
        with col23:
            if "remaining_iterations" in st.session_state.train_response_dict.keys():
                sats_balance = st.session_state.train_response_dict["remaining_iterations"][0]
                col23.metric("Sats balance for session",sats_balance)
            else:
                col23.metric("Sats balance for session","--")
        with col24:
            labeled_df = convert_labels_to_df()
            st.download_button("Download labeled dataset",data=labeled_df,file_name="labeled_dataset.csv",mime="application/octet-stream")
        with col25:
            file_bytes = download_model(st.session_state.session_id,st.session_state.payment_preimage)
            if file_bytes is not None:
                st.download_button("Download Active Learning Model",data = file_bytes,file_name = "active_learning_model.pkl")
            

if __name__=="__main__":
    streamlit_display()

