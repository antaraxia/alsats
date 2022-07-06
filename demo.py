from sklearn.datasets import fetch_openml
from fastapi.testclient import TestClient
from main import app
from customer.payment_utils import Client
from json import loads, dumps
from numpy import random
import streamlit as st


test_app = TestClient(app)
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

def client_name_to_port(client_name:str=None)->str:
    "Maps client name to port number for demo"
    name_to_port = {"bob":"8082","carol":"8083"}
    return name_to_port[client_name]

def active_learning_run(client:Client=None, num_iterations:int=20):
    """
    Performs an Active Learning run for the Client on the MNIST dataset.
    """
    print('{} number of iterations requested.'.format(num_iterations))
    response = test_app.post('/pay/initialize/'+str(num_iterations))
    print('{} response code'.format(response.status_code))
    headers = response.headers
    response_payload = response.json()
    payment_request = headers['payment_request']
    session_id = response_payload['session_id']
    payreq_dict = client.decode_invoice(payment_request)
    sats_invoice = int(payreq_dict['num_satoshis'])
    if client.balance>sats_invoice:

        print('Paying Invoice...')
        payment_dict = client.pay_invoice(payment_request)
        payment_preimage = payment_dict['payment_preimage']
        print('Payment PreImage -- \n {}'.format(payment_preimage))

        init_idx = random.randint(0,X.shape[0])
        x_train = [list(X[init_idx])]
        print(x_train)
        y_train = [int(y[init_idx])]
        print(y_train)
        data = dumps({"algorithm":"rf","x_train":x_train,"y_train":y_train})
        response = test_app.post("/train/"+session_id,headers=headers,data=data)
        print(response.status_code)
        print(response.json["detail"])
    else:
        return

def streamlit_display():
    """
    Streamlit Demos Display related text+logic
    """
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
    st.title('alsats - active learning (for a few) sats')

    st.markdown('*Label data intelligently using Active Learning. Pay in (mili)sats *only* for the compute you consume.*')
    st.markdown('alsats reduces the time and cost needed to create minimum viable datasets in supervised learning problems.')
    st.markdown('It has the following features:')
    st.markdown('* API based model training and data labeling.')
    st.markdown('* Intelligent labeling - Label training examples based on model uncertainty predictions for said examples.')
    st.markdown('* Iterative Learning - Learn models starting with as little as one data point. As more data is labeled and trained, model metrics and labeling suggestions improve.')
    st.markdown('* Flexible payments - Pay ONLY for the compute consumed in the process.')
    st.markdown('* Train and label trustlessly - No need to register/sign-up for an account.')
    st.markdown('* Data security - alsats does not store any user data in the current implementation. Future implementations will store data only if the customer wants to.')
    st.markdown('Below, we demonstrate some of the key features of alsats.')
    
    client_name='bob'
    client = Client(name=client_name,port=client_name_to_port(client_name))
    client_balance = client.get_wallet_balance()
    num_iterations = st.text_input("Number of train + label iterations")
    st.button("Start",on_click=active_learning_run,args=(client,num_iterations))



if __name__=="__main__":
    streamlit_display()

