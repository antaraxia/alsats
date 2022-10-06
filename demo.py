from sklearn.datasets import fetch_openml
from fastapi.testclient import TestClient
from main import app
from customer.payment_utils import Client
from server.server_utils import Service
from json import loads, dumps
from numpy import random, array
import streamlit as st
from pandas import read_csv
import matplotlib.pyplot as plt
from io import BytesIO
from time import sleep


test_app = TestClient(app)
# Needs mnist.csv with a 28X28 image unraveled into a 784 dim vector and the corresponding label as the 729th column
dataset = read_csv('./mnist.csv')
X = dataset.to_numpy()
X,y = X[:,:-1],X[:,-1]

def rolling_accuracy(pred_class:array,ref_class:array,num_points=100):
    """
    Computes prediction accuracy of last 100 points
    """
    if len(pred_class)<num_points:
        accuracy = sum(pred_class==ref_class)/len(pred_class)
    else:
        accuracy = sum(pred_class[-num_points:]==ref_class[-num_points:])/num_points
    
    return accuracy

def streamlit_display():
    """
    Streamlit Demos Display related text+logic
    """

    # Start display
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
    st.subheader('*Label data intelligently using Active Learning. Pay in sats *only* for the compute you consume.*')

    st.header("About the demo")
    st.markdown("Here, we give the classic MNIST classification task an Active Learning twist.")
    st.markdown("Instead of using all ~70K labeled examples and working hard on improving classification accuracy, the demo aims to minimize the number of labeled examples needed to get a target classifier accuracy (90%)")
    st.markdown("The client (API user) first enters a desired number of train + label iterations. Under the hood, the demo code hits the alsats /pay/initialize API endpoint to pay for the requested number of compute iterations.")
    st.markdown("Next, a MNIST image is selected at client-side randomly and sent to the /label endpoint. Here the Active Learner initialized for the compute session decides and communicates back to the client if this image needs labeling.")
    st.markdown("If the image needs labeling, the image and its label are sent to the /train endpoint so that the Active Learner can iteratively be trained and improved.")
    st.markdown("The process is repeated until the number of iterations paid for are reached.")
    st.text(" ")
    st.text(" ")
    # Init client (bob)
    client = Client()
    
    # Init service (alice)
    service = Service()
 
    with st.container() as cont1:
        _,_,col_iters,_,cost_per,_,col_cost,_ = st.columns(8)
        with col_iters:
            num_iterations = st.number_input("Num. train + label iterations requested ",min_value=1,max_value=10000)
        with cost_per:
            cost_per.metric("Cost per iteration (sats)",value=1)
        with col_cost:
            col_cost.metric("Total cost (sats)",value=num_iterations)
        sleep(2)

        st.text(" ")
        st.text(" ")
        _,_,col1,col3,col4,col5,_,col6,_,_ = st.columns(10)
        
        with col1:
            emp_count1 = st.empty()
            emp_count2 = st.empty()
            emp_count3 = st.empty()
            emp_count4 = st.empty()
            emp_count5 = st.empty()
            img_cont = st.empty()
        with col3:
            plot_cont = st.empty()
        with col6:
            iter_count = st.empty()
            client_bal = st.empty()
            service_bal = st.empty()
        
        if num_iterations and num_iterations>0:    
            response = test_app.post('/pay/initialize/'+str(num_iterations))
            headers = response.headers
            response_payload = response.json()
            payment_request = headers["payment_request"]
            session_id = response_payload["session_id"]
            payreq_dict = client.decode_invoice(payment_request)
            sats_invoice = int(payreq_dict["num_satoshis"])
            if client.local_channel_balance>sats_invoice:

                print('Paying Invoice...')
                payment_dict = client.pay_invoice(payment_request)
                payment_preimage = payment_dict['payment_preimage']
                print('Payment PreImage -- \n {}'.format(payment_preimage))

                init_idx = random.randint(0,X.shape[0])
                x_train = [list(X[init_idx])]
                y_train = [int(y[init_idx])]
                data = dumps({"algorithm":"rf","x_train":x_train,"y_train":y_train})
                headers = {"preimage":payment_preimage}
                response = test_app.post("/train/"+session_id,headers=headers,data=data)
                i=0

                algo_score=[]
                iterations=[]
                pred_class=[]
                ref_class=[]
                roll_acc=[]
                if response.status_code==200:
                    algo_score.append(response.json()["score"])
                    iterations.append(i)
                    ref_class.append(int(y_train[0]))
                    pred_class.append(int(response.json()["class"]))
                    roll_acc.append(rolling_accuracy(array(pred_class),array(ref_class)))
                    while i<num_iterations:
                        idx = random.randint(0,X.shape[0])
                        # First ask for label
                        x_label = [list(X[idx])]
                        headers = {"preimage":payment_preimage}
                        data = dumps({"algorithm":"rf","x_label":x_label})
                        response = test_app.post("/label/"+session_id,headers=headers,data=data)
                        if response.status_code==200:
                            i+=1
                            if response.json()["decision"]=="label":
                                y_train = [int(y[idx])]
                                x_train = x_label
                                data = dumps({"algorithm":"rf","x_train":x_train,"y_train":y_train})
                                response = test_app.post("/train/"+session_id,headers=headers,data=data)
                                if response.status_code==200:
                                    i+=1
                                    algo_score.append(response.json()["score"])
                                    iterations.append(i)
                                    ref_class.append(int(y_train[0]))
                                    pred_class.append(int(response.json()["class"]))
                                    roll_acc.append(rolling_accuracy(array(pred_class),array(ref_class)))

                            with col3:
                                fig,axes = plt.subplots(figsize=(5,3))
                                axes.scatter(iterations,roll_acc)
                                axes.set_title('Rolling accuracy (last 100 pred.)')
                                axes.set_xlabel('Train + label Iterations')
                                axes.set_ylabel('Accuracy')
                                axes.set_ylim(0,1.0)
                                axes.set_yticks([i*0.1 for i in range(10)])
                                axes.text(1,0.95,"Target Accuracy",fontsize=8)
                                axes.hlines(0.9,0,i,linestyles='dashed',label='Target accuracy')
                                buf = BytesIO()
                                fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
                                plot_cont.image(buf, width=300)
                                plt.close(fig)
                            with col1:
                                emp_count1.write(" ")
                                emp_count2.write(" ")
                                emp_count3.write(" ")
                                emp_count4.write(" ")
                                emp_count5.write(" ")
                                img = X[idx].reshape((28,28))
                                img_cont.image(img,caption='Input image',clamp=True,use_column_width='auto')
                            with col6:
                                iter_count.metric("Num. Iters. completed",value=i)
                                client_bal.metric("Client Channel Balance (Sats)",value=client.get_channel_local_balance())
                                service_bal.metric("Service Channel Balance (Sats)",value=service.get_channel_local_balance())

if __name__=="__main__":
    streamlit_display()