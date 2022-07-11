from fastapi.testclient import TestClient
from main import app
from customer.payment_utils import Client
from json import loads, dumps
from numpy import random
from server.server_utils import get_session_info
test_app = TestClient(app)

def test_pay_initialize():
    """
    Test the /pay/initialize/{num_iterations} endpoint
    """
    response = test_app.post('/pay/initialize/20')
    assert response.status_code == 200
    headers = response.headers
    assert type(headers["payment_request"]) == str 
    response_payload = response.json()
    assert type(response_payload["session_id"]) == str
    return response_payload, headers


def test_train():
    """
    Test the /train endpoint
    """
    # test if endpoint throws 404 when session id is not provided
    response = test_app.post("/train/")
    assert response.status_code==404
    assert response.json()["detail"] == "Not Found"

    # test if endpoint throws 404 when session id is empty string or spaces
    response = test_app.post("/train/ ")
    assert response.status_code==400
    assert response.json()["detail"] == "Need valid session ID field"

    # test if endpoint throws 404 when header doesnt contain preimage
    response = test_app.post("/train/abcd")
    assert response.status_code==400
    assert response.json()["detail"] == "Need preimage in header"

    # test if endpoint throws 404 when header contains preimage but no train params
    headers = {"preimage":"abcd"}
    response = test_app.post("/train/abcd",headers=headers)
    assert response.status_code==400
    assert "Pass a JSON containing all" in response.json()["detail"]

    # test if endpoint throws 400 when header and session info don't match anything in the DB
    # but train params have been passed.
    headers = {"preimage":"abcd"}
    data = dumps({"algorithm":"rf","x_train":[[0.0,2.0,2.0]],"y_train":[1.0]})
    response = test_app.post("/train/abcd",headers=headers,data=data)
    assert response.status_code==400
    assert "Invalid Session" in response.json()["detail"]

    # test a fully valid session - 1 iteration
    response_payload, headers = test_pay_initialize()
    session_id = response_payload["session_id"]
    payment_request = headers['payment_request']
    al_client = Client(name="bob",port="8082")
    payreq_dict = al_client.decode_invoice(payment_request)
    sats_invoice = int(payreq_dict['num_satoshis'])

    if al_client.balance>sats_invoice:
        payment_dict = al_client.pay_invoice(payment_request)
        payment_preimage = payment_dict['payment_preimage']
        print('Payment PreImage -- \n {}'.format(payment_preimage))
        headers = {"preimage":payment_preimage}
        response = test_app.post("/train/"+session_id,headers=headers,data=data)
        assert response.status_code==200
        assert "Train successful" in response.json()["message"]
        session = get_session_info(session_id)
        assert session["completed_iterations"] == 1
    else:
        assert True==False

    # Continue to train until client session is valid - i.e. for 20 sessions
    for i in range(1,21):
        x = [[random.randint(0,10),random.randint(0,10),random.randint(0,10)]]
        y = [random.randint(0.0,1.0)]
        data = dumps({"algorithm":"rf","x_train":x,"y_train":y})
        response = test_app.post("/train/"+session_id,headers=headers,data=data)
        assert response.status_code==200
        session = get_session_info(session_id)
        assert session["completed_iterations"] == i+1
    # Run it one more time to check for invalid session
    response = test_app.post("/train/"+session_id,headers=headers,data=data)
    assert response.status_code==400
    assert "Invalid Session" in response.json()["detail"]   

    # TODO test for inconsistent dimensionality and graceful exit message

def test_label():
    """
    Test the /train endpoint
    """
    # test if endpoint throws 404 when session id is not provided
    response = test_app.post("/label/")
    assert response.status_code==404
    assert response.json()["detail"] == "Not Found"

    # test if endpoint throws 404 when session id is empty string or spaces
    response = test_app.post("/label/ ")
    assert response.status_code==400
    assert response.json()["detail"] == "Need valid session ID field"

    # test if endpoint throws 400 when header doesnt contain preimage
    response = test_app.post("/label/abcd")
    assert response.status_code==400
    assert response.json()["detail"] == "Need preimage in header"

    # test if endpoint throws 400 when header contains preimage but no train params
    headers = {"preimage":"abcd"}
    response = test_app.post("/label/abcd",headers=headers)
    assert response.status_code==400
    assert "Pass a JSON containing all" in response.json()["detail"]

    # test if endpoint throws 400 when header and session info don't match anything in the DB
    # but train params have been passed.
    headers = {"preimage":"abcd"}
    train_data = dumps({"algorithm":"rf","x_train":[[0.0,2.0,2.0]],"y_train":[0]})
    label_data = dumps({"algorithm":"rf","x_label":[[0.0,1.0,1.0]]})
    response = test_app.post("/label/abcd",headers=headers,data=label_data)
    assert response.status_code==400
    assert "Invalid Session" in response.json()["detail"]

    # Test valid initialize+train+label session
    response_payload, headers = test_pay_initialize()
    session_id = response_payload["session_id"]
    payment_request = headers['payment_request']
    al_client = Client(name="bob",port="8082")
    payreq_dict = al_client.decode_invoice(payment_request)
    sats_invoice = int(payreq_dict['num_satoshis'])

    if al_client.balance>sats_invoice:
        payment_dict = al_client.pay_invoice(payment_request)
        payment_preimage = payment_dict['payment_preimage']
        print('Payment PreImage -- \n {}'.format(payment_preimage))
        headers = {"preimage":payment_preimage}
        response = test_app.post("/train/"+session_id,headers=headers,data=train_data)
        assert response.status_code==200
        assert "Train successful" in response.json()["message"]
        session = get_session_info(session_id)
        assert session["completed_iterations"] == 1
    else:
        assert True==False

    
    # Continue to train until client session is valid - i.e. for 20 sessions
    for i in range(1,21):
        x = [[random.randint(0,10),random.randint(0,10),random.randint(0,10)]]
        data = dumps({"algorithm":"rf","x_label":x})
        response = test_app.post("/label/"+session_id,headers=headers,data=data)
        assert response.status_code==200
        session = get_session_info(session_id)
        assert session["completed_iterations"] == i+1
    # Run it one more time to check for invalid session
    response = test_app.post("/label/"+session_id,headers=headers,data=data)
    assert response.status_code==400
    assert "Invalid Session" in response.json()["detail"]   

