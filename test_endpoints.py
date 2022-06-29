from fastapi.testclient import TestClient
from alsats import app
client = TestClient(app)

def test_pay_initialize():
    """
    Test the /pay/initialize/{num_iterations} endpoint
    """
    response = client.post('/pay/initialize/20')
    assert response.status_code == 200
    headers = response.headers
    assert type(headers["payment_request"]) == str 
    response_payload = response.json()
    assert type(response_payload["session_id"]) == str