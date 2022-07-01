from .server_utils import *
from .cached_models import *
from datetime import datetime

def test_get_system_params():
    """
    Tests if continuous and save payments are a fixed value.
    If these have been changed this test will fail.
    Should privde an additional check for changes made to payment level info.
    """
    system_params = get_system_params()
    assert system_params['continuous_mode_fixed_payment']==1
    assert system_params['save_payment']==2

def test_get_session_id():
    """
    Tests if a valid session id of string type is being generated
    """
    session_id = get_session_id()
    assert type(session_id)==str

def test_save_session_info():
    """
    Tests if a dummy session has been properly saved in the DB.
    """
    session_id = get_session_id()
    session_type = 'iterations'
    payment_request = 'lnbcrt100n1p3t063rpp59480ttn26ewxrq8jpep5gdhqvhh0qj86q7kkk6z9zcchunmtwuasdp4ge5hyum5ypcxzumnyphkvgznv4e8v6trv5syxmrfv4h8ggzrdajx2cqzpgsp5s85yep8u79t8mjcnlwh5qfwgj3lagfxmjm949kuewr9y46grevtq9qyyssqd7zt3qa6m8e5hpzlm4d8l7lrq8mhfde99peul4mprsd74ycsalvxrc7v7ezt956jjpx4l3uz0tdzeuzlfegp8dpyzaujthkq6hwpgncpgc4vxh'
    r_hash = '2d4ef5ae6ad65c6180f20e434436e065eef048fa07ad6b684516317e4f6b773b'
    num_iterations = 5
    start_time = datetime.now().isoformat()
    end_time  = datetime.now().isoformat()
    completed_iterations = 0

    save_session_info(session_id, session_type,payment_request, r_hash, num_iterations,start_time, end_time, completed_iterations)

    session = get_session_info(session_id)
    assert session["session_id"] == session_id
    assert session["session_type"] == session_type
    assert session["num_iterations"] == num_iterations

    return session

def test_update_session():
    """
    Tests if a dummy session has been properly saved in the DB.
    """
    session = test_save_session_info()
    update_session(session['session_id'],success=True)
    session = get_session_info(session['session_id']) # Read again to ensure value got updated
    assert session["completed_iterations"] == 1

    update_session(session['session_id'],success=False)
    session = get_session_info(session['session_id'])
    assert session["completed_iterations"] == 1

def test_session_validity_info():
    """
    Tests if an invalid session is being returned as such.
    And if a valid session is being returned as such
    """
    preimage = 'abcdefgh'
    session = test_save_session_info()
    session_valid_dict = session_validity_info(session['session_id'],preimage)
    assert type(session_valid_dict) == dict
    assert session_valid_dict['valid_session']==False

    # Create invoice, pay and post preimage
    # Create and pay invoice on alice. take hex encoded r_preimage from lncli listinvoices and paste it here.
    preimage = '25a5f9fa1d9131af5ab180f51867576c6ad845fcb3f72e65765b4336d1a4de4a'  # Hex encoded payment preimage
    #preimage = base64.b64encode(bytes.fromhex(preimage_hex)).decode("utf-8") # Base64 encoded preimage
    session = test_save_session_info()
    session_valid_dict = session_validity_info(session['session_id'],preimage)
    assert type(session_valid_dict) == dict
    assert session_valid_dict['valid_session']==True

def test_initialize_iterations():
    """
    Tests if a iterations mode session is being initialized correctly or not
    """
    num_iterations = 5
    iterations_mode_dict = initialize_iterations_mode(num_iterations)
    assert bool(iterations_mode_dict) == True
    assert type(iterations_mode_dict['session_id'])==str
    session = get_session_info(iterations_mode_dict['session_id'])
    #assert type(session) == dict
    payment_request = session['payment_request']
    assert type(payment_request) == str
    service = Service()
    payreq_dict = service.decode_invoice(payment_request)
    sats_invoice = int(payreq_dict['num_satoshis'])
    ref_payment = num_iterations*get_continuous_mode_fixed_payment()
    assert sats_invoice==ref_payment





    
    

    