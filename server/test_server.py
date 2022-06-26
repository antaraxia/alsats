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
    assert system_params['continuous_mode_fixed_payment']==10
    assert system_params['save_payment']==20

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
    payment_request = 'ABC***SBCDSRE'
    r_hash = 'ss2433dS'
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
    preimage_hex = 'f088bc132f986e63b78a645d7ad3a882fdad3b3c0c823e4a5be02c2afb24a100'  # Hex encoded payment preimage
    preimage = base64.b64encode(bytes.fromhex(preimage_hex)).decode("utf-8") # Base64 encoded preimage
    session = test_save_session_info()
    session_valid_dict = session_validity_info(session['session_id'],preimage)
    assert type(session_valid_dict) == dict
    assert session_valid_dict['valid_session']==True


    
    

    