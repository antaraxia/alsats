import base64, codecs, json, requests, os
from json import dumps, loads
from hashlib import sha256
from datetime import datetime
from numpy.random import randint
from tinydb import TinyDB, Query
from tinydb.operations import increment
from traceback import print_exc

#from tinydb_constraint import ConstraintTable
#TinyDB.table_class = ConstraintTable
import os

__RANDMAX__=2e7
__RANDMIN__=-2e7


ALSATS_DIR = os.environ.get('ALSATS_DIR','~/lightning/alsats')

session_db = TinyDB(ALSATS_DIR+'/server/session_info.json')
#session_db.set_schema({'session_id':str,'session_type':str,'payment_request':str,'r_hash':str,'num_iterations':int,'start_time':str,'end_time':str,'completed_iterations':int})

def get_system_params()->dict:
    
    """ Fetches system parameters as a dictionary"""

    f = open(ALSATS_DIR+"/server/system_params.json","r")
    system_params = loads(f.read())
    f.close()
    return system_params

def get_session_info(session_id:str=None)->dict:
    """ Reads a row corresponding session info from the DB """
    session = session_db.search(Query().session_id==session_id)
    return session[0] # should correspond to a single session

def get_session_id()->str:
    
    """ Returns a unique string that identifies a session """
    
    sid_str = datetime.now().isoformat()+str(randint(__RANDMIN__,__RANDMAX__))
    session_id = sha256(bytes(sid_str,'utf-8')).digest().hex()
    return session_id

def get_continuous_mode_fixed_payment()->int:
    
    """ Returns fixed payment corresponding to continuous mode session"""
    
    system_params = get_system_params()
    return system_params["continuous_mode_fixed_payment"]

def update_session(session_id:str=None,success=False)->dict:
    """ Updates a session and returns the session object """
    session=None
    if session_id is not None and success==True:
        session_db.update(increment('completed_iterations'),Query()['session_id']==session_id)
    session = get_session_info(session_id)
    return session

def is_valid_preimage(preimage:str=None,r_hash:str=None):
    """
    Verifies that the preimage produces a valid r_hash.
    Tries both base64 decoding and hex decoding for preimage.
    ASsumes that r_hash is hex encoded.
    """
    if preimage is None or r_hash is None:
        return False

    # Check base64 decoding
    preimage_bytes_b64 = base64.b64decode(preimage)
    payhash_bytes_b64 = sha256(preimage_bytes_b64).digest()
    r_hash_str_b64 = payhash_bytes_b64.hex()
    if r_hash_str_b64 == r_hash:
        return True
    
    # Check hex decoding
    preimage_bytes_hex = bytes.fromhex(preimage)
    payhash_bytes_hex = sha256(preimage_bytes_hex).digest()
    r_hash_str_hex = payhash_bytes_hex.hex()
    if r_hash_str_hex == r_hash:
        return True
    
    return False
    

def save_session_info(session_id:str=None,\
                        session_type:str="continuous",\
                        payment_request:str=None,\
                        r_hash:str=None,\
                        num_iterations:int=1,\
                        start_time:str=None,\
                        end_time:str=None,\
                        completed_iterations:int=0):
    """ Save a session into DB """
    # TODO: Create session info object and save from there
    session_db.insert({'session_id':session_id,'session_type':session_type,'payment_request':payment_request,'r_hash':r_hash,'num_iterations':num_iterations,'start_time':start_time,'end_time':end_time,'completed_iterations':completed_iterations})


def session_validity_info(session_id:str=None,preimage:str=None)->dict:
    """ Is a session valid? 
        Conditions for validity:
        1. Session ID is not None -- Already checked in function calloing this function
        2. Invoice preimage is not None -- Already checked in function calling this function
        3. Session ID exists in the Database
        4. Invoice preimage corresponds to a valid payment hash for that session ID
        5. The number of remaining iterations is not zero
    """
    session_validity_info={"valid_session": False, "completed_iterations" : -1}
    service = Service()
    # Condition 3
    try:
        session = session_db.search(Query().session_id==session_id)
        if session and type(session)==list:
            session = session[0] 
            session_validity_info['completed_iterations'] = session['completed_iterations']
            # Condition 4
            #if is_valid_preimage(preimage,session["r_hash"]):
            if service.invoice_paid(preimage)==True or service.invoice_paid_hex_preim(preimage)==True:
                remaining_iterations = int(session['num_iterations']-session['completed_iterations'])
                # Condition 5
                if remaining_iterations >= 0:
                    session_validity_info['valid_session'] = True
    except Exception:
        print_exc()    
    return session_validity_info
    
def initialize_continuous_mode()->dict:
    
    """ Initializes a fixed price (100 sats) continuous mode session """
    
    continuous_mode_dict = {}
    service = Service()
    session_id = get_session_id()
    session_type = "continuous"
    try:
        # Fetch payment required per continuous mode session and create invoice with session id as memo
        invoice_dict = service.create_invoice(sats=get_continuous_mode_fixed_payment(),memo=session_id)
        
        # TODO: Check if invoice_dict is empty. Raise exception.

        # Fetch payment request that needs to be sent back in returned dict. Also fetch r_hash.
        payment_request = invoice_dict['payment_request']
        r_hash = invoice_dict['r_hash'] 
        start_time=datetime.now().isoformat()
        # Save session info (session_id-->payment_request,r_hash,n_iterations,start_time,proj_end_time) in database
        save_session_info(session_id,session_type,payment_request,r_hash,start_time=start_time)
        # Return response dict 
        continuous_mode_dict = {"session_id":session_id,"payment_request":payment_request, "start_time":start_time}
    except Exception as e:
        print(e)
    
    return continuous_mode_dict

def initialize_iterations_mode(num_iterations:int=1)->dict:
    
    """ Initializes a session for a fixed number of AL iterations. Payment per iteration is fixed. """
    
    iterations_mode_dict = {}
    service = Service()
    session_id = get_session_id()
    session_type = "iterations"
    try:
        # payment = (fixed continuous mode payment * num_iterations) and create invoice with session id as memo
        invoice_dict = service.create_invoice(sats=num_iterations*get_continuous_mode_fixed_payment(),memo=session_id)
        
        # TODO: Check if invoice_dict is empty. Raise exception.

        # Fetch payment request that needs to be sent back in returned dict. Also fetch r_hash.
        payment_request = invoice_dict['payment_request']
        r_hash = invoice_dict['r_hash'] 
        start_time=datetime.now().isoformat()
        # Save session info (session_id-->payment_request,r_hash,n_iterations,start_time,proj_end_time) in database
        save_session_info(session_id,session_type,payment_request,r_hash,num_iterations,start_time=start_time)
        # Return response dict 
        iterations_mode_dict = {"session_id":session_id,"payment_request":payment_request,"start_time":start_time}
    except Exception as e:
        print_exc()
    
    return iterations_mode_dict


class Service:
    def __init__(self):
        self.lnd_dir = os.environ['SERVICE_LND_DIR']
        self.tls_cert = os.environ['SERVICE_LND_CERT']
        self.lnd_port = os.environ['SERVICE_LND_PORT']
        self.macaroon_path = os.environ['SERVICE_MACAROON_PATH']
        self.lnd_base_url = "https://localhost:"+self.lnd_port+"/"
        self.macaroon = codecs.encode(open(self.macaroon_path, 'rb').read(), 'hex')
        self.headers={'Grpc-Metadata-macaroon': self.macaroon}
        self.balance = self.get_wallet_balance()
    

    def create_invoice(self,sats:int,memo:str=None):
        
        """ Create Invoice for service """
        
        invoice_dict={}
        api_endpoint = self.lnd_base_url+'v1/invoices'
        data = dumps({"value":sats,"memo":memo})
        r = requests.post(api_endpoint,headers=self.headers,verify=self.tls_cert,data=data)
        if r.status_code==200:
            invoice_dict=r.json()
        else:
            print("Status Code {} returned.".format(r.status_code))
            print(r.text)
        return invoice_dict

    def decode_invoice(self,payment_request:str):
        """ Decode invoice to find details of payment """
        payreq_dict={}
        api_endpoint = self.lnd_base_url+'v1/payreq/'+payment_request
        r =  requests.get(api_endpoint,headers=self.headers,verify = self.tls_cert)
        if r.status_code==200:
            payreq_dict=r.json()
        else:
            print("Status Code {} returned.".format(r.status_code))
            print(r.text)
        return payreq_dict
         
    def invoice_paid(self,preimage:str)->bool:
        
        """ Checks if invoice was paid. PreImage is a Base64 encoded string"""
        #1. Decode Base64encoded string to get preimage raw bytes base64.b64decode(<base64 encoded string>)
        #2. sha256 preimage raw bytes to get raw bytes of payment hash sha256(<preimage raw bytes>).digest()
        #3. Encode raw bytes of payment hash to suitable format (base64 etc) 
        #3. Use payment hash encoded string to lookup the right invoice
        #4. Check if invoice is settled 
        
        paid = False
        preimage_bytes= base64.b64decode(preimage)
        payhash_bytes = sha256(preimage_bytes).digest()
        r_hash_str = payhash_bytes.hex()
        api_endpoint = self.lnd_base_url+'v1/invoice/'+r_hash_str
        r =  requests.get(api_endpoint,headers=self.headers,verify = self.tls_cert)
        if r.status_code==200:
            payreq_dict=r.json()
            if payreq_dict['settled']==True:
                paid = True
        else:
            print("Status Code {} returned.".format(r.status_code))
            print(r.text)
        return paid

    def invoice_paid_hex_preim(self,preimage:str)->bool:
        
        """ Checks if invoice was paid. PreImage is a hex encoded string"""
        #1. Decode hex encoded string to get preimage raw bytes
        #2. sha256 preimage raw bytes to get raw bytes of payment hash sha256(<preimage raw bytes>).digest() 
        #3. Use payment hash encoded string to lookup the right invoice
        #4. Check if invoice is settled 
        
        paid = False
        preimage_bytes = bytes.fromhex(preimage)
        payhash_bytes = sha256(preimage_bytes).digest()
        r_hash_str = payhash_bytes.hex()
        api_endpoint = self.lnd_base_url+'v1/invoice/'+r_hash_str
        r =  requests.get(api_endpoint,headers=self.headers,verify = self.tls_cert)
        if r.status_code==200:
            payreq_dict=r.json()
            if payreq_dict['settled']==True:
                paid = True
        else:
            print("Status Code {} returned.".format(r.status_code))
            print(r.text)
        return paid       

    def get_wallet_balance(self):
        
        """ Compute Service Wallet Balance in Satoshis """
        
        api_endpoint = self.lnd_base_url+'v1/balance/blockchain'
        r =  requests.get(api_endpoint,headers=self.headers,verify = self.tls_cert)
        if r.status_code==200:
            response_dict = r.json()
            try:
                if 'total_balance' in response_dict.keys():
                    total_balance = int(response_dict['confirmed_balance'])
                else:
                    total_balance = 0

                if 'locked_balance' in response_dict.keys():
                    locked_balance = int(response_dict['locked_balance'])
                else:
                    locked_balance = 0
                
                available_balance = total_balance - locked_balance
            except Exception as e:
                print(e)
                return None
        else:
            print("Status Code {} returned.".format(r.status_code))
            print(r.text)
        return available_balance

    def set_pay_thresh_abs(self,threshold:int):
        
        """ Set a threshold for payment of invoices. Amounts above this threshold are not paid """
        
        return None