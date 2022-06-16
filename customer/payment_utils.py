import base64, codecs, json, requests, os
from json import dumps
from hashlib import sha256

class Client:
    def __init__(self, name:str=None, port:str=None):
        if name is None:
            self.lnd_dir = os.environ['CLIENT_LND_DIR']
        else:
            self.lnd_dir = os.environ['LND_ROOT']+name+'/'
        self.tls_cert = self.lnd_dir+'tls.cert'
        if port is None:
            self.lnd_port = os.environ['CLIENT_LND_PORT']
        else:
            self.lnd_port = port
        self.macaroon_path = self.lnd_dir+'data/chain/bitcoin/regtest/admin.macaroon'
        self.lnd_base_url = "https://localhost:"+self.lnd_port+"/"
        self.macaroon = codecs.encode(open(self.macaroon_path, 'rb').read(), 'hex')
        self.headers={'Grpc-Metadata-macaroon': self.macaroon}
        self.balance = self.get_wallet_balance()
        self.pay_threshold = 1000 # Sats by default
    

    def pay_invoice(self,payment_request:str):
        """ Pay invoice encoded by payment request """
        payment_dict=None
        api_endpoint = self.lnd_base_url+'v1/channels/transactions'
        data = dumps({"payment_request":payment_request})
        r = requests.post(api_endpoint,headers=self.headers,verify=self.tls_cert,data=data)
        if r.status_code==200:
            payment_dict=r.json()
        else:
            print("Status Code {} returned.".format(r.status_code))
            print(r.text)
        return payment_dict

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

    def get_wallet_balance(self):
        """ Compute Client Balance in Satoshis """
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




if __name__=="__main__":
    service = Service()
    invoice_dict = service.create_invoice(1000,"First pass of Service Client Code")
    payreq = invoice_dict['payment_request']
    print('Payment Request:\n {}'.format(payreq))

    client = Client(name="carol",port="8083")
    payreq_dict = client.decode_invoice(payreq)
    sats_invoice = int(payreq_dict['num_satoshis'])
    print('Decoding Payment Request...')
    print('Sats Requested -- {}'.format(sats_invoice))
    print('Wallet Available Balance = {}'.format(client.balance))
    
    if client.balance>sats_invoice:
        print('Paying Invoice...')
        payment_dict = client.pay_invoice(payreq)
        payment_preimage = payment_dict['payment_preimage']
        print('Payment PreImage -- \n {}'.format(payment_preimage))

    print("Checking for payment of invoice...")
    paid = service.invoice_paid(payment_preimage)
    if paid is not None and paid==True:
        print("Invoice is paid")

    
