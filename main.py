from typing import Optional, Union
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
import uvicorn
from server.server_utils import Service, initialize_continuous_mode, initialize_iterations_mode, session_valid
from server.active_learning_utils import TrainParams, LabelParams, train_model
from numpy import array
from pydantic import BaseModel
from traceback import print_exc

app = FastAPI()



@app.get("/")
async def home_page():
    return {"Alpa": "Intelligent labeling. For just a few sats."}}

@app.post("/pay/initialize/{num_iterations}")
async def pay_initialize(num_iterations:int=None)->dict:
  """
  Returns a Lightning payment request that must be fulfilled to initialize a compute session/model.
  """
  service = Service()
  content = headers = {}
  if num_iterations is not None and num_iterations>0:
    iter_dict = initialize_iterations_mode(num_iterations)
    content["session_id"] = iter_dict["session_id"]
    content["start_time"] = iter_dict["start_time"]
    headers["payment_request"] = iter_dict["payment_request"]
  else:
    raise HTTPException(status_code=400, detail="Must pass num_iterations greater than zero.")
  return JSONResponse(content=content,headers=headers)


@app.get("/pay/save"):
async def pay_save():
  """
  Returns a payment request that must be fulfilled to save a compute session/model.
  """
  pass


@app.get("/train/{session_id}")
async def train(session_id:str=None,train_params:TrainParams=None, preimage: Union[str, None] = Header(default=None)):
  """
  Trains an Active Learning model for a valid compute session. Initializes a learner if not initialized.
  Must be provided with "x_train", "y_train" in the params json as a list of comma separated strings.
  The strings themselves are features and labels respectively.
  e.g. "x_train":["1.0,2.0,3.0,4.0","5.0,6.0,7.0,8.0","1.0,2.0,3.0,5.0"] and "y_train":["1","1","0"]
  """`
  if session_id is None:
    raise HTTPException(status_code=400, detail="Need valid session ID field")
  
  if preimage is None:
    raise HTTPException(status_code=400, detail="Need valid preimage")

  if (train_params is None or train_params.x_train is None or train_params.y_train is None:
    raise HTTPException(status_code=400, detail="Pass a JSON containing all following fields:\"x_train\", \"y_train\" ")
  
  session_validity_info = get_session_validity_info(session_id,preimage)
  if session_validity_info["valid_session"]==True:
    # If model hasn't been initialized, initialize it. Else train.
    response_dict = train_model(train_params,session_id,session_validity_info["completed_iterations"])
    if response_dict:
      return JSONResponse(content=response_dict)
    else:
      raise HTTPException(status_code=500,"Compute failed. You still have {} compute iterations remaining".format(session_validity_info["completed_iterations"]))
  else:
    raise HTTPException(status_code=400, detail="Invalid Session. Either the session has no iterations remaining or payment preimage is not valid ")


@app.get("/label/{session_id}")
async def label(session_id:str=None,label_params:LabelParams=None, preimage: Union[str, None] = Header(default=None)):
  """
  Returns a "Label"/"Do not Label" categorization for an inbound feature "x_label" passed in HTTP request params json.
  e.g. "x_label":["1.0,2.0,3.0,4.0","5.0,6.0,7.0,8.0","1.0,2.0,3.0,5.0"]
  """
  if session_id is None:
    raise HTTPException(status_code=400, detail="Need valid session ID field")
  
  if preimage is None:
    raise HTTPException(status_code=400, detail="Need valid preimage")

  if (train_params is None or train_params.x_train is None or train_params.y_train is None:
    raise HTTPException(status_code=400, detail="Pass a JSON containing all following fields:\"x_train\", \"y_train\" ")
  
  session_validity_info = get_session_validity_info(session_id,preimage)
  if session_validity_info["valid_session"]==True:
    # Fetch labels
    response_dict = train_model(train_params,session_id,session_validity_info["completed_iterations"])
    if response_dict:
      return JSONResponse(content=response_dict)
    else:
      raise HTTPException(status_code=500,"Compute failed. You still have {} compute iterations remaining".format(session_validity_info["completed_iterations"]))
  else:
    raise HTTPException(status_code=400, detail="Invalid Session. Either the session has no iterations remaining or payment preimage is not valid ")
 

@app.post("/save")
aync def save():
  """
  Saves an Active Learning model versus the session ID. 
  """
  pass


@app.post("/compute/initialize/{session_id}")
async def initialize(session_id:str=None,learner_params:LearnerParams=None):
  
  initialization_dict = {}
  if session_id is None:
    raise HTTPException(status_code=404, detail="Item not found")

  if (learner_params is None or learner_params.x_initial is None or learner_params.y_initial is None or learner_params.preimage is None :
    raise HTTPException(status_code=400, detail="Pass a JSON containing \"algorithm\", \"x_initial\", \"y_initial\" and \"preimage\" fields.")
  
  if session_valid(session_id,learner_params.preimage):
    initialization_dict = initialize_model(learner_params)

  return initialization_dict

@app.post("/compute/teach/{session_id}"):
async def teach(session_id:str=None,learner_params:LearnerParams=None):
  pass

@post.post("/compute/label/{session_id}"):
async def label(session_id:str=)
  
if __name__=="__main__":
  uvicorn.run(app, host="127.0.0.1", port=8000)