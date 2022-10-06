from typing import Optional, Union
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
import uvicorn
import server.server_utils as server #import Service, initialize_iterations_mode, get_session_validity_info
import server.active_learning_utils as al#import TrainParams, LabelParams, train_model, fetch_label
from numpy import array

app = FastAPI()

@app.get("/")
async def home_page():
  return {"ALsats": "Intelligent labeling. For just a few sats."}

@app.post("/pay/initialize/{num_iterations}")
async def pay_initialize(num_iterations:int=None)->dict:
  """
  Returns a Lightning payment request that must be fulfilled to initialize a compute session/model.
  """
  if num_iterations is not None and num_iterations>0:
    print("Requesting session with {} compute iterations...".format(num_iterations))
    iter_dict = server.initialize_iterations_mode(num_iterations)
    content = {"session_id": iter_dict["session_id"], "start_time":iter_dict["start_time"]}
    headers = {"payment_request":iter_dict["payment_request"]}
  else:
    raise HTTPException(status_code=400, detail="Must pass num_iterations greater than zero.")
  return JSONResponse(content=content,headers=headers)


@app.get("/pay/save")
async def pay_save():
  """
  Returns a payment request that must be fulfilled to save a compute session/model.
  """
  pass


@app.post("/train/{session_id}")
async def train(session_id:str=None,train_params:al.TrainParams=None, preimage: Union[str, None] = Header(default=None)):
  """
  Trains an Active Learning model for a valid compute session. Initializes a learner if not initialized.
  Expects a JSON data payload (in the "data" field of the HTTP request).
  payload = json.dumps({"algorithm":"rf",
                        "x_train":[[0.0,1.0],[1.0,2.0]],
                        "y_train":[0,1]
                        })
  """
  print('Session ID is {}'.format(session_id))
  if session_id is None or bool(session_id.strip())==False:
    raise HTTPException(status_code=400, detail="Need valid session ID field")
  
  if preimage is None:
    raise HTTPException(status_code=400, detail="Need preimage in header")

  if train_params is None or train_params.x_train is None or train_params.y_train is None:
    raise HTTPException(status_code=400, detail="Pass a JSON containing all following fields:\"x_train\", \"y_train\" ")
  
  session_validity_info = server.session_validity_info(session_id,preimage)
  if bool(session_validity_info) and session_validity_info["valid_session"]==True:
    # If model hasn't been initialized, initialize it. Else train.
    response_dict = al.train_model(train_params,session_id,session_validity_info["completed_iterations"])
    if response_dict:
      return JSONResponse(content=response_dict)
    else:
      raise HTTPException(status_code=500,detail="Compute failed. You still have {} compute iterations remaining".format(session_validity_info["completed_iterations"]))
  else:
    raise HTTPException(status_code=400, detail="Invalid Session. Either the session has no iterations remaining or payment preimage is not valid ")


@app.post("/label/{session_id}")
async def label(session_id:str=None,label_params:al.LabelParams=None, preimage: Union[str, None] = Header(default=None)):
  """
  Returns a "Label"/"Do not Label" categorization for an inbound feature "x_label" passed in HTTP request params json.
  e.g. "x_label":["1.0,2.0,3.0,4.0","5.0,6.0,7.0,8.0","1.0,2.0,3.0,5.0"]
  """
  if session_id is None or bool(session_id.strip())==False:
    raise HTTPException(status_code=400, detail="Need valid session ID field")
  
  if preimage is None:
    raise HTTPException(status_code=400, detail="Need preimage in header")

  if label_params is None or label_params.x_label is None:
    raise HTTPException(status_code=400, detail="Pass a JSON containing all following fields:\"x_label\"")
  
  session_validity_info = server.session_validity_info(session_id,preimage)
  if session_validity_info["valid_session"]==True:
    # Fetch labels
    response_dict = al.fetch_label(label_params,session_id,session_validity_info["completed_iterations"])
    if response_dict:
      return JSONResponse(content=response_dict)
    else:
      raise HTTPException(status_code=500, detail="Compute failed. You still have {} compute iterations remaining".format(session_validity_info["completed_iterations"]))
  else:
    raise HTTPException(status_code=400, detail="Invalid Session. Either the session has no iterations remaining or payment preimage is not valid ")
 
@app.get("/session_info/{session_id}/{preimage}")
async def session_validity(session_id:str,preimage:str):
  """ Returns session info. Valid sessions show completed iterations. """
  if session_id is None or bool(session_id.strip())==False:
    raise HTTPException(status_code=400, detail="Need valid session ID field")
  if preimage is None or bool(preimage.strip())==False:
    raise HTTPException(status_code=400, detail="Need valid preimage field")
  session_validity_info = server.session_validity_info(session_id,preimage)
  if session_validity_info:
    return JSONResponse(content=session_validity_info,status_code=200)
  else:
    raise HTTPException(status_code=400, detail="Invalid Session. Either the session has no iterations remaining or payment preimage is not valid ")

@app.post("/save")
async def save():
  """
  Saves an Active Learning model versus the session ID. 
  """
  pass


  
if __name__=="__main__":
  uvicorn.run(app, host="127.0.0.1", port=8000)