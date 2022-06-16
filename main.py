from typing import Optional
from fastapi import FastAPI, HTTPException
import uvicorn
from active_learning.core_AL_utils import get_learner
from server.server_utils import Service, initialize_continuous_mode, initialize_iterations_mode
from server.active_learning_utils import LearnerParams
from numpy import array
from pydantic import BaseModel
from traceback import print_exc

app = FastAPI()



@app.get("/")
async def home_page():
    return {"Alpa": "Intelligent labeling. For just a few sats."}}

@app.get("/mirror")
async def mirror(learner_params:LearnerParams=None):
  return learner_params

@app.post("/handshake/{session_type}/{num_iterations}")
async def hand_shake(session_type: str="continuous",num_iterations:int=None):
  """ Initial handshake between client and service where they exchange info about  """
  service = Service()
  response_dict=None
  if session_type=="continuous":
    response_dict = initialize_continuous_mode()
  elif session_type=="iterations":
    response_dict = initialize_iterations_mode(num_iterations)
  else:
    raise HTTPException(status_code=404, detail="Item not found")
  return response_dict

@app.post("/initialize/{session_id}")
async def initialize(session_id:str=None,learner_params:LearnerParams=None):
  
  initilization_dict = {}
  if session_id is None:
    raise HTTPException(status_code=404, detail="Item not found")
  if (learner_params is None or learner_params.x_initial is None or learner_params.y_initial is None or learner_params.preimage is None :
    raise HTTPException(status_code=400, detail="Pass a JSON containing \"algorithm\", \"x_initial\", \"y_initial\" and \"preimage\" fields.")
  if session_valid(session_id,learner_params.preimage):
    initilization_dict = initialize_model(learner_params)

  return initilization_dict

  
if __name__=="__main__":
  uvicorn.run(app, host="127.0.0.1", port=8000)