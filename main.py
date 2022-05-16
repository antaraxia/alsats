from typing import Optional
from fastapi import FastAPI
import uvicorn
from active_learning_utils.core_AL_utils import get_learner
from numpy import array
from pydantic import BaseModel
from traceback import print_exc

app = FastAPI()

class LearnerParams(BaseModel):
  algorithm:str = "rf"
  x_initial:list = None
  y_initial:list = None
  params:Optional[dict] = None


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/mirror")
async def mirror(learner_params:LearnerParams=None):
  return learner_params

@app.get("/initialize")
async def initialize_model(learner_params:LearnerParams=None):
  if learner_params is None:
    return {"Error":"No initialization parameters passed"}


  if learner_params.x_initial is None or learner_params.y_initial is None:
    return {"Error":"x_initial and/or y_initial fields are not present in payload"}

  print([list(map(float,i.split(','))) for i in learner_params.x_initial])
  
  try:
    x_initial = array([list(map(float,i.split(','))) for i in learner_params.x_initial])
    y_initial = array([list(map(float,i.split(','))) for i in learner_params.y_initial])
from replit import db
    learner = get_learner(x_initial,y_initial,learner_params.algorithm)

    score = learner.score(x_initial,y_initial)

    return {"score":score}
    
  except Exception:
    print_exc()
    return {"Error":"Exception occurred"}

  
if __name__=="__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)