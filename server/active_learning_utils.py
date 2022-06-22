from typing import Optional
from pydantic import BaseModel
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import classifier_uncertainty
from server_utils import update_session
from cached_models import models

class TrainParams(BaseModel):
  def __init__(self):
    self.algorithm:str = "rf"
    self.x_train:list = None
    self.y_train:list = None
    self.params:Optional[dict] = None

class LabelParams(BaseModel):
  def __init__(self):
    self.algorithm:str = "rf"
    self.x_label:list = None
    self.params:Optional[dict] = None

def get_classifier(algo:str="rf",params:dict=None):
  """
  Maps a algorithm string to sklearn classifier.
  Returns a Gradient Boosted Classifier or a Random Forest classifier.
  """
  algo_to_classifier_map = {"rf":RandomForestClassifier(),\
                        "gbc":GradientBoostingClassifier()}

  return algo_to_classifier_map[algo]

def get_learner(X_train:ndarray=None,y_train:ndarray=None,algo="rf",params:dict=None)->ActiveLearner:
  """
  Returns an initially trained modAL Active Learner given initial training inputs
  and classifier options
  """

  classifier = get_classifier(algo,params)

  # Check for X and y values and raise error
  if X_train is None or y_train is None:
    raise ValueError("X or y is None. X and y need to be Numpy ndarrays")
    
  # initialize the learner
  learner = ActiveLearner(
      estimator=classifier,
      X_training=X_train, y_training=y_train
  )
  return learner

def get_learner_score(learner:ActiveLearner,X:ndarray=None,y:ndarray=None)->float:
  """
  Returns score/accuracy of the initialized active learner
  """
  return learner.score(X,y)

def streamed_sampling_iteration(learner:ActiveLearner=None,
                                X_candidate:ndarray=None,
                                uncertainty_threshold:float=0.5)->dict:
  """
  Computes classifier uncertainty for a single candidate feature vector/input.
  Compares this uncertainty to a threshold passed by the user to determine if labeling is needed.

  Returns: {"classifier_uncertainty":float,"label":"Yes"}
  """
  label = "Yes" # By default ask for a label
  uncertainty = 1 # By default high uncertainty
                                  
  uncertainty = classifier_uncertainty(learner, X_candidate.reshape(1, -1)) 
                                  
  if classifier_uncertainty > uncertainty_threshold:
    label = "No"

  return {"classifier_uncertainty":uncertainty,"label":label}
  
def train_learner_single_sample(learner:ActiveLearner=None,
                                X_sample:ndarray=None,
                                y_sample:ndarray=None)->ActiveLearner:
  """
  Single training pass for the Active Learner given a feature vector X_sample and label y_sample
  """
  learner.teach(X_sample.reshape(1,-1),y_sample.reshape(1,-1))
  return learner


def train_model(train_params:TrainParams=None,session_id:str=None,completed_iterations:int=None):
  response_dict={}
  
  try:
    x_train = ndarray([list(map(float,i.split(','))) for i in train_params.x_train])
    y_train = ndarray([list(map(float,i.split(','))) for i in train_params.y_train])

    if completed_iterations==0: # Valid session, not yet started
      learner = get_learner(x_train,y_train,train_params.algorithm)
      models[session_id] = learner
    else: # Fetch learner and teach.
      learner = models[session_id]
      learner.teach(x_train,y_train)
    score = learner.score(x_train,y_train)
    session = update_session(session_id,success=True)
    remaining_iterations = int(session["completed_iterations"]-session["num_iterations"])
    response_dict = {"message":"Train successful, {} compute iterations completed,\
                      {} iterations remaining".format(session["completed_iterations"],remaining_iterations),"score":score}
  except Exception as e:
    print(e)
  
  return response_dict
