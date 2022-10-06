from typing import Optional
from pydantic import BaseModel
from numpy import array
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import classifier_uncertainty
from .server_utils import update_session
from .cached_models import models
from traceback import print_exc

class TrainParams(BaseModel):
    algorithm:str = "rf"
    x_train:list = None
    y_train:list = None
    params:Optional[dict] = None

class LabelParams(BaseModel):
    algorithm:str = "rf"
    x_label:list = None
    params:Optional[dict] = None

def get_classifier(algo:str="rf",params:dict=None):
  """
  Maps a algorithm string to sklearn classifier.
  Returns a Gradient Boosted Classifier or a Random Forest classifier.
  """
  algo_to_classifier_map = {"rf":RandomForestClassifier(),\
                        "gbc":GradientBoostingClassifier()}

  return algo_to_classifier_map[algo]

def get_learner(X_train:array=None,y_train:array=None,algo="rf",params:dict=None)->ActiveLearner:
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

def get_learner_score(learner:ActiveLearner,X:array=None,y:array=None)->float:
  """
  Returns score/accuracy of the initialized active learner
  """
  return learner.score(X,y)

def streamed_sampling_iteration(learner:ActiveLearner=None,
                                X_candidate:array=None,
                                uncertainty_threshold:float=0.5)->dict:
  """
  Computes classifier uncertainty for a single candidate feature vector/input.
  Compares this uncertainty to a threshold passed by the user to determine if labeling is needed.

  Returns: {"classifier_uncertainty":float,"label":"Yes"}
  """
  label = "label" # By default ask for a label
                                  
  uncertainty = classifier_uncertainty(learner, X_candidate.reshape(1, -1)) 
                                  
  if uncertainty is not None and uncertainty < uncertainty_threshold:
      label = "do not label"

  return {"uncertainty":uncertainty,"label":label}
  
def train_learner_single_sample(learner:ActiveLearner=None,
                                X_sample:array=None,
                                y_sample:array=None)->ActiveLearner:
  """
  Single training pass for the Active Learner given a feature vector X_sample and label y_sample
  """
  learner.teach(X_sample.reshape(1,-1),y_sample.reshape(1,-1))
  return learner


def train_model(train_params:TrainParams=None,session_id:str=None,completed_iterations:int=None):
  """
  Train an active learner. Existing active learner trains only on new data.
  New active learner gets initialized. Session gets updated. 
  """
  response_dict = {"message":"Train unsuccessful. Internal error. {} compute iterations completed in this session".format(completed_iterations),\
    "score":None}
  
  try:
    #TODO - Check for dimension consistency of inputs and outputs.
    x_train = array(train_params.x_train)
    y_train = array(train_params.y_train).ravel()
    if session_id not in models.keys():
      learner = get_learner(x_train,y_train,train_params.algorithm)
      models[session_id] = learner
    else:
      learner = models[session_id]
    learner.teach(x_train,y_train)
    predicted_class = learner.predict(x_train)
    session = update_session(session_id,success=True)
    score = learner.score(x_train,y_train)
    remaining_iterations = int(session["num_iterations"]-session["completed_iterations"])
    response_dict = {"message":"Train successful, {} compute iterations completed,\
                      {} iterations remaining".format(session["completed_iterations"],remaining_iterations),\
                      "score":score,"remaining_iterations":[remaining_iterations],\
                      "predicted_label":predicted_class}
  except Exception as e:
    print_exc(e)
  
  return response_dict

def fetch_label(label_params:LabelParams=None,session_id:str=None,completed_iterations:int=None):
  """
  Label an incoming feature vector.
  """
  response_dict={"message":"Label unsuccessful. Internal error. You still have {} compute iterations in this session".format(completed_iterations)}
  try:
    #TODO - Check for dimension consistency of inputs
    x_label = array([label_params.x_label])

    # Currently labeling is allowed only after training for at least one iteration
    if session_id in models.keys(): # ==> you have a pre-trained model, need to return error otherwise
      learner = models[session_id]
      predicted_class = learner.predict(x_label)
      label_dict = streamed_sampling_iteration(learner, x_label, 0.5) # TODO make uncertainty threshold a parameter and remove hardcoding.
      if label_dict:
        session = update_session(session_id,success=True)
        remaining_iterations = int(session["completed_iterations"]-session["num_iterations"])
        response_dict = {"message":"Label request successful, {} compute iterations completed,\
        {} iterations remaining in session.".format(session["completed_iterations"],remaining_iterations),\
        "decision":label_dict["label"], "uncertainty":list(label_dict["uncertainty"]),\
        "remaining_iterations":[remaining_iterations],\
        "predicted_label":predicted_class}
    else:
        session = update_session(session_id,success=True)
        remaining_iterations = int(session["completed_iterations"]-session["num_iterations"])
        response_dict = {"message":"Data point needs labeling, {} compute iterations completed,\
        {} iterations remaining in session.".format(session["completed_iterations"],remaining_iterations),\
        "decision":"label", "uncertainty":[1.0],\
        "remaining_iterations":[remaining_iterations],\
        "predicted_label":"None"}      
  except Exception as e:
    print_exc(e)
  
  return response_dict

