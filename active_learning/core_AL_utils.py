from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from modAL.models import ActiveLearner
from modAL.uncertainty import classifier_uncertainty

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

                        

                                  



