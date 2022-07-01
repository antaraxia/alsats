# AL problem -- Learn a square using streamed sampling implemented in active_learning_utils.py
# and using the methods in server_utils.py
# i.e. everything within the boundaries of a square of length 'a' = 1 centered at 
# (x,y)=(a/2,a/2) everything outside is 0.
from .active_learning_utils import *
from .server_utils import *
from numpy import linspace, random


def test_create_dataset():
    """
    Create a X,y dataset with each row in X as coordinates in the x,y place and y = 0 or 1
    e.g. x1 = 0.0,1.0" y1 = 1
    """
    x1 = x2 = linspace(0,10,20) 
    x_coords = [[i,j] for i in x1 for j in x2]
    y_coords = [1 if i<=5.0 and j<=5.0 else 0 for i in x1 for j in x2]
    assert x_coords[0] == [0,0]
    assert y_coords[0] == 1
    index_extreme = x_coords.index([10.0,10.0])
    assert y_coords[index_extreme] == 0
    return x_coords, y_coords

def test_al_iteration():
    """
    Test a streamed sampling iteration
    """
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=array([[1.0,1.0]]), y_training=array([1])
    )

    uncertainty_dict = streamed_sampling_iteration(learner,array([2.0,2.0]),0.5)
    assert bool(uncertainty_dict) == True


def test_init_model():
    """
    Use the test dataset created to initialize a session and save an AL model
    """
    # Create dataset and modify to format suitable for learning
    x,y = test_create_dataset()
    x_init = [",".join(map(str,x[0])), ",".join(map(str,x[-1]))]
    y_init = [str(y[0]),str(y[-1])]
    assert x_init ==["0.0,0.0","10.0,10.0"]
    assert y_init == ["1","0"]
    # Create Train params
    train_params = TrainParams()
    train_params.algorithm = "rf"
    train_params.x_train = x_init
    train_params.y_train = y_init

    # Create num_iterations training session
    num_iterations = 20
    iter_dict = initialize_iterations_mode(num_iterations)
    assert bool(iter_dict) == True
    session_id = iter_dict['session_id']
    payment_request = iter_dict['payment_request']

    # Create a learner
    train_model(train_params,session_id,0)
    assert bool(models[session_id]) == True
    return session_id

def test_train():
    """
    Train on an initialized model for num_iterations.
    Note - this does not check if a session is valid.
    Just checks if the active learning is happening per plan.
    """
    x,y = test_create_dataset()
    session_id = test_init_model()
    session = get_session_info(session_id)
    num_iterations = session['num_iterations']
    for i in range(num_iterations):
        idx = random.randint(0,len(x))
        x_train = [",".join(map(str,x[idx]))]
        y_train = [str(y[idx])]
        # Create Train params
        train_params = TrainParams()
        train_params.algorithm = "gbc"
        train_params.x_train = x_train
        train_params.y_train = y_train
        train_model(train_params,session_id,i+1)
    session = get_session_info(session_id)
    assert session['completed_iterations']==num_iterations+1
    return session

def test_label():
    """
    Label an initialized, trained model.
    """
    x,y = test_create_dataset()
    # Initialize model
    session_id = test_init_model()
    session = get_session_info(session_id)
    num_iterations = session['num_iterations']
    # label model
    idx = random.randint(0,len(x))
    x_label = [",".join(map(str,x[idx]))]
    label_params = LabelParams()
    label_params.x_label = x_label
    # test unsuccessful labeling by setting completed_iterations = 0
    response = fetch_label(label_params,session_id,0)
    assert "Label unsuccessful" in response["message"]
    # test successful labeling by setting completed_iterations = 1
    response = fetch_label(label_params,session_id,1)
    assert "Label successful" in response["message"]
    session = get_session_info(session_id)
    assert session['completed_iterations']==2 # this needs to be fixed.
    return session

def test_al_loop():
    """
    Initialize, label (iter-by-iter), train (iter-by-iter) model
    """
    x,y = test_create_dataset()
    train_params = TrainParams()
    label_params = LabelParams()

    # Initialize model
    session_id = test_init_model()
    session = get_session_info(session_id)
    num_iterations = session['num_iterations']
    i=1
    while i<=num_iterations:
        # label model
        idx = random.randint(0,len(x))
        x_label = [",".join(map(str,x[idx]))]
        label_params.x_label = x_label
        label_response = fetch_label(label_params,session_id,i)
        assert "Label successful" in label_response["message"]
        i+=1
        if label_response["decision"] == "label":
            train_params.x_train = x_label
            train_params.y_train = [str(y[idx])]
            train_response = train_model(train_params,session_id,i)
            assert "Train successful" in train_response["message"]
            i+=1
        
            
                  


    








