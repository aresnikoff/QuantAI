import numpy as np
import random

from collections import deque, namedtuple
from models import MarketNN
from core.preprocess import preprocess


import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('Agent')
log.setLevel(logging.INFO)

Actions = namedtuple("actions", ["positions", "confidence", "type"])

class Agent(object):

  def __init__(self, param_dict):

    self.n_securities = param_dict["n_securities"]


    self.price_factors = param_dict["price_factors"]
    self.fund_factors = param_dict["fund_factors"]
    self.other_factors = param_dict["other_factors"]

    self.lr = param_dict["lr"]
    self.batch_size = param_dict["batch_size"]

    # discount rate
    self.gamma = param_dict["gamma"]

    # initialize memory
    self.memory = deque(maxlen = 2000)

    self.n_days = 10


    # intitalize network
    self.model = self.build_model(param_dict)

    self.param_dict = param_dict


  ## AGENT INTERFACE
  def remember(self, state, action, risk, returns):

    self.memory.append((state, action, risk, returns))


  def act(self, state):

    # reshape input to model graph

    state = self.reshape_input(state)

    # get model predictions on input
    predictions = self.model.predict(state)

    actions = self.reshape_output(predictions)

    return actions

  def replay(self):

    # get minibatch from memory
    minibatch = random.sample(self.memory, self.batch_size)

    for state, actions, risk, returns in minibatch:

      #state.sort_index(inplace = True)

      state_input = self.reshape_input(state)

      target = self.model.predict(state_input)

      stock_target = target

      decision_accuracy = {"correct": 0, "total": 0}

      positions = actions.positions
      confidence = actions.confidence
      _type = actions.type

      # run states through
      for i in xrange(self.n_securities):

        stock = state.iloc[i].name
        sec_id = stock.sid
        
        sec_returns = returns[sec_id]
        sec_risk = risk[sec_id]
        
        # this may change
        decision = positions[i]
        # find stock in next_day

        if decision > 0 and sec_returns > 0 or \
            decision < 0 and sec_returns < 0:
            decision_accuracy["correct"] += 1
        if decision > 0 or decision < 0:
            decision_accuracy["total"] += 1

        tolerance = 1.5
        reward = np.cos(sec_risk*.5*np.pi)*sec_returns

        if _type == "long":

          if reward > tolerance:

            stock_target[0][0][i*3 + 2] = 1 #np.reshape([0,0,reward],  [1,3])
            stock_target[i+1][0][0][2] = 1

          elif -tolerance <= reward <= tolerance:


            stock_target[0][0][i*3 + 1] = .1#np.reshape([0,1,0],  [1,3])
            stock_target[i+1][0][0][1] = .1

          else:
              
            stock_target[0][0][i*3] = 1# np.reshape([reward, 0,0], [1,3])
            stock_target[i+1][0][0][0] = 1

        else:

          if reward > tolerance:

            stock_target[i][0][0][2] = 1 #np.reshape([0,0,reward],  [1,3])

          elif -tolerance <= reward <= tolerance:

            stock_target[i][0][0][1] = .1#np.reshape([0,1,0],  [1,3])

          else:
              
            stock_target[i][0][0][0] = 1# np.reshape([reward, 0,0], [1,3])

          n_correct = decision_accuracy["correct"]
          total = decision_accuracy["total"]
          predicted_well = n_correct / float(total) > .5 if total > 0 else 0
          if predicted_well:

            stock_target[-1] = np.reshape([1,0], [1,2])
          else:
            stock_target[-1] = np.reshape([0,1], [1,2])

      target = stock_target

      self.model.fit(state_input, target, verbose = 0)
    log.info("Model updated")
              
  def save(self, name = "last"):

    path = "models/"

    # serialize model to json
    model_json = self.model.to_json()

    # create file and save the model
    with open(path + name + ".json", 'w') as json_file:

      # write the json file
      json_file.write(model_json)

      # save the model weights
      self.model.save_weights(path + name + ".h5")

      log.info("Saved model (" + name + ") to disk\n")

  def load(self, name):

    path = "models/"

    # load json and create model
    json_file = open(path + name + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path + name + ".h5")
    self.model = loaded_model
    self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    #self.epsilon = .75
    log.info("Loaded model (" + name + ") from disk")

  def reshape_input(self, state):

    n_days = self.n_days
    n_factors = self.price_factors
    n_securities = self.n_securities

    state = preprocess(state, self.param_dict)
    columns = state.columns.tolist()

    _input = []
    for i in xrange(n_days):

      frame_cols = columns[n_days - i - 1: n_factors - i]

      frame = state[frame_cols]
      _input.append(frame)

    _state = np.dstack(tuple(_input))

    _state = np.reshape(_state, [1, n_days, n_securities, n_factors - n_days + 1])

    # samples, timesteps, features

    _states = []
    for i in xrange(n_securities):

      stock = state.iloc[i]
      prices = np.reshape(stock.values, [1, 1, n_factors])
      _states.append(prices)    

    return [_state] + _states


    # n_days = self.n_days
    # n_factors = self.price_factors
    # n_securities = self.n_securities

    # columns = state.columns.tolist()
    # cnn_frames = []
    # for i in xrange(n_days):

    #   factors = columns[n_days - i:n_factors - i]
    #   #factors = columns[i:n_factors - n_days + i]
    #   cnn_frames.append(state[factors])

    # market_input = np.stack(cnn_frames, axis = 0)
    # market_input = np.reshape(market_input, [1, n_days, n_securities, n_factors - n_days])

    # security_input = []
    # for i in xrange(state.shape[0]):

    #   stock = state.iloc[i]
    #   prices = np.reshape(stock.values, [1, 1, state.shape[1]])
    #   security_input.append(prices)

    # return [state] + security_input
        

  def interpret(self, predictions):


    trades = []
    for i in xrange(len(predictions)):


      prediction = predictions[i][0]

      if not prediction.any():

        trades.append(0)
      else:
        guess = np.argmax(prediction) - 1
        trades.append(guess)

    return trades

  def interpret_long(self, predictions):

    trades = []
    for i in xrange(self.n_securities):

      nodes = predictions[i*3:i*3 + 3]

      if not nodes.any():

        trades.append(0)
      else:
        guess = np.argmax(nodes) - 1
        trades.append(guess)
    return trades

  def reshape_output(self, predictions):

    n_securities = self.n_securities

    # first is market output
    market_predictions = predictions[0]

    security_predictions = predictions[1:n_securities + 1]

    s_positions = self.interpret(security_predictions)

    if market_predictions.shape == (1, 3*n_securities):

      m_positions = self.interpret_long(market_predictions[0])

      positions = []
      for i in xrange(n_securities):

        m_pos = m_positions[i]
        s_pos = s_positions[i]

        positions.append(0 if m_pos != s_pos else m_pos)




      output = Actions(positions = positions,
                       confidence = 1,
                       type="long")
      return output

    n_sec = self.n_securities
    positions = self.interpret(predictions[:n_sec])
    #confidence = predictions[-1][0][0]

    output = Actions(positions = positions,
                     confidence = 1)#confidence)
    return output


  def _update_target(self, stock, risk, returns, action):

    pass


  def build_model(self, param_dict):

    log.info("building model...")
    NN = MarketNN(param_dict)
    return NN.build_model()






