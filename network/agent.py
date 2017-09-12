import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('Agent')
log.setLevel(logging.INFO)

class Agent(object):

	def __init__(self, n_securities, batch_size, lr):

		pass




	def remember(self, state, action, _state, risk, returns):

		self.memory.append((state, action, _state, risk, returns))

	def act(self, state):

		# reshape input to model graph
		state = self.reshape_input(state)

		# get model predictions on input
		predictions = self.model.predict(state)

		# interpret the predictions as actions to take
		actions = self.interpret(predictions)

		return actions

	def replay(self):

		# get minibatch from memory
		minibatch = random.sample(self.memory, self.batch_size)

		for state, action, _state, risk, returns in minibatch:

			state_input = self.reshape_input(state)
			_state_input = self.reshape_input(_state)

			target = self.model.predict(state_input)
			_target = self.model.predict(_state_input)


	def update_target(self, stock, risk, returns, action):

		pass


	def _build_model(self, code = "default"):

		pass

	def save(self, name):

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






