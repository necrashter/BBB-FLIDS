pragma solidity >=0.8.15;

contract FederatedLearningContract {
	// Owner address of this contract
	// Owner will act as the "server" for federated learning
	// Others will send their updates via events, the ownder will average them to update the model
	address public owner;

	// Enum representing current stage of the federated learning model
	enum Stage{ PREPROCESS_MEANS, PREPROCESS_STDS, TRAINING }
	Stage public stage;

	// Current global epoch.
	uint public epoch;
	// Total size of the submitted data within the current epoch.
	uint public dataSize;

	// Representation of the model in bytes.
	bytes public model;
	// For data standardization
	bytes public means;
	bytes public stds;

	// This event is fired when a client reports a local update at given epoch.
	event LocalUpdate(address indexed from, uint epoch, uint size, bytes model);
	// These events are fired during the preprocessing stage by the clients.
	event LocalMeans(address indexed from, uint size, bytes data);
	event LocalStds(address indexed from, uint size, bytes data);

	constructor(bytes memory initialModel) public {
		// Owner is the one who ran the constructor method
		owner = msg.sender;
		model = initialModel;
		epoch = 0;
		dataSize = 0;
		stage = Stage.PREPROCESS_MEANS;
	}

	// With this modifier, only owner can call the given function.
	modifier OwnerOnly() {
		require(msg.sender == owner, "Only the contract owner can call this function!");
		_;
	}

	// When a client runs gradient descent and computes the local model update,
	// it runs this function to commit this update to blockchain.
	function localUpdate(uint localEpoch, uint size, bytes memory localModel) public {
		require(stage == Stage.TRAINING, "Can only be called in training stage!");
		require(localEpoch == epoch, "Local update epoch is not the same as global epoch!");
		dataSize += size;
		emit LocalUpdate(msg.sender, localEpoch, size, localModel);
	}

	// Individual client reports of means.
	// Similar to localUpdate in terms of operation.
	function localMeans(uint size, bytes memory data) public {
		require(stage == Stage.PREPROCESS_MEANS, "Can only be called in means preprocessing stage!");
		emit LocalMeans(msg.sender, size, data);
	}

	// Individual client reports of stds.
	// Similar to localUpdate in terms of operation.
	function localStds(uint size, bytes memory data) public {
		require(stage == Stage.PREPROCESS_STDS, "Can only be called in std preprocessing stage!");
		emit LocalStds(msg.sender, size, data);
	}

	// After the weight updates are averaged, this function is called
	// to update the model on blockchain.
	// Can be called by owner-only, since the actual model updates are limited to the owner.
	// Note that this function being public is irrelevant.
	// In Solidity, private access type only prevents other contracts from calling this function.
	function globalUpdate(bytes memory updatedModel) public OwnerOnly {
		require(stage == Stage.TRAINING, "Can only be called in training stage!");
		model = updatedModel;
		// Advance epoch
		epoch += 1;
		// Reset the submitted data size
		dataSize = 0;
	}

	// Used by the owner to submit global means and advance the stage to std preprocessing.
	function globalMeans(bytes memory data) public OwnerOnly {
		require(stage == Stage.PREPROCESS_MEANS, "Can only be called in means preprocessing stage!");
		means = data;
		// Advance stage
		stage = Stage.PREPROCESS_STDS;
	}

	// Used by the owner to submit global stds and advance the stage to training.
	function globalStds(bytes memory data) public OwnerOnly {
		require(stage == Stage.PREPROCESS_STDS, "Can only be called in std preprocessing stage!");
		stds = data;
		// Advance stage
		stage = Stage.TRAINING;
	}

	// Public getter methods follow.

	function getModel() view public returns(bytes memory) {
		return model;
	}

	function getMeans() view public returns(bytes memory) {
		return means;
	}

	function getStds() view public returns(bytes memory) {
		return stds;
	}

	function getEpoch() view public returns(uint) {
		return epoch;
	}

	function getDataSize() view public returns(uint) {
		return dataSize;
	}
}
