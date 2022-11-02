"""
Tests the functionality of federated learning contract
"""
from web3 import Web3
from solcx import compile_source
from eth_tester.exceptions import TransactionFailed

with open("FL.sol", 'r') as f:
    solidity_code = f.read()

# Solidity source code
compiled_sol = compile_source(solidity_code, output_values=['abi', 'bin'])

# retrieve the contract interface
contract_id, contract_interface = compiled_sol.popitem()

# get bytecode / bin
bytecode = contract_interface['bin']

# get abi
abi = contract_interface['abi']

# web3.py instance
w3 = Web3(Web3.EthereumTesterProvider())

# set pre-funded account as sender
w3.eth.default_account = w3.eth.accounts[0]

FL = w3.eth.contract(abi=abi, bytecode=bytecode)

print("Creating the contract")
# Submit the transaction that deploys the contract
tx_hash = FL.constructor(b'genesis model').transact()

# Wait for the transaction to be mined, and get the transaction receipt
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

contract = w3.eth.contract(
    address=tx_receipt.contractAddress,
    abi=abi)

print("Contract created")

def printContractState():
    print("Contract state:")
    print("Epoch:", contract.functions.getEpoch().call())
    print("Data size:", contract.functions.getDataSize().call())
    print("Model:", contract.functions.getModel().call())
    print()
printContractState()

def passPreprocess():
    """
    Pass the preprocess stage of the contract.
    """
    tx_hash = contract.functions.globalMeans(b'').transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    tx_hash = contract.functions.globalStds(b'').transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print("Skipping preprocess stage")
passPreprocess()

print("Sending a local update")
tx_hash = contract.functions.localUpdate(0, 666, b'This is a local update').transact()
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print("Local update event logs:")
logs = contract.events.LocalUpdate().processReceipt(tx_receipt)
for log in logs:
    print(log["args"])
print()
printContractState()

print("Sending global update")
tx_hash = contract.functions.globalUpdate(b'This is a GLOBAL update').transact()
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
printContractState()

print("Sending a local update with incorrect epoch")
try:
    tx_hash = contract.functions.localUpdate(0, 666, b'Local update with incorrect epoch').transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("What? Transaction succeeded?")
except TransactionFailed as e:
    print("Transaction failed as expected.")
    print("Exception:", e)

print("\nSending global update with a different account")
try:
    w3.eth.default_account = w3.eth.accounts[1]
    tx_hash = contract.functions.globalUpdate(b'This is an ILLEGAL GLOBAL update').transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("What? Transaction succeeded?")
except TransactionFailed as e:
    print("Transaction failed as expected.")
    print("Exception:", e)
finally:
    w3.eth.default_account = w3.eth.accounts[0]


print("\nMultiple local updates")
receipts = [w3.eth.wait_for_transaction_receipt(contract.functions.localUpdate(1, 10, b'Local Model %d' % i).transact()) for i in range(4)]
for tx_receipt in receipts:
    logs = contract.events.LocalUpdate().processReceipt(tx_receipt)
    assert(len(logs) == 1)
    print(logs[0]["args"])

printContractState()

print("Sending global update")
tx_hash = contract.functions.globalUpdate(b'Another GLOBAL update').transact()
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
printContractState()

