"""
This utility checks the gas fees associated with global and local model updates.
"""
from web3 import Web3
from solcx import compile_source
from eth_tester.exceptions import TransactionFailed
from eth.exceptions import OutOfGas

dataSize = 10000
dataNum = 0
def getBinaryData():
    global dataNum
    binaryData = (str(dataNum).encode() + b' ') * dataSize
    binaryData = binaryData[:dataSize]
    assert(len(binaryData) == dataSize)
    dataNum += 1
    return binaryData

with open("FL.sol", 'r') as f:
    solidity_code = f.read()
compiled_sol = compile_source(solidity_code, output_values=['abi', 'bin'])
contract_id, contract_interface = compiled_sol.popitem()
bytecode = contract_interface['bin']
abi = contract_interface['abi']

# web3.py instance
w3 = Web3(Web3.EthereumTesterProvider())

# set pre-funded account as sender
my_account = w3.eth.accounts[1]
w3.eth.default_account = my_account
print("Initial balance before all operations:", w3.eth.get_balance(my_account))

FL = w3.eth.contract(abi=abi, bytecode=bytecode)

print("Creating the contract")
# Submit the transaction that deploys the contract
tx_hash = FL.constructor(b'').transact()

# Wait for the transaction to be mined, and get the transaction receipt
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

contract = w3.eth.contract(
    address=tx_receipt.contractAddress,
    abi=abi)

print("Contract created")

def printContractState():
    print("Contract state:")
    print("\tEpoch:", contract.functions.getEpoch().call())
    print("\tData size:", contract.functions.getDataSize().call())
    print("\tModel:", contract.functions.getModel().call()[:100], "...")
    print()
printContractState()

def printEventLog(log):
    print("Event log:")
    print("\tEpoch:", log["args"]["epoch"])
    print("\tData size:", log["args"]["size"])
    print("\tModel:", log["args"]["model"][:100], "...")

def passPreprocess():
    """
    Pass the preprocess stage of the contract.
    """
    tx_hash = contract.functions.globalMeans(b'').transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    tx_hash = contract.functions.globalStds(b'').transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

def stressTest():
    starting_balance = w3.eth.get_balance(my_account)
    print("Starting balance:", starting_balance)

    print("Sending global update")
    tx_hash = contract.functions.globalUpdate(getBinaryData()).transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("Gas used:", tx_receipt["gasUsed"])
    print("diff:", starting_balance - w3.eth.get_balance(my_account))
    printContractState()

    print("\nMultiple local updates")
    receipts = [w3.eth.wait_for_transaction_receipt(contract.functions.localUpdate(1, 10, getBinaryData()).transact()) for i in range(4)]
    for tx_receipt in receipts:
        print("Gas used:", tx_receipt["gasUsed"])
        logs = contract.events.LocalUpdate().processReceipt(tx_receipt)
        assert(len(logs) == 1)
        printEventLog(logs[0])

    print("Sending global update")
    tx_hash = contract.functions.globalUpdate(getBinaryData()).transact()
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("Gas used:", tx_receipt["gasUsed"])
    printContractState()

    ending_balance = w3.eth.get_balance(my_account)
    print("Final balance:", ending_balance)
    print("Difference:", starting_balance-ending_balance)

def estimateGasFees():
    # Estimated gas fees are pretty accurate
    print("Estimated gas fees:")
    globalGas = contract.functions.globalUpdate(getBinaryData()).estimateGas()
    localGas = contract.functions.localUpdate(0, 666, getBinaryData()).estimateGas()
    print("\tGlobal update:", globalGas)
    print("\tLocal update:", localGas)
    overall = 10*localGas + globalGas
    print("\t10 local + 1 global:", overall)
    return overall


passPreprocess()
# stressTest()

if True:
    stressTest()
else:
    dataSize = 1
    print("For model size = %d" %(dataSize))
    baseGas = estimateGasFees()
    print()

    for dataSize in [10, 100, 1000, 10000, 20000, 30000, 40000, 50000]:
        print("For model size = %d" %(dataSize))
        try:
            gas = estimateGasFees()
            print("\tRatio to base gas:", gas/baseGas)
        except OutOfGas:
            print("\tOut of gas!")
            break
