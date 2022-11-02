"""
This module contains Ethereum Testing platform for Federated Learning.
"""
from web3 import Web3
from solcx import compile_source
from dataclasses import dataclass


@dataclass
class ContractInfo:
    """
    Represents contract information for clients.
    """
    contract_id: str
    # Application binary interface
    abi: list
    address: str


def compileContract(w3, filename, *vargs):
    """
    Given web3.py instance and .sol filename, create a contract with the default account.
    The remaining arguments are passed to contract's constructor.
    """
    with open("FL.sol", 'r') as f:
        solidity_code = f.read()

    compiled_sol = compile_source(solidity_code, output_values=['abi', 'bin'])

    # retrieve the contract interface
    contract_id, contract_interface = compiled_sol.popitem()

    bytecode = contract_interface['bin']
    abi = contract_interface['abi']

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    # Submit the transaction that deploys the contract
    tx_hash = contract.constructor(*vargs).transact()
    # Wait for the transaction to be mined, and get the transaction receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    # Get the contract address from the receipt
    address = tx_receipt.contractAddress

    return ContractInfo(contract_id, abi, address)


def useAccount(func):
    """
    A decorator for functions that need to use the account set in self.account
    """
    def f(self, *vargs, **kwargs):
        EthPlatform.w3.eth.default_account = self.account
        return func(self, *vargs, **kwargs)
    return f

class EthPlatform:
    contractFilename = "FL.sol"
    contractInfo = None
    w3 = None

    @staticmethod
    def initAccounts(amount: int):
        EthPlatform.w3 = Web3(Web3.EthereumTesterProvider())

        amount = min(amount, len(EthPlatform.w3.eth.accounts))
        users = []
        for i, account in zip(range(amount), EthPlatform.w3.eth.accounts):
            users.append(EthPlatform.Account(account))
        return users

    class Account:
        """
        Wraps accounts with helper functions and some additional data.
        """
        def __init__(self, account):
            self.account = account

        @useAccount
        def deploy(self, *vargs):
            """
            Deploys the contract with this account and obtain a reference to it.
            """
            EthPlatform.contractInfo = compileContract(
                    EthPlatform.w3, EthPlatform.contractFilename, *vargs)
            self.obtainContract()

        def obtainContract(self):
            """
            After the contract has been deployed by one user, the other users will call
            this function to obtain a reference to it in self.contract.
            """
            self.contract = EthPlatform.w3.eth.contract(
                    address=EthPlatform.contractInfo.address,
                    abi=EthPlatform.contractInfo.abi)

        @useAccount
        def getUpdateEvents(self, receipts):
            """
            From a list of receipts get the processed update events.
            """
            events = []
            seenAddresses = set()
            epoch = self.getEpoch()
            for tx_receipt in receipts:
                logs = self.contract.events.LocalUpdate().processReceipt(tx_receipt)
                assert(len(logs) == 1)
                args = logs[0]["args"]
                address = args["from"]
                if address in seenAddresses:
                    log.warning(f"Ignoring repeated update from address {address}")
                    continue
                seenAddresses.add(address)
                updateEpoch = epoch
                if epoch != updateEpoch:
                    log.warning(f"Ignoring update with incorrect epoch {updateEpoch} from {address}")
                    continue
                size = args["size"]
                modelBytes = args["model"]
                events.append((size, modelBytes))
            return events

        @useAccount
        def getMeanEvents(self, receipts):
            """
            From a list of receipts get the processed mean events.
            """
            events = []
            seenAddresses = set()
            for tx_receipt in receipts:
                logs = self.contract.events.LocalMeans().processReceipt(tx_receipt)
                assert(len(logs) == 1)
                args = logs[0]["args"]
                address = args["from"]
                if address in seenAddresses:
                    log.warning(f"Ignoring repeated mean report from address {address}")
                    continue
                seenAddresses.add(address)
                size = args["size"]
                means = args["data"]
                events.append((size, means))
            return events

        @useAccount
        def getStdEvents(self, receipts):
            """
            From a list of receipts get the processed std events.
            """
            events = []
            seenAddresses = set()
            for tx_receipt in receipts:
                logs = self.contract.events.LocalStds().processReceipt(tx_receipt)
                assert(len(logs) == 1)
                args = logs[0]["args"]
                address = args["from"]
                if address in seenAddresses:
                    log.warning(f"Ignoring repeated mean report from address {address}")
                    continue
                seenAddresses.add(address)
                size = args["size"]
                means = args["data"]
                events.append((size, means))
            return events

        @useAccount
        def globalUpdate(self, modelBytes):
            """
            Update the global model after weight averaging.
            Should be called by owner only.
            """
            tx_hash = self.contract.functions.globalUpdate(modelBytes).transact()
            tx_receipt = EthPlatform.w3.eth.wait_for_transaction_receipt(tx_hash)
            return tx_receipt

        @useAccount
        def localUpdate(self, *vargs):
            """
            Trigger a local update event.
            """
            tx_hash = self.contract.functions.localUpdate(*vargs).transact()
            tx_receipt = EthPlatform.w3.eth.wait_for_transaction_receipt(tx_hash)
            return tx_receipt

        @useAccount
        def globalMeans(self, meanBytes):
            """
            Update the global means after mean averaging.
            Should be called by owner only.
            """
            tx_hash = self.contract.functions.globalMeans(meanBytes).transact()
            tx_receipt = EthPlatform.w3.eth.wait_for_transaction_receipt(tx_hash)
            return tx_receipt

        @useAccount
        def localMeans(self, *vargs):
            """
            Trigger a local means event.
            """
            tx_hash = self.contract.functions.localMeans(*vargs).transact()
            tx_receipt = EthPlatform.w3.eth.wait_for_transaction_receipt(tx_hash)
            return tx_receipt

        @useAccount
        def globalStds(self, stdBytes):
            """
            Update the global stds after std averaging.
            Should be called by owner only.
            """
            tx_hash = self.contract.functions.globalStds(stdBytes).transact()
            tx_receipt = EthPlatform.w3.eth.wait_for_transaction_receipt(tx_hash)
            return tx_receipt

        @useAccount
        def localStds(self, *vargs):
            """
            Trigger a local stds event.
            """
            tx_hash = self.contract.functions.localStds(*vargs).transact()
            tx_receipt = EthPlatform.w3.eth.wait_for_transaction_receipt(tx_hash)
            return tx_receipt

        # The following public accessor functions don't need to use account
        def getModel(self):
            return self.contract.functions.getModel().call()

        def getEpoch(self):
            return self.contract.functions.getEpoch().call()

        def getDataSize(self):
            return self.contract.functions.getDataSize().call()

        def getMeans(self):
            return self.contract.functions.getMeans().call()

        def getStds(self):
            return self.contract.functions.getStds().call()
