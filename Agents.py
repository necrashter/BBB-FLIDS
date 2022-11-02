"""
This module contains classes that represent various agents in the federated
learning system.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from FederatedModel import *
from config import *

from log import log

class FL:
    LossFunc = None
    Xdtype = INTERNAL_DTYPE.numpy
    # will be set by driver function
    Xfeatures = None

    class User:
        """
        A base class representing all users of the federated learning system.
        """
        def __init__(self, account, model):
            self.account = account
            self.model = model


    class Client(User):
        """
        Each FL Client is responsible for the following functions in blockchain:
        1. Keep track of the latest model in blockchain.
        2. Backpropagate this global model with the private dataset.
        3. Report the updates back to the blockchain.
        """
        # Number of total clients
        count = 0
        def __init__(self, account, model, dataloader):
            """
            Server must be deployed and contractInfo set before any client initialization.
            """
            super(FL.Client, self).__init__(account, model)
            self.dataloader = dataloader
            account.obtainContract()
            self.index = FL.Client.count
            FL.Client.count += 1

        def localMeans(self):
            """
            Report the local mean values to blockchain.
            Returns a transaction receipt.
            """
            X = torch.cat([x for x, y in self.dataloader])
            size = X.shape[0]
            mean = X.mean(0, keepdim=True)
            # Commit to blockchain
            tx_receipt = self.account.localMeans(size, mean.numpy().tobytes())
            return tx_receipt

        def getMeans(self):
            """
            Get the means values from the blockchain.
            """
            # NOTE: using Tensor instead of tensor gives a warning about non-writable
            self.means = torch.tensor(np.frombuffer(self.account.getMeans(), dtype=FL.Xdtype).reshape((1, FL.Xfeatures)))

        def localStds(self):
            """
            Report the local std values to blockchain.
            Returns a transaction receipt.
            """
            X = torch.cat([x for x, y in self.dataloader])
            size = X.shape[0]
            stds = (X - self.means).square().mean(0, keepdim=True)
            # Commit to blockchain
            tx_receipt = self.account.localStds(size, stds.numpy().tobytes())
            return tx_receipt

        def getStds(self):
            """
            Get the std values from the blockchain.
            """
            self.stds = torch.tensor(np.frombuffer(self.account.getStds(), dtype=FL.Xdtype).reshape((1, FL.Xfeatures)))
            # Handle 0 stds to avoid division by zero
            self.stds[self.stds == 0.0] = 1.0

        def localUpdate(self):
            """
            Perform a local update and trigger an event in blockchain.
            Returns a transaction receipt.
            """
            # Load the latest model from blockchain
            epoch = self.account.getEpoch()
            modelBytes = self.account.getModel()
            self.model.from_bytes(modelBytes)

            datasize = 0
            for X, y in self.dataloader:
                datasize += X.shape[0]

            train_loss = 0.0

            global LOCAL_EPOCHS, LEARNING_RATE, MOMENTUM
            optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
            for i in range(LOCAL_EPOCHS):
                for batch, (X, y) in enumerate(self.dataloader):
                    normX = (X - self.means) / self.stds
                    pred = self.model(normX)
                    loss = FL.LossFunc(pred, y)
                    train_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # len(dataloader) == number of batches
            loss = train_loss / (len(self.dataloader) * LOCAL_EPOCHS)
            log.info(f"FL Client {self.index} local loss: {loss}")

            # Commit to blockchain
            tx_receipt = self.account.localUpdate(epoch, datasize, self.model.to_bytes())

            return tx_receipt


    class Server(User):
        """
        Federated learning server is the user who performs the model averaging step.
        """
        def __init__(self, account, model):
            """
            Initialize and deploy a server.
            """
            super(FL.Server, self).__init__(account, model)
            account.deploy(model.to_bytes())

        def combineMeans(self, receipts):
            """
            Collect local mean report events from the given list of receipts and set global means.
            Returns the transaction receipt.
            """
            means = combine_means([
                (n, np.frombuffer(byteMeans, dtype=FL.Xdtype).reshape((1, FL.Xfeatures)))
                for n, byteMeans in self.account.getMeanEvents(receipts)
                ]).astype(FL.Xdtype)
            tx_receipt = self.account.globalMeans(means.tobytes())
            return tx_receipt

        def combineStds(self, receipts):
            """
            Collect local std report events from the given list of receipts and set global stds.
            Returns the transaction receipt.
            """
            stds = combine_stds([
                (n, np.frombuffer(byteMeans, dtype=FL.Xdtype).reshape((1, FL.Xfeatures)))
                for n, byteMeans in self.account.getStdEvents(receipts)
                ]).astype(FL.Xdtype)
            tx_receipt = self.account.globalStds(stds.tobytes())
            return tx_receipt

        def skipPreprocess(self):
            """
            Skips the preprocessing stage by setting the means to 0 and the stds to 1.
            """
            means = np.zeros((1, FL.Xfeatures), dtype=FL.Xdtype)
            stds = np.ones((1, FL.Xfeatures), dtype=FL.Xdtype)
            self.account.globalMeans(means.tobytes())
            self.account.globalStds(stds.tobytes())

        def averageUpdates(self, receipts):
            """
            Collect local update events from the given list of receipts and process them.
            Returns the transaction receipt.
            """
            epoch = self.account.getEpoch()
            totalDataSize = self.account.getDataSize()
            log.info(f"Averaging model from {len(receipts)} local update(s)...")
            self.model.zero()
            for size, modelBytes in self.account.getUpdateEvents(receipts):
                # Weight of each update should be proportional to the dataset size
                weight = size / totalDataSize
                self.model.federate_from_bytes(modelBytes, weight)
            # Model is now ready
            # Update model on blockchain
            tx_receipt = self.account.globalUpdate(self.model.to_bytes())
            log.info(f"Epoch {epoch} finished and committed to blockchain.")

            return tx_receipt

        def getModel(self):
            modelBytes = self.account.getModel()
            self.model.from_bytes(modelBytes)
            return self.model

