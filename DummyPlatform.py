"""
Dummy federated learning blockchain backend for faster testing.
"""

class DummyPlatform:
    modelBytes = None
    epoch = 0
    dataSize = 0
    means = None
    stds = None

    @staticmethod
    def initAccounts(amount: int):
        return [DummyPlatform.Account() for i in range(amount)]

    class Account:
        """
        Wraps accounts with helper functions and some additional data.
        """
        def deploy(self, modelBytes):
            """
            Deploys the contract with this account and obtain a reference to it.
            """
            DummyPlatform.modelBytes = modelBytes

        def obtainContract(self):
            """
            After the contract has been deployed by one user, the other users will call
            this function to obtain a reference to it in self.contract.
            """
            pass

        def getUpdateEvents(self, receipts):
            """
            From a list of receipts get the processed events.
            """
            return receipts

        def getMeanEvents(self, receipts):
            """
            From a list of receipts get the processed mean events.
            """
            return receipts

        def getStdEvents(self, receipts):
            """
            From a list of receipts get the processed std events.
            """
            return receipts

        def globalUpdate(self, modelBytes):
            """
            Update the global model after weight averaging.
            Should be called by owner only.
            """
            DummyPlatform.modelBytes = modelBytes
            DummyPlatform.epoch += 1
            DummyPlatform.dataSize = 0
            return None

        def localUpdate(self, *vargs):
            """
            Trigger a local update event.
            """
            DummyPlatform.dataSize += vargs[1]
            return (vargs[1], vargs[2])

        def globalMeans(self, means):
            """
            Update the global means.
            Should be called by owner only.
            """
            DummyPlatform.means = means
            return None

        def localMeans(self, *vargs):
            """
            Trigger a local means event.
            """
            return (vargs[0], vargs[1])

        def globalStds(self, stds):
            """
            Update the global stds.
            Should be called by owner only.
            """
            DummyPlatform.stds = stds
            return None

        def localStds(self, *vargs):
            """
            Trigger a local stds event.
            """
            return (vargs[0], vargs[1])

        # The following public accessor functions don't need to use account
        def getModel(self):
            return DummyPlatform.modelBytes

        def getEpoch(self):
            return DummyPlatform.epoch

        def getDataSize(self):
            return DummyPlatform.dataSize

        def getMeans(self):
            return DummyPlatform.means

        def getStds(self):
            return DummyPlatform.stds
