import logging

# FLIDS: Federated Learning Intrusion Detection System
log = logging.getLogger("BBB-FLIDS")
log.setLevel(logging.INFO)
try:
    from tqdm import tqdm
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)
    log.addHandler(TqdmLoggingHandler())
except ModuleNotFoundError:
    log.warning("Warning: tqdm is not found! Install it for progress bar.")
    tqdm = lambda a: a

