from time import time
  
def timefunc(printFunc):
    def decorator(func):
        def f(*args, **kwargs):
            startTime = time()
            result = func(*args, **kwargs)
            passedTime = time() - startTime
            printFunc(f'Function {func.__name__!r} finished execution in {passedTime:.4f}s')
            return result
        return f
    return decorator

