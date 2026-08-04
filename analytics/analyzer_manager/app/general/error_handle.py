import traceback
import sys

from general.analyzer_general import log_exception


def try_except_raise(logger, raise_exception=True, return_on_exception=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                tb_str = ''.join(traceback.format_tb(exc_tb))
                print(f"An error occurred in {func.__name__}: {str(e)}\n{tb_str}")
                log_exception(logger, f"An error occurred in {func.__name__}", e)
                # logger.error(f"An error occurred in {func.__name__}: {str(e)}\n{tb_str}")
                if raise_exception:
                    raise e
                else:
                    return return_on_exception
        return wrapper
    return decorator