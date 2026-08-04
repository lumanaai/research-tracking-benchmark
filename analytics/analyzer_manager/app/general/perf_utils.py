import time

import psutil


def time_sync():
    # pytorch-accurate time (for single thread only)
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    return time.time()


def get_process_id():
    return psutil.Process()


def memory_usage(pid=None):

    memory_info = pid.memory_info()
    return memory_info.rss


def mem_bytes_2_human(n, p_format="%(value).3f%(symbol)s"):
    """Used by various scripts. See:
    http://goo.gl/zeJZl

    >>> bytes2human(10000)
    '9.8K'
    >>> bytes2human(100001221)
    '95.4M'
    """
    symbols = ("B", "K", "M", "G", "T", "P", "E", "Z", "Y")
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return p_format % locals()
    return p_format % dict(symbol=symbols[0], value=n)
