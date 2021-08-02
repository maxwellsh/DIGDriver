import multiprocessing as mp

def get_cpus():
    try:
        c = max(1, mp.get_cpus() - 2)
    except:
        c = 5
    return c
