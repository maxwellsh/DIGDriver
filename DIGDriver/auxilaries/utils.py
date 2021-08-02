import multiprocessing as mp

def get_cpus():
    try:
        c = min(max(1, mp.cpu_count() - 2), 20)
    except:
        c = 5
    return c
