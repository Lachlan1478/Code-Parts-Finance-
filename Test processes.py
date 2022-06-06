import time
import concurrent
from concurrent import futures


start = time.perf_counter()

Scores = []

def sleep(secs):
    print('start sleep')
    time.sleep(secs)
    secs = secs * secs
    print('Done sleeping')

    Scores.append(secs)

processes = []
if __name__ == '__main__':

    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [1,2,3,4,5]
        for sec in secs:
            results = executor.map(sleep(sec))

print(Scores)
finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')
    
