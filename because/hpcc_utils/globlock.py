"""
Implement a quasi-atomic global lock.  This is used for synchronizing between
HPCC channels on the same node.  It provides a common mechanism for
implementing a global lock.  This is typically only used during initialization.
While there is no way to ensure 100% atomicity, this is very unlikely to fail.
This logic is centralized so that it can be changed in one place if a better
mechanism is found.
"""
import threading
import time


"""
Allocate the global lock.
"""
def allocate():
    global GLOBLOCK
    # There may be a context switch between checking for GLOBLOCK and
    # setting it.  We allow that contention, and let the last thread in
    # define the lock.
    if 'GLOBLOCK' not in globals():
        GLOBLOCK = threading.Lock()
    # We sleep for a short interval to make sure (with high probability)
    # that anyone who fell through the test has allocated a lock.
    time.sleep(.00001)
    # At this point, GLOBLOCK should be the lock allocated by the last
    # thread to fall through the above test.  All threads should now
    # be contending over the same (truly) atomic lock when they call
    # acquire.
"""
Acquire the global lock
"""
def acquire():
    GLOBLOCK.acquire()

"""
Release the global lock
"""
def release():
    GLOBLOCK.release()
