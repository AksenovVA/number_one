import numpy as np


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """
    x = np.diag(x)
    k = np.prod(x != 0)
    return k
    
    

def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """
    x = np.sort(x, kind='heapsort')
    y = np.sort(y, kind='heapsort')
    return np.array_equal(x, y)

def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """
    y = x[:-1]
    i = np.where(y == 0)[0] + 1
    return np.max(x[i])

    


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """

    return np.dot(img, coefs)


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """

    xx = np.where(np.diff(x) != 0)[0] + 1
    ans1 = x[xx - 1]
    ans2 = np.diff(np.concatenate(([0], xx, [len(x)])))

    return (ans1, ans2)


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """

    ans = np.sqrt(np.sum((x - y) ** 2, axis=-1))
    return ans
