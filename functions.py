import math


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    
    k = 1
    for i in range(min(len(x), len(x[0]))):
        if x[i][i] != 0:
            k *= x[i][i]
    return k
   


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    x = sorted(x)
    y = sorted(y)
    return x == y

def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    mx = -1
    for i in range(len(x) - 1):
        if x[i] == 0:
            mx = max(mx, x[i + 1])
    return mx
    


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """

    height, width, num_channels = img.shape
    
    # Создаем матрицу для оттенков серого
    f = np.zeros((height, width))
    
    # Проходим по каждому пикселю изображения
    for i in range(height):
        for j in range(width):
            # Считаем взвешенную сумму значений каналов
            f[i, j] = sum(img[i, j, c] * coef[c] for c in range(num_channels))
    
    return f


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    ans1 = [x[0]]
    ans2 = [1]
    
    for i in range(1, len(x)):
        if ans1[-1] == x[i]:
            ans2[-1] += 1
        else:
            ans1.append(x[i])
            ans2.append(1)
    return (ans1, ans2)


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """
    ans = 0
    for i in range(len(x)):
        for j in range(len(x[0])):
            ans += (x[i][j] - y[i][j]) ** 2

    return math.sqrt(ans)