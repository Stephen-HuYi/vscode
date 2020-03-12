import math


def closest(X, n):
    X = sorted(X)
    Y = sorted(X, key=lambda last: last[-1])
    return closest_pair(X, Y, n)


def closest_pair(X, Y, n):
    if n <= 3:
        return brute_force(X, n)
    mid = n / 2
    Y_left = sorted(X[:mid], key=lambda last: last[-1])
    Y_right = sorted(X[mid:], key=lambda last: last[-1])
    dis_left = closest_pair(X[:mid], Y_left, mid)
    dis_right = closest_pair(X[mid:], Y_right, n-mid)
    min_dis = min(dis_left, dis_right)
    strip = []
    for (x, y) in Y:
        if abs(x-X[mid][0]) < min_dis:
            strip.append((x, y))
    return min(min_dis, strip_closest(strip, min_dis))


def brute_force(X, n):
    min_d = distance(X[0], X[1])
    for i, (x, y) in enumerate(X):
        for j in range(i+1, n):
            if distance(X[i], X[j]) < min_d:
                min_d = distance(X[i], X[j])
    return min_d


def distance(a, b):
    return math.sqrt(math.pow((a[0]-b[0]), 2)+math.pow((a[1]-b[1]), 2))


def strip_closest(strip, d):
    min_d = d
    for i, (x, y) in enumerate(strip):
        for j in range(i+1, 6):
            if i + j < len(strip):
                temdis = distance(strip[i], strip[j])
                if temdis < min_d:
                    min_d = temdis
    return min_d


if __name__ == "__main__":
    points = [(2, 3), (10, 1), (3, 25), (23, 15),
              (18, 3), (8, 9), (12, 30), (25, 30),
              (9, 2), (13, 10), (3, 4), (5, 6),
              (22, 32), (5, 32), (23, 9), (19, 25),
              (14, 1), (11, 25), (26, 26), (12, 9),
              (18, 9), (27, 13), (32, 13)]
    print(closest(points, len(points)))
