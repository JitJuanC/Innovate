"""Sort Algorithms

A collection of sorting algorithms written from scratch. For practice and to find out the time complexity for each one.

There is a Time Complexity associated with each algorithm (function).

Call a function below and input an array (the argument) to start!
"""
# Swap function (Will be called in the sorting algorithms)
def swap(array_, first, second):
    temp = array_[first]
    array_[first] = array_[second]
    array_[second] = temp
    return array_

# Bubble Sort O(n**2)
def bubble_sort(array_):
    total = len(array_)
    for i in range(total):
        for j in range(total - 1 - i):
            if array_[j] > array_[j + 1]:
                swap(array_, j, j + 1)
    return array_

# Selection Sort O(n**2)
def selection_sort(array_):
    total = len(array_)
    for i in range(total):
        minimum = i
        for j in range(total - i):
            if array_[i + j] < array_[minimum]:
                minimum = i + j
        swap(array_, minimum, i)
    return array_

# Insertion Sort O(n**2)
def insertion_sort(array_):
    total = len(array_)
    for i in range(total - 1):
        current = array_[i + 1]
        j = i
        while j >= 0 and current < array_[j]:
            array_[j + 1] = array_[j]
            j -= 1
        j += 1
        array_[j] = current
    return array_

# Merge sort O(n log n)
def merge_sort(array_):
    if len(array_) < 2:
        return array_

    half_idx = len(array_) // 2
    first_half = array_[:half_idx]
    second_half = array_[half_idx:]

    first = merge_sort(first_half)
    second = merge_sort(second_half)

    left_idx = 0
    right_idx = 0
    combined = len(first) + len(second) # total

    for i in range(combined):
        if len(first) == left_idx:
            array_[i] = second[right_idx]
            right_idx += 1
        elif len(second) == right_idx:
            array_[i] = first[left_idx]
            left_idx += 1
        else:
            if first[left_idx] < second[right_idx]:
                array_[i] = first[left_idx]
                left_idx += 1
            else:
                array_[i] = second[right_idx]
                right_idx += 1
    return array_

if __name__ == '__main__':
    # array_ = [3, 2, 1, 4, 5, 6]
    array_ = [8, 2, 4, 1, 3]

    print(merge_sort(array_))