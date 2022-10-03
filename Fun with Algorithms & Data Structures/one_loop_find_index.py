"""
List, Indexing

Finding index of the 2 numbers which sums up to a particular number within list of numbers in O(n), linear time complexity
The list has to be sorted in order to make it possible with one loop.

Input: list with numbers (Should be sorted or there is a method below to sort it as well; take note, sort() is O(n log n))

Output: Index of the 2 numbers(0:) - eg. "first index: 2 | second index: 5"
Output(pair not found): -1, -1 - eg. "first index: -1 | second index: -1"
"""
# list must be sorted
def find_index_one_loop_sorted(num_list, target):
    front = 0
    back = len(num_list) - 1
    while True:
        answer = num_list[front] + num_list[back]
        # if index are the same, return -1 for both indexes (not found)
        if front == back:
            return -1, -1
        # if answer is target, return indexes in list
        if answer == target:
            return front, back
        elif answer > target:
            back -= 1
        elif answer < target:
            front += 1
    
if __name__ == '__main__':
    # Target Number to find
    target = 15

    # sorted
    input = [3, 5, 6, 9, 13, 20]
    # unsorted
    # input = [3, 9, 20, 5, 13, 6]
    
    # for unsorted uncomment all below
    # unsorted = input.copy()
    # input.sort()

    first, second = find_index_one_loop_sorted(input, target)
    
    # if not sorted will require this method below to find the true index
    # first, second = unsorted.index(input[first]), unsorted.index(input[second])

    print(f"first index: {first} | second index: {second}")
