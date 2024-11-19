---
title: 'Common Algorithmic Patterns'
description: Certain patterns repeat ad nauseum; these are good candidates for practice and memorization.
tags:
  - algos
  - python
date: 2023-12-20
---
## Search

```python
def binary_search(nums, target):
    low, high = 0, len(nums) - 1

    while low <= high:
        mid = (low + high) // 2

        if nums[mid] == target:
            return (True, mid)
        elif nums[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return (False, low)
```

```python
def bisect(a, target, left, right, compare):
    while left < right:
        mid = (left + right) // 2
        if compare(a[mid], target):
            left = mid + 1
        else:
            right = mid
    return left

a = [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 10]

start = bisect(a, 4, 0, len(a) - 1, lambda x, y: x < y)  # 4
end = bisect(a, 4, 0, len(a) - 1, lambda x, y: x <= y)  # 10 (not 9)
# The 4s are located at indices range(4,10).
```

```python
def find_rotation_point(arr):
    left, right = 0, len(arr) - 1
    
    # If the array is not rotated (smallest element is at the start)
    if arr[left] < arr[right]:
        return 0
    
    while left <= right:
        mid = left + (right - left) // 2
        
        # Check if mid is the rotation point
        if mid < right and arr[mid] > arr[mid + 1]:
            return mid + 1
        if mid > left and arr[mid] < arr[mid - 1]:
            return mid
        
        # Decide the search space
        if arr[mid] >= arr[left]:
            left = mid + 1
        else:
            right = mid - 1
    
    return 0
```

## Sorting

##### Stability in Sorting

A stable sort preserves the order in which repeated elements were sorted after a subsequent sort.

```python
x = [
    ("B", 1),
    ("C", 7),
    ("C", 8),
    ("D", 1),
    ("D", 4),
    ("E", 9),
    ("E", 10),
    ("F", 2),
    ("F", 7),
    ("F", 9),
]

x.sort(key=lambda x: x[0])

output = [
    ("B", 1),
    ("C", 7),
    ("C", 8),
    ("D", 1),
    ("D", 4),
    ("E", 9),
    ("E", 10),
    ("F", 2),
    ("F", 7),
    ("F", 9),
]
```

Notice how ('D',1) precedes ('D', 4). Since the element by which the list was sorted is x[0], the repeated elements are kept in the same order as the original list.

```python
def selection_sort(arr: list[int]) -> list[int]:
    for outer in range(len(arr)):
        minimum = outer
        for inner in range(outer, len(arr)):
            if arr[inner] < arr[minimum]:
                minimum = inner
        (arr[outer], arr[minimum]) = (arr[minimum], arr[outer])

    return arr
```

```python
def bubble(self, nums: List[int]) -> None:
    n = len(nums)
    for i in range(n):
        for j in range(n - 1, i, -1):
            if nums[j] < nums[j - 1]:
                nums[j], nums[j - 1] = nums[j - 1], nums[j]
```

```python
def insertion_sort(arr):
    for i in range(len(arr)):
        j = i
        while j > 0 and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j = j - 1
    return arr
```

```python
def msort(A, start, end):
    if start >= end:
        return

    # Divide
    mid = (start + end) // 2

    # Conquer
    msort(A, start, mid)
    msort(A, mid + 1, end)

    # Combine
    i = start
    j = mid + 1
    aux = []
    while i <= mid and j <= end:
        if A[i] <= A[j]:
            aux.append(A[i])
            i += 1
        else:
            aux.append(A[j])
            j += 1

    while i <= mid:
        aux.append(A[i])
        i += 1
    while j <= end:
        aux.append(A[j])
        j += 1

    # Gather
    A[start : end + 1] = aux


def mergesort(A):
    msort(A, 0, len(A) - 1)
    return A
```

```python
# Hoare's Partition -> 2-Pointer Paradigm

import random


def hoare(A, low, high):
    pivot = A[random.randint(low, high)]

    i, j = low - 1, high + 1
    while True:
        i += 1
        while A[i] < pivot:
            i += 1
        j -= 1
        while A[j] > pivot:
            j -= 1
        if i >= j:
            return j
        A[i], A[j] = A[j], A[i]


def quicksort(A, low=0, high=None):
    if high is None:
        high = len(A) - 1
    if low < high:
        p = hoare(A, low, high)
        quicksort(A, low, p)
        quicksort(A, p + 1, high)
    return A
```

```python
# QuickSelect puts the 1-indexed kth smallest element at index k-1.
def quickselect(A, k, l, r):
    if l==r:
        return A[l]
    p = hoare(A, l, r)
    if k<=p:
        return quickselect(A, k, l, p)
    elif k==p+1:
        return A[p]
    else:
        return quickselect(A, k, p+1, r)
```

```python
# Partition using 3-Pointers for Duplicates
import random


def partition(A, low, high):
    pivot = A[random.randint(low, high)]

    l, r = low - 1, high + 1
    i = low
    while i < r:
        if A[i] < pivot:
            l += 1
            A[l], A[i] = A[i], A[l]
            i += 1
        elif A[i] > pivot:
            r -= 1
            A[r], A[i] = A[i], A[r]
        else:
            i += 1
    return l, r


def quick_sort(A, low=0, high=None):
    if high is None:
        high = len(A) - 1
    if low < high:
        l, r = partition(A, low, high)
        quick_sort(A, low, l)
        quick_sort(A, r, high)
    return A
```

```python
# Lomuto's Partition - 2-Pointer Paradigm
def lomuto(A, low, high):
    p = random.randint(low, high)
    A[p], A[high] = A[high], A[p]

    i = low - 1  # boundary for lower group
    for j in range(low, high):
        if A[j] <= A[high]:
            # when j is less than or equal to the pivot the pivot
            # swap with the boundary index
            i += 1  # move boundary up
            A[i], A[j] = A[j], A[i]
    A[high], A[i + 1] = A[i + 1], A[high]
    return i + 1


def quick_sort(A, low=0, high=None):
    if high is None:
        high = len(A) - 1
    if low < high:
        p = lomuto(A, low, high)
        quick_sort(A, low, p - 1)
        quick_sort(A, p + 1, high)
    return A
```

```python
def heapify(arr, n, root):
    largest = root
    l = 2 * root + 1
    r = 2 * root + 2
    if l < n and arr[root] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != root:
        arr[root], arr[largest] = arr[largest], arr[root]
        heapify(arr, n, largest)


def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr
```

## Linked Lists

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

```python
def reverse_linked_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

```python
def get_midpoint(head: ListNode) -> ListNode:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```


## Recursion

### Combinatorial Enumeration

| $$n$$ objects, $$k$$ spots | w/o replacement      | w/ replacement        |
| -------------------------- | -------------------- | --------------------  |
| ordered -> permutation     | $$\frac{n!}{k!}$$    | $$n^{k}$$             |
| unordered -> combination   | $$n \choose k$$      | $${n+k-1 \choose k}$$ |

*Note* $$n \choose k = \frac{n!}{k!\times(n-k)!}$$

##### Permutation with Replacement | Ordered with Repetition | Pincodes

```python
""" Generate all strings consisting of digits of length n. """


def generate(n):
    enumerator(list(range(10)), "", n)


def enumerator(slate, solution, n):
    if n == 0:
        print(solution)
    for i in slate:
        enumerator(slate, solution + str(i), n - 1)
```

##### Permutation without Replacement | Ordered without Repetition | Seating Orders

```python
""" Generate all orderings of a set of numbers. """


def enumerator(slate, solution, outputs):
    if len(slate) == 0:
        outputs.append(list(solution))
        return

    for i in list(slate):
        slate.remove(i)
        enumerator(slate, solution + [i], outputs)
        slate.add(i)
    return outputs


def get_permutations(arr):
    slate = set(arr)
    return enumerator(slate, [], outputs)
```

##### Next Lexicographic Permutation

```python
""" Generate next lexicographic permutation. """
def next_permutation(nums: List[int]) -> None:
    # Step 1: Find the largest index i such that nums[i] < nums[i+1]
    # arr = [1,2,3,5,4]
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i+1]:
        i -= 1
    # i = 2 where arr[i]=3
    # if i==-1, you can reverse and return the original here
    # Step 2: Find the largest index j such that nums[j] > nums[i]
    if i >= 0:
        j = len(nums) - 1
        while j >= 0 and nums[j] <= nums[i]:
            j -= 1
        # find the largest inversion from the left, here arr[4]
        # Step 3: Swap nums[i] and nums[j]
        nums[i], nums[j] = nums[j], nums[i]
        # arr = [1,2,4,5,3]

    # Step 4: Reverse the subarray from index i+1 to the end
    """ in-place reverse """
    left, right = i + 1, len(nums) - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
    """
    ' uses extra space  '
    nums[i+1:] = reversed(nums[i+1:])
    """
    # arr = [1,2,4,3,5]

```

##### Combination without Replacement | Unordered without Replacement | Subsets

```python
""" Generate all subsets of a set of numbers """


def enumerator(slate, solution, outputs):
    if slate:
        nxt = slate[-1]
        enumerator(slate[:-1], solution + [nxt], outputs)
        enumerator(slate[:-1], solution, outputs)
        # no need to reset slate since
        # all we do is pass a subset of the slate to the next call
        # we don't actually edit the slate
    else:
        outputs.append(tuple(solution))


def generate_all_subsets(s):
    outputs = []
    enumerator(s[::-1], [], outputs)
    return outputs
```

```python
""" Generate all subsets of a set of numbers using a set based implementation"""


def enumerator(slate, solution, outputs):
    if slate:
        nxt = slate.pop()
        enumerator(slate, solution + [nxt], outputs)
        enumerator(slate, solution, outputs)
        slate.add(nxt)
    else:
        outputs.append(tuple(solution))


def generate_all_subsets(s):
    outputs = []
    enumerator(set(s[::-1]), [], outputs)
    return outputs
```

##### Combinations with Replacement | from `n` bins choose `k` objects

```python
"""
Select a group of k objects from n bins.
Each bin contains multiple copies of the same kind of object.
"""


def combinations_with_replacement(elements, r):
    if r == 0:
        return [[]]

    result = []
    for i, element in enumerate(elements):
        sub_combinations = combinations_with_replacement(elements[i:], r - 1)
        for sub_combination in sub_combinations:
            result.append([element] + sub_combination)

    return result
```

```python
""" All combinations with repetition of len==k of range(0,n) """
def combine(n: int, k: int) -> List[List[int]]:
    def backtrack(start, path, result):
        if len(path)==k:  # If target becomes negative, stop recursion
            return result + [path]
        for i in range(start, n):
            # Explore all combinations starting from the current index
            result = backtrack(i, path + [i], result)
        return result

    result = backtrack(0, [], [])
    return result

""" Without repetition  """
def combine(n: int, k: int) -> List[List[int]]:
    def backtrack(start, path, result):
        if len(path)==k:  # If target becomes negative, stop recursion
            return result + [path]
        for i in range(start, n):
            # Explore all combinations starting from the current index
            result = backtrack(i, path + [i], result)
        return result

    result = backtrack(0, [], [])
    return result



```

## Trees

### Tree Traversal

```python
def bfs_binary_tree(root):
    q = deque([root])
    values = []

    while q:
        values.append([])
        # len(q) = number of nodes in the current level
        for _ in range(len(q)):
            current = q.popleft()
            values[-1].append(current.value)
            if current.left:
                q.append(current.left)
            if current.right:
                q.append(current.right)

    return values
```

```python
from collections import deque


def bfs_n_ary_tree(root):
    q = deque([root])
    values = []

    while q:
        values.append([])
        nodes_counter = len(q)
        for _ in range(nodes_counter):
            current = q.popleft()
            values[-1].append(current.value)
            for child in current.children:
                q.append(child)

    return values
```

```python
def dfs(root):
    values = []

    def traverse(root, values):
        if root:
          # 1 if pre-order -> values.append(root)
          if root.left: traverse(root.left, values)
          # 2 if in-order: -> values.append(root)
          if root.right: traverse(root.right, values)
          # 3 if post-order -> values.append(root)

    traverse(root, values)
    return values
```

#### DFS - Bottom Up Information Flow

```python
def lowestCommonAncestor(root, p, q):
    def dfs(root, a, b):
        # base case
        if root in [a, b, None]:
            return root

        # recursive step
        l = dfs(root.left, a, b)
        r = dfs(root.right, a, b)

        # Post-Order traversal after information retrieval from children.
        # We're looking for either p or, which are unique in the tree.
        # It's unecessary to know which one was found.
        if l and r:
            return root
        elif l:
            return l
        else:
            return r

    return dfs(root, p, q)
```

### BST

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert(self.root, val)

    def _insert(self, node, val):
        if val < node.val:
            if not node.left:
                node.left = TreeNode(val)
            else:
                self._insert(node.left, val)
        elif val > node.val:
            if not node.right:
                node.right = TreeNode(val)
            else:
                self._insert(node.right, val)

    def search(self, val):
        return self._search(self.root, val)

    def _search(self, node, val):
        if not node:
            return False
        if val == node.val:
            return True
        elif val < node.val:
            return self._search(node.left, val)
        else:
            return self._search(node.right, val)

    def predecessor(self, val):
        pred = None
        node = self.root
        while node:
            if node.val < val:
                pred = node.val
                node = node.right
            else:
                node = node.left
        return pred

    def successor(self, val):
        succ = None
        node = self.root
        while node:
            if node.val > val:
                succ = node.val
                node = node.left
            else:
                node = node.right
        return succ
```

## Graphs

### Representations

```python
def edge_list_to_adjacency_list(edge_list, undirected=True):
    adjacency_list = {}
    for edge in edge_list:
        v1, v2 = edge
        adjacency_list.setdefault(v1, []).append(v2)
        if undirected:
            adjacency_list.setdefault(v2, []).append(v1)
    return adjacency_list
```

```python
def edge_list_to_adjacency_matrix(edge_list, num_vertices):
    """Edge ids go from 1 to n."""
    matrix = [[0] * num_vertices for _ in range(num_vertices)]

    for v1, v2 in edge_list:
        """edge 1, 2 is stored at 0,1"""
        matrix[v1 - 1][v2 - 1] = 1
        matrix[v2 - 1][v1 - 1] = 1

    return matrix
```

#### Grid Representation

```python
def get_neighbors(matrix, row, col):
    n_rows, n_cols = len(matrix), len(matrix[0])
    offsets = [1, 0, -1]

    for a in offsets:
        for b in offsets:
            i, j = row + a, col + b

            # Are the new coordinates within the boundaries?
            if 0 <= i < n_rows and 0 <= j < n_cols and (a != 0 or b != 0):
                # Check (i, j) is non-zero (0 represents no in/out edges).
                if matrix[i][j] != 0:
                    yield (i, j)
```

```python
from typing import List

class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []

        nrows, ncols = len(heights), len(heights[0])

        def dfs(node, visited):
            visited.add(node)
            x, y = node
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                        0 <= nx < nrows # row is in bounds
                        and 0 <= ny < ncols # column is in bounds
                        and (nx, ny) not in visited # node is unvisited
                        and heights[nx][ny] >= heights[x][y]
                    ):
                    visited = visited | dfs((nx, ny), visited)
            return visited

        pacific = set()
        atlantic = set()

        for i in range(nrows):
            # first column
            pacific |= dfs((i, 0), set())
            # last column
            atlantic |= dfs((i, ncols - 1), set())

        for j in range(ncols):
            # first row
            pacific |= dfs((0, j), set())
            # last row
            atlantic |= dfs((nrows - 1, j), set())

        return list(pacific & atlantic)


```

### Graph Traversal

```python
visited = set()


def dfs(graph, node, visited):
    if node not in visited:
        print(node)
        visited.add(node)

        for neighbor in graph[node]:
            dfs(graph, neighbor, visited)
```

```python
from collections import deque


def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        current = queue.popleft()
        if current not in visited:
            print(current)
            visited.add(current)

            # Enqueue unvisited neighbors
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.extend(neighbor)
```

### TopSort

```python
""" Returns if the graph is cyclic, i.e. not a DAG. """
from collections import deque, defaultdict
from typing import List
def top_sort(numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Build adjacency list and in-degrees
        adj_list = defaultdict(list)
        in_degrees = [0] * numCourses

        for course, prereq in prerequisites:
            adj_list[prereq].append(course)
            in_degrees[course] += 1

        # Initialize a queue with nodes having in-degree 0
        queue = deque([node for node in range(numCourses) if in_degrees[node] == 0])

        # Perform BFS
        while queue:
            current = queue.popleft()
            for neighbor in adj_list[current]:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    queue.append(neighbor)

        # If there are still nodes with in-degree > 0, there is a cycle
        return sum(in_degrees) == 0

```

```python

""" Topological sorting of a graph if the graph is a DAG. """
from collections import defaultdict, deque

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:

        def build_adj_indegrees(prerequisites):
            adj_list = defaultdict(list)
            indegree = defaultdict(int)
            for course, prereq in prerequisites:
                adj_list[prereq].append(course)
                indegree[course] += 1
            return adj_list, indegree

        adj_list, in_degrees = build_adj_indegrees(prerequisites)

        queue = deque(node for node in range(numCourses) if in_degrees[node] == 0)
        order = []
        visited = set(queue)  # Initialize visited with nodes having 0 in-degree
        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in adj_list[current]:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    queue.append(neighbor)
                    visited.add(neighbor)  # Mark as visited when added to queue

        if len(order) < numCourses:
            return []  # Not all courses can be taken
        return order
```

## Trie

```python

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```
