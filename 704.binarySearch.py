from typing import List
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1

        while l <= r:
            mid = (r + l) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return -1


# nums = [-1,0,3,5,9,12], tagert = 2
# if yes -> index, O.W -> -1

# l = 2 , r = 2, mid = 1 , nums[1] = 0 -> return -1


# 2 pointers: l, r , mid 
# T: O(logn)
# S: O(1)