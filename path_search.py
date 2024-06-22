import heapq
import numpy as np
import torch
import heapq
from collections import deque

def _distance(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

        
def modified_astar(start: tuple, end: tuple, bombs: torch.Tensor, point_inside_bomb: callable, map_size: int, interval: int=1):
    # Find the shortest path around the bombs
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: _distance(start, end)}
    open_heap = []

    heapq.heappush(open_heap, (fscore[start], start))

    while open_heap:
        # Get the point with the lowest fscore
        current = heapq.heappop(open_heap)[1]

        # If the current point is the end point, reconstruct the path and return it
        if current == end:
            path = deque()
            i = interval
            while current in came_from:
                if i == interval:
                    path.appendleft(current)
                    i = 0
                current = came_from[current]
                i += 1
            # Add the last point on the end fruit
            if start in came_from:
                path.appendleft(start)
            return list(path)
        
        # Add the current point to the close set
        close_set.add(current)
        # Check the neighbors of the current point
        for dx, dy in directions:
            # Get the neighbor point
            neighbor = (current[0] + dx, current[1] + dy)
            # If the neighbor is in the close set, outside the screen, or inside a bomb, skip it
            if (neighbor in close_set or neighbor[0] < 0 or neighbor[0] >= map_size or neighbor[1] < 0 or neighbor[1] >= map_size or point_inside_bomb(neighbor[0], neighbor[1], True)):
                close_set.add(neighbor)
                continue

            # Calculate the gscore of the neighbor
            tentative_gscore = gscore[current] + 1
            # If the neighbor is not in the open set or the gscore is lower than the current gscore of the neighbor
            if (neighbor not in gscore or tentative_gscore < gscore[neighbor]):
                # Update the gscore, fscore, and add the neighbor to the open set
                came_from[neighbor] = current
                gscore[neighbor] = tentative_gscore
                fscore[neighbor] = gscore[neighbor] + _distance(neighbor, end)
                heapq.heappush(open_heap, (fscore[neighbor], neighbor))

    # If there is no path, return an empty list
    return []
