import torch

class Bomb:
    def __init__(self, xywhn: torch.Tensor, maze_size: int):
        # print("BOMB: ", xywhn)
        self.xywhn = xywhn
        self.x = float(xywhn[0])
        self.y = float(xywhn[1])
        # Make the bomb a large square to avoid it when its moving
        size_multiplier = 1.25
        # self.size_x = float(xywhn[2]) * size_multiplier
        # self.size_y = float(xywhn[3]) * size_multiplier
        self.size_x = 0.19 * size_multiplier
        self.size_y = 0.19 * size_multiplier
        self.top_left = (self.x - self.size_x, self.y - self.size_y)
        self.top_right = (self.x + self.size_x, self.y - self.size_y)
        self.bottom_left = (self.x - self.size_x, self.y + self.size_y)
        self.bottom_right = (self.x + self.size_x, self.y + self.size_y)

        self._clamp_corners()

        self.scale_top_left = (self.top_left[0] * maze_size, self.top_left[1] * maze_size)
        self.scale_top_right = (self.top_right[0] * maze_size, self.top_right[1] * maze_size)
        self.scale_bottom_left = (self.bottom_left[0] * maze_size, self.bottom_left[1] * maze_size)
        self.scale_bottom_right = (self.bottom_right[0] * maze_size, self.bottom_right[1] * maze_size)


    def _clamp_corners(self):
        # Clamp the corner values to be inside the screen
        self.top_left = self._clamp(self.top_left)
        self.top_right = self._clamp(self.top_right)
        self.bottom_left = self._clamp(self.bottom_left)
        self.bottom_right = self._clamp(self.bottom_right)

    def _clamp(self, coords: tuple):
        # Clamps values to be inside the screen
        x, y = coords
        x = max(0, min(x, 1))
        y = max(0, min(y, 1))
        return (x, y)

    def is_colliding(self, x: float, y: float, rescale: bool = False):
        # Check if the given point is inside the bomb
        top_left = self.top_left
        top_right = self.top_right
        bottom_right = self.bottom_right
        if rescale:
            top_left = self.scale_top_left
            top_right = self.scale_top_right
            bottom_right = self.scale_bottom_right

        # print("COMPARING: ", x, y, top_left, top_right, bottom_right)
        # Check if x is left of the bomb
        if x < top_left[0]:
            return False
        # Check if x is right of the bomb
        if x > top_right[0]:
            return False
        # Check if y is above the bomb
        if y < top_right[1]:
            return False
        # Check if y is below the bomb
        if y >  bottom_right[1]:
            return False
        # If all the above conditions are false, then the point is inside the bomb
        # print("RETURNING TRUE")
        return True
