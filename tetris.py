from copy import deepcopy
import cv2
import numpy as np
import random
from typing import Final, NamedTuple, TypeAlias

Grid: TypeAlias = list[list[int]]
Offsets: TypeAlias = tuple[tuple[int, int], ...]
Reward: TypeAlias = int
Features: TypeAlias = np.ndarray

class Action(NamedTuple):
    rotation_index: int
    c: int

class Tetromino:
    NUM_TETROMINOS: Final[int] = 7
    TETROMINO_OFFSETS: Final[tuple[tuple[Offsets, ...], ...]] = (
        ( # O
            ((0, 0), (0, 1), (1, 0), (1, 1)),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
            ((0, 0), (0, 1), (1, 0), (1, 1))
        ),
        ( # I
            ((1, 0), (1, 1), (1, 2), (1, 3)),
            ((0, 2), (1, 2), (2, 2), (3, 2)),
            ((2, 0), (2, 1), (2, 2), (2, 3)),
            ((0, 1), (1, 1), (2, 1), (3, 1))
        ),
        ( # Z
            ((0, 0), (0, 1), (1, 1), (1, 2)),
            ((0, 2), (1, 1), (1, 2), (2, 1)),
            ((1, 0), (1, 1), (2, 1), (2, 2)),
            ((0, 1), (1, 0), (1, 1), (2, 0))
        ),
        ( # S
            ((0, 1), (0, 2), (1, 0), (1, 1)),
            ((0, 1), (1, 1), (1, 2), (2, 2)),
            ((1, 1), (1, 2), (2, 0), (2, 1)),
            ((0, 0), (1, 0), (1, 1), (2, 1))
        ),
        ( # J
            ((0, 0), (1, 0), (1, 1), (1, 2)),
            ((0, 1), (0, 2), (1, 1), (2, 1)),
            ((1, 0), (1, 1), (1, 2), (2, 2)),
            ((0, 1), (1, 1), (2, 0), (2, 1))
        ),
        ( # L
            ((0, 2), (1, 0), (1, 1), (1, 2)),
            ((0, 1), (1, 1), (2, 1), (2, 2)),
            ((1, 0), (1, 1), (1, 2), (2, 0)),
            ((0, 0), (0, 1), (1, 1), (2, 1))
        ),
        ( # T
            ((0, 1), (1, 0), (1, 1), (1, 2)),
            ((0, 1), (1, 1), (1, 2), (2, 1)),
            ((1, 0), (1, 1), (1, 2), (2, 1)),
            ((0, 1), (1, 0), (1, 1), (2, 1))
        ),
    )

    def __init__(self) -> None:
        self.tetromino_index: int
        self.rotation_index: int
        self.tetromino_offsets: Offsets
        self.r: int
        self.c: int

class Tetris:    
    GRID_ROWS: Final[int] = 20
    GRID_COLS: Final[int] = 10

    def __init__(self) -> None:
        self.grid: Grid
        self.tetromino_bag: list[int]
        self.current_tetromino: Tetromino = Tetromino()
        self.next_tetromino: Tetromino = Tetromino()
        self.reward: Reward
        self.reset()
    
    def reset(self) -> tuple[Reward, Features]:
        self.grid = [[0 for c in range(Tetris.GRID_COLS)] for r in range(Tetris.GRID_ROWS)]
        self.tetromino_bag = []
        self._spawn(self.current_tetromino)
        self._spawn(self.next_tetromino)
        self.reward = 0

        return self.reward, self._get_features(self.grid, self.current_tetromino)

    def _spawn(self, tetromino: Tetromino) -> None:
        if len(self.tetromino_bag) == 0:
            self.tetromino_bag += list(range(Tetromino.NUM_TETROMINOS))
            self.tetromino_bag += list(range(Tetromino.NUM_TETROMINOS))
            random.shuffle(self.tetromino_bag)

        tetromino.tetromino_index = self.tetromino_bag.pop()
        tetromino.rotation_index = 0
        tetromino.tetromino_offsets = Tetromino.TETROMINO_OFFSETS[tetromino.tetromino_index][tetromino.rotation_index]
        tetromino.r = 0
        tetromino.c = 0

    def _check_for_collision(self) -> bool:
        for r0, c0 in self.current_tetromino.tetromino_offsets:
            r: int = self.current_tetromino.r + r0
            c: int = self.current_tetromino.c + c0

            if r >= Tetris.GRID_ROWS or self.grid[r][c] == 1:
                return True

        return False

    def _add_to_grid(self) -> Grid:
        grid: Grid = deepcopy(self.grid)

        for r0, c0 in self.current_tetromino.tetromino_offsets:
            grid[self.current_tetromino.r + r0][self.current_tetromino.c + c0] = 1

        return grid

    def _clear_rows(self, grid: Grid) -> tuple[int, Grid]:
        num_cleared_rows: int = 0

        for r in range(Tetris.GRID_ROWS):
            if all(grid[r]):
                num_cleared_rows += 1
                del grid[r]
                grid.insert(0, [0 for c in range(Tetris.GRID_COLS)])
        
        return num_cleared_rows, grid

    def _get_features(self, grid: Grid, tetromino: Tetromino) -> Features:
        heights: list[int] = [0 for c in range(Tetris.GRID_COLS)]
        num_holes: list[int] = [0 for c in range(Tetris.GRID_COLS)]

        for c in range(Tetris.GRID_COLS):
            r: int = 0
            while r < Tetris.GRID_ROWS and grid[r][c] == 0:
                r += 1
            
            heights[c] = Tetris.GRID_ROWS - r

            while r < Tetris.GRID_ROWS:
                num_holes[c] += 1 if grid[r][c] == 0 else 0
                r += 1

        return np.array([heights + num_holes + [1 if i == tetromino.tetromino_index else 0 for i in range(Tetromino.NUM_TETROMINOS)]]).astype("float")

    def get_next_states(self) -> dict[Action, tuple[Reward, Features]]:
        next_states: dict[Action, tuple[Reward, Features]] = {}
        rotation_indices: list[int] = [0] if self.current_tetromino.tetromino_index == 0 else [0, 1, 2, 3]

        for rotation_index in rotation_indices:
            self.current_tetromino.rotation_index = rotation_index
            self.current_tetromino.tetromino_offsets = Tetromino.TETROMINO_OFFSETS[self.current_tetromino.tetromino_index][self.current_tetromino.rotation_index]
            min_c: int = min([c for _, c in self.current_tetromino.tetromino_offsets])
            max_c: int = max([c for _, c in self.current_tetromino.tetromino_offsets])

            for c in range(-min_c, Tetris.GRID_COLS - max_c):
                self.current_tetromino.r = 0
                self.current_tetromino.c = c

                while not self._check_for_collision():
                    self.current_tetromino.r += 1
                self.current_tetromino.r -= 1

                if self.current_tetromino.r >= 0:
                    grid: Grid = self._add_to_grid()
                    num_cleared_rows, grid = self._clear_rows(grid)
                    reward: Reward = 1 + num_cleared_rows * Tetris.GRID_COLS
                    next_states[Action(rotation_index=rotation_index, c=c)] = (reward, self._get_features(grid, self.next_tetromino))

        return next_states

    def step(self, action: Action, render: bool=False) -> Reward:
        self.current_tetromino.rotation_index = action.rotation_index
        self.current_tetromino.tetromino_offsets = Tetromino.TETROMINO_OFFSETS[self.current_tetromino.tetromino_index][self.current_tetromino.rotation_index]
        self.current_tetromino.r = 0
        self.current_tetromino.c = action.c
        self.reward = 0

        while not self._check_for_collision():
            if render:
                self.render()
            self.current_tetromino.r += 1
        self.current_tetromino.r -= 1

        self.grid = self._add_to_grid()
        num_cleared_rows, self.grid = self._clear_rows(self.grid)

        if render:
            self.render()

        self.reward = 1 + num_cleared_rows * Tetris.GRID_COLS
        self.current_tetromino.tetromino_index = self.next_tetromino.tetromino_index
        self.current_tetromino.rotation_index = self.next_tetromino.rotation_index
        self.current_tetromino.tetromino_offsets = self.next_tetromino.tetromino_offsets
        self.current_tetromino.r = self.next_tetromino.r
        self.current_tetromino.c = self.next_tetromino.c
        self._spawn(self.next_tetromino)

        return self.reward

    def render(self) -> None:
        grid: Grid = self._add_to_grid()
        image = cv2.resize(np.array(grid).astype(np.uint8) * 255, None, fx=30, fy=30, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(winname='image', mat=np.array(image))
        cv2.waitKey(1)