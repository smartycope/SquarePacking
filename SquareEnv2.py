import dis
from shapely import Point
import enum
import itertools
import gymnasium as gym
import random
from typing import Iterable, Literal
from gymnasium import spaces
import shapely.geometry as sg
import shapely.ops as so
import matplotlib.pyplot as plt
import numpy as np
import pygame
from pygame import gfxdraw
import numpy as np
from typing import List, Tuple
import math
from math import cos, dist, sin, tan, pi
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.affinity import rotate
from shapely.ops import unary_union
from Cope.gym import SimpleGym
# This is safe to remove if you're using pygame or nothing to render
from IPython.display import display


def multiPolygon2Space(multi):
    rtn = []
    for geom in multi.geoms:
        # Compute the x, y, and rotation angle values for this square
        # Skip the last coordinate, because the first and last are the same
        corner1, corner2, corner3, corner4, _ = geom.exterior.coords

        rot_rad = math.atan2(corner4[1] - corner3[1], corner4[0] - corner3[0])
        # Normalize the angles to all be positive (due to our space requirements)
        # if rot_rad < 0:
        #     rot_rad += math.pi/2

        # Add the x, y, and rotation angle values to the result list
        rtn.append(np.array((
            # x
            (corner1[0] + corner2[0] + corner3[0] + corner4[0]) / 4,
            # y
            (corner1[1] + corner2[1] + corner3[1] + corner4[1]) / 4,
            # Rotation (radians)
            rot_rad
        )))

    return np.array(rtn)

def space2MultiPolygon(space, side_len=1):
    # Autoreshape it if it's flat
    if len(space.shape) == 1:
        space = space.reshape((int(len(space)/3), 3))
    # return MultiPolygon([Polygon(corners) for corners in compute_corners(space, sideLen=side_len)])
    return MultiPolygon(map(Polygon, compute_all_corners(space, side_len=side_len)))

def compute_all_corners(squares: List[Tuple[float, float, float]], side_len=1) -> np.ndarray:
    # return np.array([compute_corners(square) for square in squares])
    return np.array(list(map(lambda s: compute_corners(s, side_len=side_len), squares)))

def compute_corners(square: Tuple[float, float, float], side_len=1) -> np.ndarray: #[float, float, float, float]:
    # Rotation is in radians
    x, y, rot = square
    # Compute the coordinates of the four corners of the square
    half = side_len / 2
    return np.array([
        (x + corner[0]*math.cos(rot) - corner[1]*math.sin(rot),
         y + corner[0]*math.sin(rot) + corner[1]*math.cos(rot))
        for corner in [(half, half), (half, -half), (-half, -half), (-half, half)]
    ])


class SquareEnv(SimpleGym):
    def __init__(self, *args,
                 N            = 4,
                 search_space = None,
                 shift_rate   = .01,
                 rot_rate     = .001,
                #  flatten      = False,
                 max_steps    = 1000,
                 boundary     = 0,
                 max_overlap  = .5,
                 start_config:Literal['valid', 'array', 'random'] = 'valid',
                 screen_size  = None,
                 # mixed is loop the rotation, but clip the position
                 bound_method:Literal['clip', 'loop', 'mixed'] = 'mixed',
                 disallow_overlap=False,
                 **kwargs,
            ):
        """ N is the number of boxes
            search_space is the maximum length of the larger square we're allowing the smaller
                squares to be in
            shift_rate is the maximum rate at which we can shift the x, y values per step
            rot_rate is the maximum rate at which we can rotate a box per step
            start_config defines we we reset the squares.
                If `random`, the boxes are randomly placed
                If `valid`, the boxes are randomly placed, but not overlapping
                If `array`, the boxes are arrayed in a grid, but are randomly "jiggled"
        """
        super().__init__(
            *args,
            max_steps=max_steps,
            screen_size=screen_size,
            background_color=(255, 255, 255),
            print_color=(0, 0, 0),
            name='Square Packing',
            # show_vars={'FPS': 'fps'},
            assert_valid_action=False,
            verbose=False,
            **kwargs
        )
        self._show_help = True

        if search_space is None:
            search_space = N

        # self.render_mode = render_mode
        self.search_space = search_space
        self.shift_rate = shift_rate
        self.rot_rate = rot_rate
        self.N = N
        # self.steps = 0
        self.scale = 20
        self.offset = 100
        self.bound_method = bound_method.lower()
        self.max_overlap = max_overlap
        self.max_steps = max_steps
        self.boundary = boundary
        self.disallow_overlap = disallow_overlap
        size = self.search_space*self.scale+(self.offset*2)
        if screen_size is None:
            self.screen_size = np.array((size, size))
        # self.screen = None
        # self.surf = None
        # self.extraNums = None
        # self.userSurf = None
        # self.userSurfOffset = 0
        # self._userPrinted = False
        # self.font = None
        # self._flat = flatten
        self.start_config = start_config
        self.movement = np.ones((self.N, 3))
        self.squares: np.ndarray # with the shape (N, 3): it gets flattened/reshaped to interface with the spaces

        ### Define the spaces ###
        # if self._flat:
        self.observation_space = spaces.Box(low=np.zeros((N*3,)),
            high=np.array([[search_space]*N, [search_space]*N, [math.pi/2]*N]).T.flatten(),
            dtype=np.float64, shape=(N*3,))

        # The action space is shifting & rotating the squares little bits at a time
        self.action_space = spaces.Box(low=np.array([[-shift_rate]*N, [-shift_rate]*N, [-rot_rate]*N]).T.flatten(),
            high=np.array([[shift_rate]*N, [shift_rate]*N, [rot_rate]*N]).T.flatten(),
            dtype=np.float64, shape=(N*3,))

        # else:
        #     self.observation_space = spaces.Box(low=np.zeros((N,3)),
        #                                         high=np.array([[search_space]*N, [search_space]*N, [math.pi/2]*N]).T,
        #                                         dtype=np.float64, shape=(N,3))

        #     # The action space is shifting & rotating the squares little bits at a time
        #     self.action_space = spaces.Box(low=np.array([[-shift_rate]*N, [-shift_rate]*N, [-rot_rate]*N]).T,
        #                                 high=np.array([[shift_rate]*N, [ shift_rate]*N, [ rot_rate]*N]).T,
        #                                 dtype=np.float64, shape=(N,3))

        self._obs_cache = None
        self._info_cache = None
        self._terminated_cache = None
        self._reward_cache = None
        self._is_valid_cache = None
        self._overlap_area_cache = None
        self._min_rotated_rect_extents_cache = None
        self._side_len_cache = None
        self._wasted_space_cache = None

    def _get_obs(self):
        if self._obs_cache is None:
            self._obs_cache = self.squares.flatten()
        return self._obs_cache

    def _get_info(self):
        if self._info_cache is None:
            self._info_cache = {
                # 'Overlaps': not self.squares.is_valid,
                'overlap': self.overlap_area,
                'len': self.side_len,
                'wasted': self.wasted_space,
                # 'loss': lossFunc(self.squares),
            }
        return self._info_cache

    def _get_terminated(self):
        if self._terminated_cache is None:
            self._terminated_cache = False
            # Optimal: 3.789, best known: 3.877084
            # There's no overlapping and we're better than the previous best
            if self.N == 11 and self.side_len < 3.877084 and self.is_valid():
                print('Holy cow, we did it!!!')
                print('Coordinates & Rotations:')
                print(self.squares)
                with open('~/SQUARE_PARAMETERS.txt', 'w') as f:
                    f.write(str(self.squares))
                self._terminated_cache = True

            # If we're almost entirely overlapping, just kill it
            elif self.overlap_area > self.max_overlap:
                self._terminated_cache = True

            # If all the angles are about the same, we've reached the trivial solution. Give up and
            # start over
            # elif len(np.unique(angles, return_counts=True)[1]) < 3:
            elif not np.any(np.round(self.squares[:,2], 1) % 1.6):
                self._terminated_cache = True

            # If we're pretty small, and we're only making small adjustments, don't reset, we're doing good!
            # print(self.movement)
            # if not np.all(np.median(self.movement, axis=1)) and self.side_len > 4.5:
            #     return True

        return self._terminated_cache

    def _get_reward(self):
        if self._reward_cache is None:
            # We generally prefer living longer
            score = 0 # Linear
            # small_side_len = 5.5 # Scalar
            # longevity_importance = 0 # Multiplied
            # side_importance = 1 # Multiplied
            side_len_importance = 1 # Multiplied
            centered_importance = 0 # Exponential
            boundary_badness = 0 # Linear
            absolute_overlap_penalty = 0 # Linear
            overlap_area_importance = 0 # Multiplied
            if not self.boundary:
                boundary_badness = 0

            # score += self.steps * longevity_importance

            # score -= math.e**(self.side_len * side_importance)
            # score -= (self.side_len - small_side_len) * side_importance
            score -= self.side_len * side_len_importance

            # We like it if they're in a small area
            # if self.side_len < small_side_len and self.start_config != 'array':
            #     score += 200

            # We want to incentivize not touching, instead of disincentivizing touching,
            # because this way it doesn't also disincentivize longer runs
            # (if the reward is positive by default (not touching), then a longer run is okay)

            # We don't like it when they overlap at all
            if self.overlap_area > 0:
                score -= absolute_overlap_penalty
                # We really don't like it when they overlap
                score -= self.overlap_area*overlap_area_importance
                # score -= math.e**(self.overlap_area)

            # This is essentially a percentage of how much they're overlapping
            # score -= self.overlap_area / (self.N - self.max_overlap)**2

            # I don't want them to just push up against the edges
            if boundary_badness or centered_importance:
                for x, y, _rot in self.squares:
                    # Left
                    # if x < self.boundary:
                    #     score -= boundary_badness
                    # # Top
                    # if y < self.boundary:
                    #     score -= boundary_badness

                    # # Right
                    # if abs(x - self.search_space) < self.boundary:
                    #     score -= boundary_badness
                    # # Bottom
                    # if abs(y - self.search_space) < self.boundary:
                    #     score -= boundary_badness

                    # We want the squares to be close to the center
                    if centered_importance:
                        # score -= (math.e ** dist([x, y], [self.search_space / 2, self.search_space / 2]) * centered_importance) / self.N
                        score -= dist([x, y], [self.search_space / 2, self.search_space / 2]) * centered_importance

            self._reward_cache = score
        return self._reward_cache

    def _step(self, action):
        # Invalidate caches
        self._obs_cache = None
        self._info_cache = None
        self._terminated_cache = None
        self._reward_cache = None
        self._is_valid_cache = None
        self._overlap_area_cache = None
        self._min_rotated_rect_extents_cache = None
        self._side_len_cache = None
        self._wasted_space_cache = None

        # The action is given flattened, but self.squares looks like [[x, y, radians], ...]
        assert action.shape == (self.N*3,), f'Action given to step is the wrong shape (Expected shape ({self.N*3},), got {action.shape})'
        action = action.reshape((self.N,3))

        # Compute the shifted squares
        new_squares = self.squares + action

        # Make sure we don't leave the observation space
        if self.bound_method == 'clip':
            new_squares[:,:2][new_squares[:,:2] >  self.search_space] = self.search_space
            new_squares[:,:2][new_squares[:,:2] < 0]                  = 0
            new_squares[:,2][new_squares[:,2]   > math.pi/2]          = math.pi/2
            new_squares[:,2][new_squares[:,2]   < 0]                  = 0
        elif self.bound_method == 'loop':
            new_squares[:,:2][new_squares[:,:2] >  self.search_space] = 0
            new_squares[:,:2][new_squares[:,:2] < 0]                  = self.search_space
            new_squares[:,2][new_squares[:,2]   > math.pi/2]          = 0
            new_squares[:,2][new_squares[:,2]   < 0]                  = math.pi/2
        # Loop the rotation, but clip the position
        elif self.bound_method == 'mixed':
            new_squares[:,:2][new_squares[:,:2] >  self.search_space] = self.search_space
            new_squares[:,:2][new_squares[:,:2] < 0]                  = 0
            new_squares[:,2][new_squares[:,2]   > math.pi/2]          = 0
            new_squares[:,2][new_squares[:,2]   < 0]                  = math.pi/2
        else:
            raise TypeError(f'Unknown `{self.bound_method}` bound_method provided')

        if self.disallow_overlap:
            for i1, square1 in enumerate(new_squares):
                # i1+1, because if a intercets b, then b intersects a. We don't need to check it again
                # We also don't need to check if a intersects a.
                for i2, square2 in enumerate(new_squares[i1+1:], start=i1+1):
                    if Polygon(compute_corners(square1)).intersects(Polygon(compute_corners(square2))):
                        new_squares[i1] = self.squares[i1]
                        new_squares[i2] = self.squares[i2]

        # self.movement = self.squares - new_squares
        self.squares = new_squares

    def _reset(self, seed=None, options=None):
        if 'manual' in options:
            self.squares = options['manual'].reshape((self.N, 3))
            return

        # Why does the Space constructor have a seed and not the .sample() method??
        if seed is None:
            # We can't be deterministic AND auto start at a valid point
            # Also make sure we're within the boundaries
            match self.start_config:
                case 'random':
                    self.squares = self.observation_space.sample().reshape((self.N, 3))
                case 'valid':
                    self.squares = self.observation_space.sample().reshape((self.N, 3))
                    while not self.within_boundary or not self.is_valid(False):
                        self.squares = self.observation_space.sample().reshape((self.N, 3))
                        # Invalidate the caches, so it actually recalculates
                        self._within_boundary_cache = None
                        self._is_valid_cache = None
                case 'array':
                    cols = math.ceil(math.sqrt(self.N))
                    added = 0
                    # Minimum so they can't overlap: 1.4142135623730951 (math.sqrt((.5**2)*2) * 2)
                    gap = 1.4143
                    startx = self.boundary + gap / 2
                    starty = self.boundary + gap / 2
                    squares = []
                    x = startx
                    y = starty
                    col = 0
                    while added < self.N:
                        squares.append([x, y, random.uniform(0, math.pi/2)])
                        added += 1
                        col += 1

                        if col >= cols:
                            y += gap
                            x = startx
                            col = 0
                        else:
                            x += gap
                    self.squares = np.array(squares)
                case _:
                    ValueError('Invalid start_config parameter given')

        else:
            # This is untested after the refactor
            # if self._flat:
            self.squares = spaces.Box(
                low  =np.zeros((self.N,3)),
                high =np.array([[self.search_space]*self.N, [self.search_space]*self.N, [math.pi/2]*self.N]).T,
                dtype=np.float64,
                shape=(self.N,3),
                seed =seed,
            ).sample()
            # else:
            #     self.squares = spaces.Box(low=np.zeros((self.N*3,)),
            #                     high=np.array([[self.search_space]*self.N, [self.search_space]*self.N, [math.pi/2]*self.N]).T.flatten(),
            #                     dtype=np.float64, shape=(self.N*3,), seed=seed).sample().reshape((self.N, 3))

    def is_valid(self, shallow=True):
        """ True if there's no overlapping """
        if self._is_valid_cache is None:
            self._is_valid_cache = (shallow and self.disallow_overlap) or space2MultiPolygon(self.squares).is_valid
        return self._is_valid_cache

    @property
    def overlap_area(self):
        if self._overlap_area_cache is None:
            if self.disallow_overlap:
                self._overlap_area_cache = 0
            else:
                area = 0
                # for i, square1 in enumerate(self.squares.geoms):
                    # for square2 in list(self.squares.geoms)[i+1:]:
                for square1, square2 in itertools.combinations(self.squares, 2):
                    area += Polygon(compute_corners(square1)).intersection(Polygon(compute_corners(square2))).area
                self._overlap_area_cache = area
        return self._overlap_area_cache

    def min_rotated_rect_extents(self, side_len=1) -> Tuple['minx', 'miny', 'maxx', 'maxy']:
        if self._min_rotated_rect_extents_cache is None:
            self._min_rotated_rect_extents_cache = space2MultiPolygon(self.squares).minimum_rotated_rectangle.exterior.coords
        return self._min_rotated_rect_extents_cache

        # rect = space2MultiPolygon(self.squares).minimum_rotated_rectangle
        # x, y = rect.exterior.coords.xy
        # edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
        # return max(edge_length)
        # corners = compute_all_corners(self.squares)
        # xs = corners[:,:,0]
        # ys = corners[:,:,1]
        # return np.min(xs), np.min(ys), np.max(xs), np.max(ys)

    @property
    def side_len(self):
        if self._side_len_cache is None:
            x, y = self.min_rotated_rect_extents().xy
            edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
            self._side_len_cache =  max(edge_length)
        return self._side_len_cache

        # minx = np.min(self.squares[])
        # x, y = self.squares.minimum_rotated_rectangle.exterior.coords.xy
        # minx, miny, maxx, maxy = self.min_rotated_rect_extents()
        # return max(maxx - minx, maxy - miny)

        # edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
        # return max(edge_length)

    @property
    def wasted_space(self):
        if self._wasted_space_cache is None:
            self._wasted_space_cache = self.side_len**2 - self.N
        return self._wasted_space_cache

    @property
    def within_boundary(self):
        if not self.boundary:
            return True
        for x, y, rot in self.squares:
            if (x < self.boundary or
                y < self.boundary or
                abs(x - self.search_space) < self.boundary or
                abs(y - self.search_space) < self.boundary
            ): return False
        return True


    # def render_matplotlib(self):
        # plt.gca().set_aspect('equal')
        # for geom in self.squares.geoms:
        #     xs, ys = geom.exterior.xy
        #     plt.fill(xs, ys, alpha=0.5, fc='r', ec='none')
        # plt.show()

    def render_shapely(self):
        # self._display_id.update(space2MultiPolygon(self.squares))
        display(space2MultiPolygon(self.squares), clear=True)

    def render_pygame(self):
        scaled_squares = self.squares.copy()
        scaled_squares[:,:2] *= self.scale
        scaled_squares[:,:2] += self.offset

        # Draw all the polygons
        for square in compute_all_corners(scaled_squares, side_len=self.scale):
            # print(square)
            pygame.draw.polygon(self.surf, (0xA3, 0xE0, 0xC2, .6), square)
            pygame.draw.polygon(self.surf, (0, 0, 0), square, width=1)

        # Draw the bounding box of the squares
        # minx, miny, maxx, maxy = self.min_rotated_rect_extents(side_len=self.scale)
        # corners = np.array([
        #     [minx, miny],
        #     [minx, maxy],
        #     [maxx, maxy],
        #     [maxx, miny],
        # ])
        # gfxdraw.polygon(self.surf, (corners*self.scale)+self.offset, (0,0,0))
        gfxdraw.polygon(self.surf, (np.array(self.min_rotated_rect_extents())*self.scale)+self.offset, (0,0,200))
        # Draw the search space
        gfxdraw.rectangle(self.surf, (self.offset, self.offset, self.search_space*self.scale, self.search_space*self.scale), (0,0,0))

        # Draw the boundaries
        if self.boundary:
            boundsColor = (210, 95, 79, 100)
            off = self.offset
            b = self.boundary * self.scale
            ss = self.search_space * self.scale
            # Top
            gfxdraw.box(self.surf, ((off, off), (ss-b-1, b)), boundsColor)
            # Right
            gfxdraw.box(self.surf, ((off+ss, off), (-b, ss-b)), boundsColor)
            # Left
            gfxdraw.box(self.surf, ((off+b, off+b), (-b, ss-b)), boundsColor)
            # Bottom
            gfxdraw.box(self.surf, ((off+b+1, off+ss-b), (ss-b, b)), boundsColor)
