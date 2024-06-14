import dis
import enum
import itertools
import gymnasium as gym
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

# from Cope import debug

# Polygon.centroid would simplify (and probably speed up) these a bunch
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

def space2MultiPolygon(space):
    # Autoreshape it if it's flat
    if len(space.shape) == 1:
        space = space.reshape((int(len(space)/3), 3))
    # return MultiPolygon([Polygon(corners) for corners in compute_corners(space, sideLen=side_len)])
    return MultiPolygon(map(Polygon, compute_all_corners(space)))

# def compute_coords(corners: List[List[Tuple[float, float]]], sideLen=1) -> List[Tuple[float, float, float]]:
    # rtn = []
    # for square_corners in corners:
    #     # Compute the x, y, and rotation angle values for this square
    #     corner1, corner2, corner3, corner4 = square_corners
    #     x = (corner1[0] + corner2[0] + corner3[0] + corner4[0]) / 4
    #     y = (corner1[1] + corner2[1] + corner3[1] + corner4[1]) / 4
    #     # rot_rad = math.atan2(corner2[1] - corner1[1], corner2[0] - corner1[0])
    #     rot_rad = math.atan2(corner4[1] - corner3[1], corner4[0] - corner3[0])

    #     # Add the x, y, and rotation angle values to the result list
    #     rtn.append(np.array((x, y, rot_rad)))
    # return rtn

def compute_all_corners(squares: List[Tuple[float, float, float]]) -> np.ndarray:
    # return np.array([compute_corners(square) for square in squares])
    return np.array(list(map(compute_corners, squares)))

def compute_corners(square: Tuple[float, float, float]) -> np.ndarray: #[float, float, float, float]:
    # Rotation is in radians
    x, y, rot = square
    # Compute the coordinates of the four corners of the square
    # half_side = sideLen / 2
    return np.array([
        (x + corner[0]*math.cos(rot) - corner[1]*math.sin(rot),
         y + corner[0]*math.sin(rot) + corner[1]*math.cos(rot))
        for corner in [(.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5)]
    ])

# TODO: caching for efficiency

class SquareEnv(SimpleGym):
    def __init__(self, *args,
                 render_mode  = None,
                 N            = 4,
                 search_space = None,
                 shift_rate   = .01,
                 rot_rate     = .001,
                #  flatten      = False,
                 max_steps    = 1000,
                 boundary     = 0,
                 max_overlap  = .5,
                 start_valid  = True,
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
        """
        super().__init__(
            *args,
            max_steps=max_steps,
            screen_size=screen_size,
            background_color=(255, 255, 255),
            print_color=(0, 0, 0),
            name='Square Packing',
            show_vars={'FPS': 'fps'},
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
        self.offset = 50
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
        self.start_valid = start_valid
        self._trying_to_overlap = False
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

    def _get_obs(self):
        return self.squares.flatten()

    def _get_info(self):
        return {
            # 'Overlaps': not self.squares.is_valid,
            'overlap': self.overlap_area,
            'len': self.side_len,
            'wasted': self.wasted_space,
            # 'loss': lossFunc(self.squares),
        }

    def _get_terminated(self):
        # Optimal: 3.789, best known: 3.877084
        # There's no overlapping and we're better than the previous best
        if self.N == 11 and self.side_len < 3.877084 and self.is_valid:
            print('Holy cow, we did it!!!')
            print('Coordinates & Rotations:')
            print(self.squares)
            with open('~/SQUARE_PARAMETERS.txt', 'w') as f:
                f.write(str(self.squares))
            return True

        # If we're almost entirely overlapping, just kill it
        # if abs(self.overlap_area() - self.squares.area) < self.max_overlap:
        if self.overlap_area > self.max_overlap:
            return True

        # if self.steps > self.max_steps:
        #     return True

        return False

    def _get_reward(self):
        # We generally prefer living longer
        score = 10000
        boundary_badness = 0
        side_importance = .09
        centered_importance = 0
        small_side_len = 7

        side_len = self.side_len
        overlap = self.overlap_area

        score -= math.e**side_len * side_importance

        # We like it if they're in a small area
        if side_len < small_side_len:
            score += 6000

        # We want to incentivize not touching, instead of disincentivizing touching,
        # because this way it doesn't also disincentivize longer runs
        # (if the reward is positive by default (not touching), then a longer run is okay)

        # We don't like it when they overlap at all
        if overlap > 0 or self._trying_to_overlap:
            score -= 100_000

        # We really don't like it when they overlap
        score -= math.e**(overlap)

        # This is essentially a percentage of how much they're overlapping
        # score -= overlap / (self.N - self.max_overlap)**2

        # I don't want them to just push up against the edges
        if boundary_badness or centered_importance:
            for square in self.squares.geoms:
                center = square.centroid
                x = center.x
                y = center.y

                # Left
                if x < self.boundary:
                    score -= boundary_badness
                # Top
                if y < self.boundary:
                    score -= boundary_badness

                # Right
                if abs(x - self.search_space) < self.boundary:
                    score -= boundary_badness
                # Bottom
                if abs(y - self.search_space) < self.boundary:
                    score -= boundary_badness

                # We want the squares to be close to the center
                score -= (math.e ** dist([x, y], [self.search_space / 2, self.search_space / 2]) * centered_importance) / self.N

        return score

    def _step(self, action):
        self.steps += 1
        # Compute the shifted squares
        # obs = multiPolygon2Space(self.squares)
        # if self._flat:
        assert action.shape == (self.N*3,), f'Action given to step is the wrong shape (Expected shape ({self.N*3},), got {action.shape})'
        action = action.reshape((self.N,3))
        # else:
        # assert action.shape == (self.N,3), f'Action given to step is the wrong shape (Expected shape ({self.N,3}), got {action.shape})'


        old_squares = self.squares
        self.squares += action

        # Make sure we don't leave the observation space
        if self.bound_method == 'clip':
            self.squares[:,:2][self.squares[:,:2] >  self.search_space] = self.search_space
            self.squares[:,:2][self.squares[:,:2] < 0]                  = 0
            self.squares[:,2][self.squares[:,2]   > math.pi/2]          = math.pi/2
            self.squares[:,2][self.squares[:,2]   < 0]                  = 0
        elif self.bound_method == 'loop':
            self.squares[:,:2][self.squares[:,:2] >  self.search_space] = 0
            self.squares[:,:2][self.squares[:,:2] < 0]                  = self.search_space
            self.squares[:,2][self.squares[:,2]   > math.pi/2]          = 0
            self.squares[:,2][self.squares[:,2]   < 0]                  = math.pi/2
        # Loop the rotation, but clip the position
        elif self.bound_method == 'mixed':
            self.squares[:,:2][self.squares[:,:2] >  self.search_space] = self.search_space
            self.squares[:,:2][self.squares[:,:2] < 0]                  = 0
            self.squares[:,2][self.squares[:,2]   > math.pi/2]          = 0
            self.squares[:,2][self.squares[:,2]   < 0]                  = math.pi/2
        else:
            raise TypeError(f'Unknown `{self.bound_method}` bound_method provided')

        # self._trying_to_overlap = False
        # squares = space2MultiPolygon(new_squares)

        if self.disallow_overlap:
            # self._trying_to_overlap = True
            # current = list(squares.geoms)
            # for square1, square2 in itertools.combinations(current, 2):
            #     if square1.intersects(square2):
            #         i1 = current.index(square1)
            #         i2 = current.index(square2)

            for i1, square1 in enumerate(self.squares):
                # i1+1, because if a intercets b, then b intersects a. We don't need to check it again
                # We also don't need to check if a intersects a.
                for i2, square2 in enumerate(self.squares[i1+1:]):
                    if Polygon(compute_corners(square1)).intersects(Polygon(compute_corners(square2))):
                    # if square1.intersects(square2):
                        self.squares[i1] = old_squares[i1]
                        self.squares[i2] = old_squares[i2]
            # squares = MultiPolygon(current)

        # terminated = self._get_terminated()
        # reward = self.lossFunc()

        # if self.render_mode is not None:
            # self.render()

        # if self._flat:
        #     newObs = newObs.flatten()
        #                                truncated?
        # return newObs, reward, terminated, False, info

    def _reset(self, seed=None, options=None):
        self.steps = 0

        # Why does the Space constructor have a seed and not the .sample() method??
        if seed is None:
            self.squares = self.observation_space.sample().reshape((self.N, 3))
            # We can't be deterministic AND auto start at a valid point
            # Also make sure we're within the boundaries
            if self.start_valid:
                while not self.within_boundary or not self.is_valid:
                    self.squares = self.observation_space.sample().reshape((self.N, 3))

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


    @property
    def is_valid(self):
        """ True if there's no overlapping """
        return space2MultiPolygon(self.squares).is_valid

    @property
    def overlap_area(self):
        area = 0
        # for i, square1 in enumerate(self.squares.geoms):
            # for square2 in list(self.squares.geoms)[i+1:]:
        for square1, square2 in itertools.combinations(self.squares, 2):
            area += Polygon(compute_corners(square1)).intersection(Polygon(compute_corners(square2))).area
        return area

    @property
    def min_rotated_rect_corners(self) -> Tuple['minx', 'miny', 'maxx', 'maxy']:
        corners = compute_all_corners(self.squares)
        xs = corners[:,:,0]
        ys = corners[:,:,1]
        return np.min(xs), np.min(ys), np.max(xs), np.max(ys)

    @property
    def side_len(self):
        # minx = np.min(self.squares[])
        # x, y = self.squares.minimum_rotated_rectangle.exterior.coords.xy
        minx, miny, maxx, maxy = self.min_rotated_rect_corners
        return max(maxx - minx, maxy - miny)
        # edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
        # return max(edge_length)

    @property
    def wasted_space(self):
        return self.side_len**2 - self.N

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
        from IPython.display import display
        disp = display(display_id=True)
        disp.update(space2MultiPolygon(self.squares))

    def render_pygame(self):
        # self._init_pygame()
        # This would be perfect if it worked
        # pygame_surface = pygame.image.load(io.BytesIO(env.squares.svg(20, '#d12d2d', 175).encode()))

        # This doesn't need to be in self, but it is because of the way Python interacts with pygame (I think)
        # self.surf.fill((255, 255, 255))

        # Draw the text from the loss function
        # self.surf.blit(self.extraNums, (150, 0))
        # self.surf.blit(self.userSurf, (0, self.search_space * self.scale + self.offset))
        # self._userPrinted = False
        # self.userSurf.fill((255, 255, 255, 255))
        # self.userSurfOffset = 0

        # Draw in the center of the window
        # space = multiPolygon2Space(self.squares)
        # space[:,:2] *= self.scale
        # space[:,:2] += self.offset
        # multi = space2MultiPolygon(space, self.scale)

        # Draw all the polygons
        for square in self.squares:
            pygame.draw.polygon(self.surf, (200, 45, 45, 175), (square[:,:2] * self.offset) + self.offset)

        # Draw the helpful texts
        # overlap = self.overlap_area
        # strings = (
        #     f'Step:          {self.steps}',
        #     f'Overlap:       {overlap:.2f} | {overlap / self.N**2:.0%}',
        #     f'Side Length:   {self.side_len():.2f}',
        #     f'Wasted:        {self.wasted_space():.2f}',
        #     f'Reward:        {self.lossFunc():.2f}',
        #     f'Within Bounds: {self.within_boundary()}',
        # )
        # For some dumb error I don't understand
        # try:
        #     for offset, string in enumerate(strings):
        #         self.surf.blit(self.font.render(string, True, (0,0,0)), (5, 5 + offset*10))
        # except:
        #     self.font = pygame.font.SysFont("Verdana", 10)
        #     for offset, string in enumerate(strings):
        #         self.surf.blit(self.font.render(string, True, (0,0,0)), (5, 5 + offset*10))

        # Draw the bounding box of the squares
        gfxdraw.polygon(self.surf, (np.array(self.min_rotated_rect_corners)*self.scale)+self.offset, (0,0,0))
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

        # self.surf = pygame.transform.flip(self.surf, False, True)
        # I don't remember what this does
        # self.surf = pygame.transform.scale(self.surf, self.screen_size)

        # # Display to screen
        # self.screen.blit(self.surf, (0, 0))
        # pygame.event.pump()
        # pygame.display.flip()
