import gymnasium as gym
from typing import Literal
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
from math import cos, sin, tan, pi
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.affinity import rotate
from shapely.ops import unary_union

from Cope import debug

# Polygon.centroid would simplify (and probably speed up) these a bunch
def multiPolygon2Space(multi, side_len=1):
    # Skip the first coordinate, because the first and last are the same
    rtn = compute_coords([list(p.exterior.coords)[1:] for p in multi.geoms])
    # Then normalize them to all be positive (due to our space requirements)
    for c in rtn:
        if c[2] < 0:
            c[2] = math.pi/2 + c[2]
    return np.array(rtn)

def space2MultiPolygon(space, side_len=1):
    # Autoreshape it if it's flat
    if len(space.shape) == 1:
        space = space.reshape((int(len(space)/3), 3))
    return MultiPolygon([Polygon(corners) for corners in compute_corners(space, sideLen=side_len)])

def compute_coords(corners: List[List[Tuple[float, float]]], sideLen=1) -> List[Tuple[float, float, float]]:
    rtn = []
    for square_corners in corners:
        # Compute the x, y, and rotation angle values for this square
        corner1, corner2, corner3, corner4 = square_corners
        x = (corner1[0] + corner2[0] + corner3[0] + corner4[0]) / 4
        y = (corner1[1] + corner2[1] + corner3[1] + corner4[1]) / 4
        # rot_rad = math.atan2(corner2[1] - corner1[1], corner2[0] - corner1[0])
        rot_rad = math.atan2(corner4[1] - corner3[1], corner4[0] - corner3[0])

        # Add the x, y, and rotation angle values to the result list
        rtn.append(np.array((x, y, rot_rad)))
    return rtn

def compute_corners(squares: List[Tuple[float, float, float]], sideLen=1) -> List[Tuple[float, float, float, float]]:
    rtn = []
    for x, y, rot_rad in squares:
        # Compute the coordinates of the four corners of the square
        half_side = sideLen / 2
        corners = [(half_side, half_side), (half_side, -half_side), (-half_side, -half_side), (-half_side, half_side)]
        rotated_corners = []
        for corner in corners:
            rotated_x = x + corner[0]*math.cos(rot_rad) - corner[1]*math.sin(rot_rad)
            rotated_y = y + corner[0]*math.sin(rot_rad) + corner[1]*math.cos(rot_rad)
            rotated_corners.append((rotated_x, rotated_y))
        rtn.append(rotated_corners)
    return rtn

# Not sure what this is or where it came from
# squares = MultiPolygon(convert2shapely([(random.uniform(0, space), random.uniform(0, space), random.uniform(1, math.pi/2)) for i in range(N)], side_len=scale))

class SquareEnv(gym.Env):
    metadata = {"render_modes": ["matplotlib", "shapely", 'pygame'], "render_fps": 4}

    def __init__(self,
                 render_mode  = None,
                 N            = 4,
                 search_space = None,
                 shift_rate   = .01,
                 rot_rate     = .001,
                 flatten      = False,
                 max_steps    = 1000,
                 boundary     = 0,
                 max_overlap  = .5,
                 bound_method:Literal['clip', 'loop'] = 'clip',
            ):
        """ N is the number of boxes
            search_space is the maximum length of the larger square we're allowing the smaller
                squares to be in
            shift_rate is the maximum rate at which we can shift the x, y values per step
            rot_rate is the maximum rate at which we can rotate a box per step
        """
        ### Parameter handling ###
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if search_space is None:
            search_space = N
        self.render_mode = render_mode
        self.search_space = search_space
        self.shift_rate = shift_rate
        self.rot_rate = rot_rate
        self.N = N
        self.steps = 0
        self.scale = 20
        self.offset = 50
        self.bound_method = bound_method.lower()
        self.max_overlap_percent = max_overlap
        self.max_steps = max_steps
        self.boundary = boundary
        size = self.search_space*self.scale+(self.offset*2)
        self.screen_size = np.array((size, size))
        self.screen = None
        self.surf = None
        self.extraNums = None
        self.userSurf = None
        self.userSurfOffset = 0
        self._userPrinted = False
        self.font = None
        self._flat = flatten

        ### Define the spaces ###
        if self._flat:
            # raise NotImplementedError('auto flattened spaces arent implemented yet')
            self.observation_space = spaces.Box(low=np.zeros((N*3,)),
                                            high=np.array([[search_space]*N, [search_space]*N, [math.pi/2]*N]).T.flatten(),
                                            dtype=np.float64, shape=(N*3,))

            # The action space is shifting & rotating the squares little bits at a time
            self.action_space = spaces.Box(low=np.array([[-shift_rate]*N, [-shift_rate]*N, [-rot_rate]*N]).T.flatten(),
                                        high=np.array([[shift_rate]*N, [ shift_rate]*N, [ rot_rate]*N]).T.flatten(),
                                        dtype=np.float64, shape=(N*3,))

        else:
            self.observation_space = spaces.Box(low=np.zeros((N,3)),
                                                high=np.array([[search_space]*N, [search_space]*N, [math.pi/2]*N]).T,
                                                dtype=np.float64, shape=(N,3))

            # The action space is shifting & rotating the squares little bits at a time
            self.action_space = spaces.Box(low=np.array([[-shift_rate]*N, [-shift_rate]*N, [-rot_rate]*N]).T,
                                        high=np.array([[shift_rate]*N, [ shift_rate]*N, [ rot_rate]*N]).T,
                                        dtype=np.float64, shape=(N,3))

    def _get_obs(self):
        rtn = multiPolygon2Space(self.squares)
        return rtn.flatten() if self._flat else rtn

    def _get_info(self):
        return {
            # 'Overlaps': not self.squares.is_valid,
            'overlap': self.overlap_area(),
            'len': self.side_len(),
            'wasted': self.wasted_space(),
            # 'loss': lossFunc(self.squares),
        }

    def _get_terminated(self):
        # Optimal: 3.789, best known: 3.877084
        # There's no overlapping and we're better than the previous best
        if self.N == 11 and self.squares.is_valid and self.side_len() < 3.877084:
            print('Holy cow, we did it!!!')
            print('Coordinates & Rotations:')
            print(multiPolygon2Space(self.squares))
            return True

        # If we're almost entirely overlapping, just kill it
        if abs(self.overlap_area() - self.squares.area) < self.max_overlap_percent:
            return True

        if self.steps > self.max_steps:
            return True

        return False

    def reset(self, seed=None, start_valid=True, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.steps = 0

        # Why does the Space constructor have a seed and not the .sample() method??
        if seed is None:
            newObs = self.observation_space.sample()
            self.squares = space2MultiPolygon(newObs)
            # We can't be deterministic AND auto start at a valid point
            # Also make sure we're within the boundaries
            while start_valid and not self.squares.is_valid and not self.within_boundary():
                newObs = self.observation_space.sample()
                self.squares = space2MultiPolygon(newObs)

        else:
            if self._flat:
                newObs = spaces.Box(low=np.zeros((self.N,3)),
                                    high=np.array([[self.search_space]*self.N, [self.search_space]*self.N, [math.pi/2]*self.N]).T,
                                    dtype=np.float64, shape=(self.N,3), seed=seed).sample()
            else:
                newObs = spaces.Box(low=np.zeros((self.N*3,)),
                                high=np.array([[self.search_space]*self.N, [self.search_space]*self.N, [math.pi/2]*self.N]).T.flatten(),
                                dtype=np.float64, shape=(self.N*3,), seed=seed).sample()

            self.squares = space2MultiPolygon(newObs)

        # if self.render_mode is not None:
            # self.render()

        return newObs, self._get_info()

    def step(self, action):
        self.steps += 1
        # Compute the shifted squares
        obs = multiPolygon2Space(self.squares)
        if self._flat:
            assert action.shape == (self.N*3,), f'Action given to step is the wrong shape (Expected shape ({self.N*3},), got {action.shape})'
            #TODO: I'm lazy and don't want to rewrite all that code
            action = action.reshape((self.N,3))
        else:
            assert action.shape == (self.N,3), f'Action given to step is the wrong shape (Expected shape ({self.N,3}), got {action.shape})'
        newObs = obs + action

        # Make sure we don't leave the observation space
        if self.bound_method == 'clip':
            # newObs = np.clip(newObs, [[0,0,0]]*self.N, [[self.search_space, self.search_space, math.pi/2]]*self.N)
            newObs[:,:2][newObs[:,:2] >  self.search_space] = self.search_space
            newObs[:,:2][newObs[:,:2] < 0]                  = 0
            newObs[:,2][newObs[:,2]   > math.pi/2]          = math.pi/2
            newObs[:,2][newObs[:,2]   < 0]                  = 0
        elif self.bound_method == 'loop':
            newObs[:,:2][newObs[:,:2] >  self.search_space] = 0
            newObs[:,:2][newObs[:,:2] < 0]                  = self.search_space
            newObs[:,2][newObs[:,2]   > math.pi/2]          = 0
            newObs[:,2][newObs[:,2]   < 0]                  = math.pi/2
        else:
            raise TypeError(f'Unknown `{self.bound_method}` bound_method provided')

        self.squares = space2MultiPolygon(newObs)

        info = self._get_info()
        terminated = self._get_terminated()
        reward = self.lossFunc()

        # if self.render_mode is not None:
            # self.render()

        if self._flat:
            newObs = newObs.flatten()
        #                                truncated?
        return newObs, reward, terminated, False, info

    def _init_pygame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('Square Packing')
            self.screen = pygame.display.set_mode(self.screen_size)

        if self.font is None:
            self.font = pygame.font.SysFont("Verdana", 10)

        if self.surf is None:
            self.surf = pygame.Surface(self.screen_size)
            self.surf.convert()
            self.surf.fill((255, 255, 255))

        if self.extraNums is None:
            self.extraNums = pygame.Surface(self.screen_size)
            self.extraNums.convert()
            self.extraNums.fill((255, 255, 255))

        if self.userSurf is None:
            self.userSurf = pygame.Surface(self.screen_size)
            self.userSurf.convert()
            self.userSurf.fill((255, 255, 255))

    def render(self):
        if self.render_mode == 'matplotlib':
            plt.gca().set_aspect('equal')
            for geom in self.squares.geoms:
                xs, ys = geom.exterior.xy
                plt.fill(xs, ys, alpha=0.5, fc='r', ec='none')
            plt.show()

        elif self.render_mode == 'shapely':
            display(self.squares)

        elif self.render_mode == 'pygame':
            self._init_pygame()
            # This would be perfect if it worked
            # pygame_surface = pygame.image.load(io.BytesIO(env.squares.svg(20, '#d12d2d', 175).encode()))

            # This doesn't need to be in self, but it is because of the way Python interacts with pygame (I think)
            self.surf.fill((255, 255, 255))

            # Draw the text from the loss function
            self.surf.blit(self.extraNums, (150, 0))
            self.surf.blit(self.userSurf, (0, self.search_space * self.scale + self.offset))
            # self._userPrinted = False
            self.userSurf.fill((255, 255, 255, 255))
            self.userSurfOffset = 0

            # Draw in the center of the window
            space = multiPolygon2Space(self.squares)
            space[:,:2] *= self.scale
            space[:,:2] += self.offset
            multi = space2MultiPolygon(space, self.scale)

            # Draw all the polygons
            for square in multi.geoms:
                pygame.draw.polygon(self.surf, (200, 45, 45, 175), square.exterior.coords)

            # Draw the helpful texts
            overlap = self.overlap_area()
            strings = (
                f'Step:          {self.steps}',
                f'Overlap:       {overlap:.2f} | {overlap / self.N**2:.0%}',
                f'Side Length:   {self.side_len():.2f}',
                f'Wasted:        {self.wasted_space():.2f}',
                f'Reward:        {self.lossFunc():.2f}',
                f'Within Bounds: {self.within_boundary()}',
            )
            # For some dumb error I don't understand
            try:
                for offset, string in enumerate(strings):
                    self.surf.blit(self.font.render(string, True, (0,0,0)), (5, 5 + offset*10))
            except:
                self.font = pygame.font.SysFont("Verdana", 10)
                for offset, string in enumerate(strings):
                    self.surf.blit(self.font.render(string, True, (0,0,0)), (5, 5 + offset*10))

            # Draw the bounding box of the squares
            gfxdraw.polygon(self.surf, (np.array(self.squares.minimum_rotated_rectangle.exterior.coords)*self.scale)+self.offset, (0,0,0))
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

            # I don't remember what this does
            self.surf = pygame.transform.scale(self.surf, self.screen_size)

            # Display to screen
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            pygame.display.flip()

        else:
            raise TypeError(f"Unknown render mode {self.render_mode}")

    def print(self, string):
        self._init_pygame()

        self.userSurf.blit(self.font.render(str(string), True, (0,0,0)), (5 + ((self.userSurfOffset // 40) * 100), 5 + (self.userSurfOffset % 40)))
        self.userSurfOffset += 10
        # self._userPrinted = True


    def overlap_area(self):
        overlapArea = 0
        for i, square1 in enumerate(self.squares.geoms):
            for square2 in list(self.squares.geoms)[i+1:]:
                if square1.intersects(square2):
                    overlapArea += square1.intersection(square2).area
        return overlapArea

    def side_len(self):
        x, y = self.squares.minimum_rotated_rectangle.exterior.coords.xy
        edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
        return max(edge_length)

    def wasted_space(self):
        return self.side_len()**2 - self.squares.area

    def lossFunc(self, verbose=True):
        # We generally prefer living longer
        score = 1500
        overlap = self.overlap_area()
        boundary_badness = 50
        side_importance = 40

        self._offset = 0
        if verbose and (self.extraNums is not None):
            self.extraNums.fill((255, 255, 255, 255))

        def _render(text):
            if verbose and (self.extraNums is not None):
                self._offset += 10
                self.extraNums.blit(self.font.render(text, True, (0,0,0)), (5, 5 + self._offset))

        # We want to incentivize not touching, instead of disincentivizing touching,
        # because this way it doesn't also disincentivize longer runs
        # (if the reward is positive by default (not touching), then a longer run is okay)
        # We don't like it when they overlap at all
        if overlap > 0:
            score -= 1000
            _render('Overlap: -1000')

        # We really don't like it when they overlap
        score -= math.e**(overlap)
        # This is essentially a percentage of how much they're overlapping
        # score -= overlap / (self.N - self.max_overlap_percent)**2
        _render(f'Overlap: -{math.e**(overlap)}')

        start = score
        # I don't want them to just push up against the edges
        for square in self.squares.geoms:
            center = square.centroid
            x = center.x
            y = center.y

            # Left
            if x < self.boundary:
                # score -= boundary_badness * x
                score -= boundary_badness
                # _render(f'Left: -{boundary_badness}')
            # Top
            if y < self.boundary:
                score -= boundary_badness
                # _render(f'Top: -{boundary_badness}')

            # Right
            if abs(x - self.search_space) < self.boundary:
                # score -= boundary_badness * abs(x - 1)
                score -= boundary_badness
                # _render(f'Right: -{boundary_badness}')
            # Bottom
            if abs(y - self.search_space) < self.boundary:
                score -= boundary_badness
                # _render(f'Bottom: -{boundary_badness}')

        _render(f'Boundaries: {score - start}')

        # score -= self.wasted_space()
        score -= self.side_len() * side_importance
        _render(f'Side: -{self.side_len() * side_importance}')

        return score

    def within_boundary(self):
        for square in self.squares.geoms:
            center = square.centroid
            if (center.x < self.boundary or
                center.y < self.boundary or
                abs(center.x - self.search_space) < self.boundary or
                abs(center.y - self.search_space) < self.boundary
            ): return False
        return True

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.font = None
