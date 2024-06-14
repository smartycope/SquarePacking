    def _get_reward(self):
        # We generally prefer living longer
        score = 10000
        boundary_badness = 50
        side_importance = .09
        centered_importance = .09
        small_side_len = 7

        side_len = self.side_len()
        overlap = self.overlap_area()

        score -= math.e**side_len * side_importance

        # We like it if they're in a small area
        if side_len < small_side_len:
            score += 6000

        # We want to incentivize not touching, instead of disincentivizing touching,
        # because this way it doesn't also disincentivize longer runs
        # (if the reward is positive by default (not touching), then a longer run is okay)

        # We don't like it when they overlap at all
        if overlap > 0:
            score -= 10000
            # _render('Overlap: -1000')

        # We really don't like it when they overlap
        score -= math.e**(overlap)

        # This is essentially a percentage of how much they're overlapping
        # score -= overlap / (self.N - self.max_overlap)**2

        # I don't want them to just push up against the edges
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
