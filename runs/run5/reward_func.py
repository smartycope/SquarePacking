    def _get_reward(self):
        # We generally prefer living longer
        score = 100
        small_side_len = 5
        longevity_importance = 1
        side_importance = 50
        centered_importance = 0
        boundary_badness = 0.1
        if not self.boundary:
            boundary_badness = 0

        score += self.steps * longevity_importance

        # score -= math.e**(self.side_len * side_importance)
        score -= (self.side_len - small_side_len) * side_importance

        # We like it if they're in a small area
        if self.side_len < small_side_len and self.start_config != 'array':
            score += 200

        # We want to incentivize not touching, instead of disincentivizing touching,
        # because this way it doesn't also disincentivize longer runs
        # (if the reward is positive by default (not touching), then a longer run is okay)

        # We don't like it when they overlap at all
        if self.overlap_area > 0:
            score -= 100_000
            # We really don't like it when they overlap
            score -= math.e**(self.overlap_area)

        # This is essentially a percentage of how much they're overlapping
        # score -= self.overlap_area / (self.N - self.max_overlap)**2

        # I don't want them to just push up against the edges
        if boundary_badness or centered_importance:
            for x, y, _rot in self.squares:
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
                if centered_importance:
                    score -= (math.e ** dist([x, y], [self.search_space / 2, self.search_space / 2]) * centered_importance) / self.N

        return score
