import numpy as np

import functions


class Particle:
    def __init__(self, y, phi, weight):
        self.y = y
        self.phi = phi
        self.weight = weight

    def move(self, d, theta):
        """Moves the particle

        :param d: distance (metres)
        :param theta: rotation (radians)
        :return:
        """

        self.y += d * np.sin(self.phi)
        self.phi += theta


class ParticleFilter:
    def __init__(self, num_particles, y_min, y_max, phi_min, phi_max, lane_width):
        """Create particle filter

        :param num_particles: number of particles to create
        :param y_min: minimum y position of particles (metres)
        :param y_max: maximum y position of particles (metres)
        :param phi_min: minimum orientation of particles (radians)
        :param phi_max: maximum orientation of particles (radians)
        """
        self.num_particles = num_particles
        self.y_min = y_min
        self.y_max = y_max
        self.phi_min = phi_min
        self.phi_max = phi_max

        self.lane_width = lane_width

        self.particles = []

        # Initialise particles
        self.initialise_particles()

    def initialise_particles(self):
        """Initialise particles with uniform distribution"""
        self.particles = []

        for i in range(0, self.num_particles):
            particle = Particle(y=np.random.uniform(self.y_min, self.y_max),
                                phi=np.random.uniform(self.phi_min, self.phi_max),
                                weight=1. / self.num_particles)

            self.particles.append(particle)

    def step(self, v, omega, t, left_lane, right_lane):
        """A complete step of the particle filter

        :param v: velocity (metres/second)
        :param omega: angular velocity (radians/second)
        :param t: time (seconds)
        :param left_lane: an array of line points for the left lane in the image, e.g. [u1, v1, u2, v2]
        :param right_lane: an array of line points for the right lane in the image, e.g. [u1, v1, u2, v2]
        :return: pose estimate (y position and orientation): [y, phi]
        """

        self.motion_prediction(v, omega, t)
        self.observation_update(left_lane, right_lane)
        self.normalise_particles()
        return self.estimate_pose()

    def motion_prediction(self, v, omega, t):
        """Move particles

        :param v: velocity (metres/second)
        :param omega: angular velocity (radians/second)
        :param t: time (seconds)
        :return:
        """

        d = v * t  # distance, metres
        theta = omega * t  # rotation, radians

        for particle in self.particles:
            # Add noise to "d" and "theta"
            d_noisy = np.random.normal(d, 0.01)  # The second argument is standard deviation in metres
            theta_noisy = np.random.normal(theta, np.pi / 16)  # The second argument is standard deviation in radians

            particle.move(d_noisy, theta_noisy)

    def observation_update(self, left_lane_image, right_lane_image):
        """Update particle weights using observation

        :param left_lane_image: an array of line points for the left lane in the image, e.g. [u1, v1, u2, v2]
        :param right_lane_image: an array of line points for the right lane in the image, e.g. [u1, v1, u2, v2]
        :return:
        """

        if left_lane_image is None and right_lane_image is None:
            # No lanes detected
            return
        elif right_lane_image is None:
            # Only left lane is detected
            left_lane_ground = functions.line_image2ground(left_lane_image)
            left_slope, left_intercept = functions.line_points2slope_intercept(left_lane_ground)

            y_expected = (self.lane_width / 2.) - left_intercept

            left_angle = np.arctan(left_slope)
            phi_expected = -left_angle

        elif left_lane_image is None:
            # Only right lane detected
            right_lane_ground = functions.line_image2ground(right_lane_image)
            right_slope, right_intercept = functions.line_points2slope_intercept(right_lane_ground)

            y_expected = (-self.lane_width / 2.) + right_intercept

            right_angle = np.arctan(right_slope)
            phi_expected = -right_angle

        else:
            # Both lanes are detected
            left_lane_ground = functions.line_image2ground(left_lane_image)
            right_lane_ground = functions.line_image2ground(right_lane_image)

            # Determine Duckiebot pose from observation
            left_slope, left_intercept = functions.line_points2slope_intercept(left_lane_ground)
            right_slope, right_intercept = functions.line_points2slope_intercept(right_lane_ground)

            # Expected y position is the deviation from the centre of the left and right intercepts
            y_expected = -(left_intercept + right_intercept) / 2.

            # Convert slopes to angles
            left_angle = np.arctan(left_slope)
            right_angle = np.arctan(right_slope)

            # Expected angle is the negative of the average of the left and right slopes
            phi_expected = -((left_angle + right_angle) / 2.)

        # Compare the position of each particle with the expected position from the observation to adjust weight
        for particle in self.particles:
            y_diff = np.abs(particle.y - y_expected)
            phi_diff = np.abs(particle.phi - phi_expected)

            # This isn't a good way of determining likelihood, but it seems to be work
            y_likelihood = max(1. - y_diff / 0.5, 0.1)
            phi_likelihood = max(1. - phi_diff / (np.pi / 4.), 0.1)

            particle.weight *= y_likelihood * phi_likelihood

    def normalise_particles(self):
        """Normalise particles"""
        weight_sum = 0.

        for particle in self.particles:
            weight_sum += particle.weight

        for particle in self.particles:
            particle.weight /= weight_sum

    def estimate_pose(self):
        """Estimate pose of the Duckiebot

        :return: pose estimate (y position and orientation): [y, phi]
        """

        y = 0.
        phi = 0.

        # Weighted average
        for particle in self.particles:
            y += particle.y * particle.weight
            phi += particle.phi * particle.weight

        return np.array([y, phi])

    def resample_particles(self):
        """Weighted resampling of particles"""

        old_particles = self.particles

        self.particles = []

        for i in range(0, self.num_particles):
            target = np.random.uniform(0., 1.)
            current = 0.
            p = 0
            while True:
                if target > current and target < current + old_particles[p].weight:
                    # numpy.random.normal is used to add noise
                    particle = Particle(y=np.random.normal(old_particles[p].y, 0.05),
                                        phi=np.random.normal(old_particles[p].phi, np.pi / 8.),
                                        weight=1. / self.num_particles)

                    self.particles.append(particle)

                    break

                current += old_particles[p].weight
                p += 1

                if p > len(old_particles):
                    p = 0
