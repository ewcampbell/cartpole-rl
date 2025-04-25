import numpy as np

class CartPoleEnv:
    def __init__(self):
        # Physics constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # Half-length of the pole
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # Seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4  # Cart position at which to fail

        # State: [cart position, cart velocity, pole angle, pole angular velocity]
        self.state = None

    def reset(self):
        # Small random starting position
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state

    def step(self, action):
        """
        action: 0 = push cart to the left, 1 = push cart to the right
        """
        state = self.state
        x, x_dot, theta, theta_dot = state

        # Force based on action
        force = self.force_mag if action == 1 else -self.force_mag

        # Physics update (based on CartPole equations)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Integrate to get new state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])

        # Check if episode is done
        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        # Reward is always 1 for each step survived
        reward = 1.0 if not done else 0.0

        return self.state, reward, done

    def render(self):
        # Very basic render: just print state
        x, x_dot, theta, theta_dot = self.state
        print(f"x: {x:.2f}, x_dot: {x_dot:.2f}, theta: {theta:.2f}, theta_dot: {theta_dot:.2f}")
