"""Pogodude Environment"""

from os import path

from environments.utils import (
    np,
    Box,
    MujocoEnv,
    BodyData,
    Vector3,
)


class PogoEnv(MujocoEnv):
    """
    # Mujoco Pogo Environment

    Pogodude is a 2D robot consisting of a head/body and upper/lower arm and
    leg segments attached to a pogo stick. Motors are placed at the hip, knee,
    shoulder, and elbow joints to control the robot's movement. The goal is
    to train the robot to jump as high as possible.

    NOTE: The 2D pogodude is constrained to the X-Z plane.


    ## Action Space

    The action space is a `Box(-1, 1, (4,), float32)`.
    All actuators are continuous hinge joints, have a range of `[-1, 1]`, and
    are measured in torque.

        0.  Torque applied to the hip ................... (N m)
        1.  Torque applied to the knee .................. (N m)
        2.  Torque applied to the shoulder .............. (N m)
        3.  Torque applied to the elbow ................. (N m)


    ## Observation Space

    The observation space has a size of 22.
    Observations consist of positional and velocitous values for each joint, and
    have a range of `(-inf, inf)`.

        0.  y-orientation of the body ................... (deg)
	    1.  x-coordinate of the body ...................... (m)
	    2.  z-coordinate of the body ...................... (m)
	    3.  angle of hip joint .......................... (deg)
	    4.  angle of knee joint ......................... (deg)
	    5.  angle of shoulder joint ..................... (deg)
	    6.  angle of elbow joint ........................ (deg)
	    7.  y-orientation of the pogo stick ............. (deg)
	    8.  x-coordinate of the pogo stick ................ (m)
	    9.  z-coordinate of the pogo stick ................ (m)
        10. rel. displacement of the rod spring joint ..... (m)
        11. y-angular-velocity of the body ............ (deg/s)
	    12. x-velocity of the body ...................... (m/s)
	    13. z-velocity of the body ...................... (m/s)
        14. angular velocity of hip joint ............. (deg/s)
	    15. angular velocity of knee joint ............ (deg/s)
	    16. angular velocity of shoulder joint ........ (deg/s)
	    17. angular velocity of elbow joint ........... (deg/s)
	    18. y-angular-velocity of the pogo stick ...... (deg/s)
	    19. x-velocity of the pogo stick ................ (m/s)
	    20. z-velocity of the pogo stick ................ (m/s)
        21. rel. velocity of the rod spring joint ....... (m/s)
	

    ## Utility/Reward Function

    Since the goal of the environment is to jump as high as possible, the reward
    function is designed to encourage the robot to jump higher. A few components
    make up the reward function:

        - *height_reward*: The reward for existing at a higher z-coordinate,
          which is calculated as `height_reward_weight * z-coordinate`.
        - *jump_reward*: The reward for jumping higher, which is calculated as
          `jump_reward_weight * clamp(z-velocity, 0, inf)`.
        - *control_cost*: A penalty for applying large forces to the motors,
          which is calculated as `control_cost_weight * sum(action)`.
        - *healthy_reward*: A constant reward for remaining upright, which is
          calculated as `healthy_reward_weight * is_alive`.

    `reward = height_reward + jump_reward + healthy_reward - control_cost`

    """

    #Declare hyperparameters
    height_reward_weight: float
    jump_reward_weight: float
    control_cost_weight: float
    healthy_reward_weight: float
    hrw_exp: float
    jrw_exp: float
    ccw_exp: float

    # Declare body parts
    _body: BodyData
    _head: BodyData
    _upper_arm: BodyData
    _lower_arm: BodyData
    _upper_leg: BodyData
    _lower_leg: BodyData
    _pogo_body: BodyData

    # Render configuration
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100
    }
    _default_camera_config = {
        "distance": 8.0
    }

    # RL environment configuration
    action_space: Box
    _reset_noise_scale = 0.1
    _episode_length = 1000
    _frame_skip = 5
    _obs_shape = 22
    _obs_space: Box

    # State variables
    _action: np.ndarray
    _position: Vector3
    _velocity: Vector3
    _step_num: int


    # Helpers and overrides

    def _get_custom_param(self, name: str, default: float | int) -> float | int:
        """Helper method to get cusom parameters from the MuJoCo model"""
        try:
            return self.model.numeric(name).data[0]
        except ValueError:
            return default

    def _set_action_space(self):
        """
        Manually setting the action space to be in the range [-1, 1] for
        all actuators; helps the performance of many RL algorithms
        """
        num_actuators = self.model.nu
        self.action_space = Box(
            low=-1, high=1, shape=(num_actuators,), dtype=np.float32
        )
        return self.action_space

    @property
    def _obs(self) -> np.ndarray:
        """
        Agent is allowed to sense positional and velocitous values for each
        degree of freedom across its joints, including global and pogo joints
        """
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        return np.concatenate((position, velocity)).astype(np.float32)

    @property
    def _should_terminate(self) -> bool:
        """Check if the episode should be terminated"""
        return not bool(np.isfinite(self._obs).all())

    @property
    def _should_truncate(self) -> bool:
        """Check if the episode should be truncated"""
        return self._step_num >= self._episode_length

    def _record_sim_step(self):
        """
        Apply simulation action and record resulting state values.
        Updates the state variables `_position`, `_velocity`, and `_step_num`.
        """
        prev_pos = self._head.com
        self.do_simulation(self._action, self.frame_skip)
        next_pos = self._head.com

        self._position = next_pos
        self._velocity = Vector3(
            (next_pos.x - prev_pos.x) / self.dt, 0,
            (next_pos.z - prev_pos.z) / self.dt
        )

        self._step_num += 1

    # Reward function helpers
    @property
    def _height_reward(self) -> float:
        """Reward for existing at a higher z-coordinate"""
        return (self.height_reward_weight * self._position.z) ** self.hrw_exp

    @property
    def _jump_reward(self) -> float:
        """Reward for jumping higher"""
        return (self.jump_reward_weight * max(0, self._velocity.z)) ** self.jrw_exp

    @property
    def _control_cost(self) -> float:
        """Penalty for applying large forces to the motors"""
        return (self.control_cost_weight * np.sum(np.square(self._action))) ** self.ccw_exp

    @property
    def _is_healthy(self) -> bool:
        """
        Check if the agent is healthy.
        Healthy means no fragile bodies touch the ground (besides pogo rod).
        """
        # Define which parts are fragile (can't touch the ground)
        fragile_parts = [
            self._body,
            self._head,
            self._upper_arm,
            self._lower_arm,
            self._upper_leg,
            self._lower_leg,
            self._pogo_body
        ]

        fragile_part_ids = list(map(lambda p: p.geom.id, fragile_parts))
        fragile_contacts = filter(
            lambda c: c.geom1 in fragile_part_ids or c.geom2 in fragile_part_ids,
            self.data.contact
        )

        return not any(fragile_contacts)

    @property
    def _healthy_reward(self) -> float:
        """Constant reward for remaining upright"""
        return self.healthy_reward_weight * float(self._is_healthy)

    @property
    def _reward(self) -> float:
        """Total reward for the agent"""
        return (
            self._height_reward
            + self._jump_reward
            + self._healthy_reward
            - self._control_cost
        )


    # Main environment methods
    def __init__(self, xml_version: str, **kwargs):
        # Determine XML file path
        major_version = xml_version.split(".")[0]
        xml_fname = path.join(path.dirname(__file__), f"pogodude_{major_version}.xml")

        # Define observation space
        self._obs_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_shape,),
            dtype=np.float32
        )

        # Initialize Mujoco environment
        super().__init__(
            model_path=xml_fname,
            frame_skip=self._frame_skip,
            observation_space=self._obs_space,
            default_camera_config=self._default_camera_config,
            **kwargs
        )

        # Initialize body part tracking
        self._body = BodyData(self, "body")
        self._head = BodyData(self, "head")
        self._upper_arm = BodyData(self, "upper_arm")
        self._lower_arm = BodyData(self, "lower_arm")
        self._upper_leg = BodyData(self, "upper_leg")
        self._lower_leg = BodyData(self, "lower_leg")
        self._pogo_body = BodyData(self, "pogo_body")

        self.height_reward_weight  = self._get_custom_param("height_reward_weight",  1.0)
        self.jump_reward_weight    = self._get_custom_param("jump_reward_weight",    1.0)
        self.control_cost_weight   = self._get_custom_param("control_cost_weight",   0.1)
        self.healthy_reward_weight = self._get_custom_param("healthy_reward_weight", 1.0)
        self.hrw_exp = self._get_custom_param("hrw_exp", 1)
        self.jrw_exp = self._get_custom_param("jrw_exp", 1)
        self.ccw_exp = self._get_custom_param("ccw_exp", 1)

    def step(self, action):
        """Update the simulation based on the observation"""
        self._action = action

        terminated = self._should_terminate
        if not terminated:
            self._record_sim_step()

        return (
            self._obs,
            self._reward,
            terminated,
            self._should_truncate,
            {"height": self._position.z}
        )

    def reset_model(self):
        """Reset the model to its initial state, applying noise"""
        qpos = self.init_qpos + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq
        ).astype(np.float32)
        qvel = (
            self.init_qvel
            + self._reset_noise_scale
            * self.np_random.standard_normal(self.model.nv)
        ).astype(np.float32)

        self.set_state(qpos, qvel)
        self._step_num = 0

        return self._obs
