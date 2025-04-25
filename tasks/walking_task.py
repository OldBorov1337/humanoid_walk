import numpy as np
import transforms3d as tf3
from tasks import rewards

class WalkingTask(object):
    """Динамически стабильная ходьба для двуногого робота."""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=None,
                 root_body: str='PELVIS_S',
                 lfoot_body: str='L_ANKLE_P_S',
                 rfoot_body: str='R_ANKLE_P_S'):
        self._client = client
        self._control_dt = dt
        self._neutral_foot_orient = neutral_foot_orient if neutral_foot_orient is not None else np.array([1,0,0,0])

        # вот этот метод теперь есть в RobotInterface
        self._mass = float(self._client.get_robot_mass())

        # параметры, настраиваются из вне
        self._goal_speed_ref = 0.0
        self._goal_height_ref = 0.0
        self._swing_duration = 0.0
        self._stance_duration = 0.0
        self._total_duration = 0.0

        self._root_body_name = root_body
        self._lfoot_body_name = lfoot_body
        self._rfoot_body_name = rfoot_body

    def calc_reward(self, prev_torque, prev_action, action):
        # скорости и силы
        _, r_vel = self._client.get_rfoot_body_vel()
        _, l_vel = self._client.get_lfoot_body_vel()
        r_frc = self._client.get_rfoot_grf()
        l_frc = self._client.get_lfoot_grf()

        reward = {
            'foot_frc_score':   0.150 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
            'foot_vel_score':   0.150 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),
            'orient_cost':      0.050 * (rewards._calc_body_orient_reward(self, self._lfoot_body_name)
                                        + rewards._calc_body_orient_reward(self, self._rfoot_body_name)
                                        + rewards._calc_body_orient_reward(self, self._root_body_name)) / 3.0,
            'root_accel':       0.050 * rewards._calc_root_accel_reward(self),
            'height_error':     0.050 * rewards._calc_height_reward(self),
            'com_vel_error':    0.200 * rewards._calc_fwd_vel_reward(self),
            'torque_penalty':   0.050 * rewards._calc_torque_reward(self, prev_torque),
            'action_penalty':   0.050 * rewards._calc_action_reward(self, action, prev_action),
        }
        return reward

    def step(self):
        # обновление фазы
        self._phase = (self._phase + 1) % int(2*self._total_duration/self._control_dt)
        return

    def done(self):
        qpos = self._client.get_qpos()
        conds = {
            'height_low':   (qpos[2] < 0.6),
            'height_high':  (qpos[2] > 1.4),
            'self_coll':    self._client.check_self_collisions(),
        }
        return any(conds.values())

    def reset(self, iter_count=0):
        # цель скорости
        self._goal_speed_ref = np.random.choice([0.0, np.random.uniform(0.3, 0.4)])
        # фазы для reward-clock
        self.right_clock, self.left_clock = rewards.create_phase_reward(
            self._swing_duration, self._stance_duration, 0.1, "grounded", 1/self._control_dt
        )
        # период полного цикла (две фазы)
        self._period = int(2*self._total_duration/self._control_dt)
        self._phase = np.random.randint(0, self._period)
