import os
import numpy as np
import transforms3d as tf3
import collections
import tempfile

from .gen_xml import builder, LEG_JOINTS
from envs.common.mujoco_env import MujocoEnv
from envs.common.robot_interface import RobotInterface
from envs.common.config_builder import load_yaml
from tasks.walking_task import WalkingTask
from robots.robot_base import RobotBase

class JvrcWalkEnv(MujocoEnv):
    def __init__(self, path_to_yaml: str = None):
        # 1) Load CONFIG from yaml
        if path_to_yaml is None:
            path_to_yaml = os.path.join(
                os.path.dirname(__file__), 'configs', 'base.yaml'
            )
        self.cfg = load_yaml(path_to_yaml)
        sim_dt     = self.cfg.sim_dt
        control_dt = self.cfg.control_dt

        # 2) Generate XML-пакет (если ещё нет)
        export_dir = os.path.join(tempfile.gettempdir(), 'mjcf-export', 'jvrc_walk')
        xml_filename = 'jvrc.xml'
        if not os.path.exists(os.path.join(export_dir, xml_filename)):
            os.makedirs(export_dir, exist_ok=True)
            builder(export_dir, config={})

        # 3) Инициализируем базовый MujocoEnv,
        # передаём каталог с описанием робота, а не путь к файлу
        super().__init__(export_dir, sim_dt, control_dt)

        # 4) PD gains
        pdgains = np.vstack([self.cfg.kp, self.cfg.kd])

        # 5) Actuators
        self.actuators = LEG_JOINTS

        # 6) Nominal pose
        base_pos  = [0, 0, 0.81]
        base_quat = [1, 0, 0, 0]
        half_deg  = [-30, 0, 0,  50, 0, -24,
                     -30, 0, 0,  50, 0, -24]
        half_pose = np.deg2rad(half_deg).tolist()
        self.nominal_pose = base_pos + base_quat + half_pose

        # 7) RobotInterface
        self.interface = RobotInterface(
            self.model, self.data,
            rfoot_body_name='R_ANKLE_P_S',
            lfoot_body_name='L_ANKLE_P_S'
        )

        # 8) WalkingTask
        self.task = WalkingTask(
            client=self.interface,
            dt=control_dt,
            neutral_foot_orient=np.array([1,0,0,0]),
            root_body=self.interface.robot_root_name,
            lfoot_body=self.interface.lfoot_body_name,
            rfoot_body=self.interface.rfoot_body_name
        )
        self.task._goal_height_ref = 0.80
        self.task._total_duration  = 1.1
        self.task._swing_duration  = 0.75
        self.task._stance_duration = 0.35

        # 8.1) Загрузка плана шагов
        scene_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'scene')
        )
        foot_file = os.path.join(scene_dir, 'footstep_plans.txt')
        import io
        with open(foot_file, 'r', encoding='utf-8') as f:
            lines = [l for l in f if not l.strip().startswith('---')]
        data = io.StringIO(''.join(lines))
        self.footsteps = np.loadtxt(data, delimiter=',')
        if hasattr(self.task, 'load_footsteps'):
            self.task.load_footsteps(self.footsteps)

        # 9) RobotBase
        self.robot = RobotBase(pdgains, control_dt, self.interface, self.task)

        # 10) Mirror indices, action/obs spaces
        base_mir_obs = [-0.1, 1,
                        -2, 3, -4,
                        11, -12, -13, 14, -15, 16,
                         5,  -6,  -7,  8,  -9, 10,
                        23, -24, -25, 26, -27, 28,
                        17, -18, -19, 20, -21, 22]
        append_obs = list(range(len(base_mir_obs), len(base_mir_obs) + 3))
        self.robot.clock_inds    = append_obs[0:2]
        self.robot.mirrored_obs  = base_mir_obs + append_obs
        self.robot.mirrored_acts = [6, -7, -8, 9, -10, 11,
                                   0, -1, -2, 3, -4, 5]

        action_size = len(self.actuators)
        self.action_space      = np.zeros(action_size)
        self.prev_prediction   = np.zeros(action_size)
        self.base_obs_len      = 32
        self.history_len       = self.cfg.obs_history_len
        self.observation_space = np.zeros(self.base_obs_len * self.history_len)
        self.observation_history = collections.deque(maxlen=self.history_len)

        obs_mean = np.concatenate((
            np.zeros(5),
            np.deg2rad(half_deg),
            np.zeros(12),
            [0.5, 0.5, 0.5]
        ))
        obs_std = np.concatenate((
            [0.2,0.2,1,1,1],
            0.5*np.ones(12),
            4*np.ones(12),
            [1,1,1]
        ))
        self.obs_mean = np.tile(obs_mean, self.history_len)
        self.obs_std  = np.tile(obs_std,  self.history_len)
    
    # остальной код get_obs, step, reset_model — без изменений  

    def get_obs(self):
        phase = self.task._phase / self.task._period
        clock = [np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)]
        ext_state = np.concatenate((clock, [self.task._goal_speed_ref]))

        qpos = self.interface.get_qpos()
        qvel = self.interface.get_qvel()
        roll, pitch, _ = tf3.euler.quat2euler(qpos[3:7])
        root_ang_vel   = qvel[3:6]
        motor_pos      = self.interface.get_act_joint_positions()
        motor_vel      = self.interface.get_act_joint_velocities()

        robot_state = np.concatenate((
            [roll], [pitch], root_ang_vel,
            motor_pos, motor_vel
        ))
        state = np.concatenate((robot_state, ext_state))
        assert state.shape == (self.base_obs_len,)

        if not self.observation_history:
            for _ in range(self.history_len):
                self.observation_history.appendleft(np.zeros_like(state))
        self.observation_history.appendleft(state)
        return np.array(self.observation_history).flatten()

    def step(self, action):
        targets = ( self.cfg.action_smoothing * action +
                   (1 - self.cfg.action_smoothing) * self.prev_prediction )
        offsets = [
            self.nominal_pose[self.interface.get_jnt_qposadr_by_name(j)[0]]
            for j in self.actuators
        ]
        rewards, done = self.robot.step(targets, np.asarray(offsets))
        obs = self.get_obs()
        self.prev_prediction = action.copy()
        return obs, sum(rewards.values()), done, rewards

    def reset_model(self):
        init_qpos = np.array(self.nominal_pose)
        init_qvel = np.zeros(self.interface.nv())
        self.set_state(init_qpos, init_qvel)
        self.task.reset(iter_count=self.robot.iteration_count)
        self.prev_prediction = np.zeros_like(self.prev_prediction)
        self.observation_history.clear()
        return self.get_obs()
