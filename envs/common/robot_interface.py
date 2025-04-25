import os
import numpy as np
import transforms3d as tf3
import mujoco
from mujoco import mj_name2id, mjtObj, mj_contactForce
import torch
import collections

class RobotInterface:
    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 rfoot_body_name: str,
                 lfoot_body_name: str,
                 path_to_nets: str = None):
        self.model = model
        self.data = data

        # Ступни и пол
        self.rfoot_body_name = rfoot_body_name
        self.lfoot_body_name = lfoot_body_name

        # floor body по геометрию "floor"
        floor_geom_id = mj_name2id(self.model, mjtObj.mjOBJ_GEOM, 'floor')
        if floor_geom_id >= 0:
            body_id = self.model.geom_bodyid[floor_geom_id]
            self.floor_body_name = mujoco.mj_id2name(self.model, mjtObj.mjOBJ_BODY, body_id)
        else:
            self.floor_body_name = None

        # root body по freejoint "root"
        root_jid = mj_name2id(self.model, mjtObj.mjOBJ_JOINT, 'root')
        if root_jid >= 0:
            root_bid = self.model.jnt_bodyid[root_jid]
            self.robot_root_name = mujoco.mj_id2name(self.model, mjtObj.mjOBJ_BODY, root_bid)
        else:
            self.robot_root_name = None

        self.stepCounter = 0

        # motor nets
        if path_to_nets:
            self.load_motor_nets(path_to_nets)

    def load_motor_nets(self, path_to_nets: str):
        self.motor_dyn_nets = {}
        for jnt in os.listdir(path_to_nets):
            net_dir = os.path.join(path_to_nets, jnt)
            if not os.path.isdir(net_dir):
                continue
            net = torch.jit.load(os.path.join(net_dir, "trained_jit.pth"))
            net.eval()
            self.motor_dyn_nets[jnt] = net.double()
        self.ctau_buffer = collections.deque(maxlen=25)
        self.qdot_buffer = collections.deque(maxlen=25)

    # ===============================
    # — базовые методы доступа
    # ===============================

    def get_robot_mass(self) -> float:
        """Total robot mass."""
        return mujoco.mj_getTotalmass(self.model)

    def get_qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    def get_qvel(self) -> np.ndarray:
        return self.data.qvel.copy()

    def nq(self) -> int:
        return self.model.nq

    def nv(self) -> int:
        return self.model.nv

    def nu(self) -> int:
        return self.model.nu

    def sim_dt(self) -> float:
        return self.model.opt.timestep

    # joint lookups
    def get_jnt_id_by_name(self, name: str) -> int:
        return mj_name2id(self.model, mjtObj.mjOBJ_JOINT, name)

    def get_jnt_qposadr_by_name(self, name: str) -> int:
        jid = self.get_jnt_id_by_name(name)
        return self.model.jnt_qposadr[jid] if jid>=0 else None

    def get_jnt_qveladr_by_name(self, name: str) -> int:
        jid = self.get_jnt_id_by_name(name)
        return self.model.jnt_dofadr[jid] if jid>=0 else None

    # root pos/vel
    def get_root_body_pos(self) -> np.ndarray:
        bid = self.model.body_name2id(self.robot_root_name)
        return self.data.xpos[bid].copy()

    def get_root_body_vel(self) -> np.ndarray:
        adr = self.get_jnt_qveladr_by_name('root')
        return self.data.qvel[adr:adr+6].copy()

    # motor actuator state
    def get_motor_positions(self):
        return self.data.actuator_length.copy()

    def get_motor_velocities(self):
        return self.data.actuator_velocity.copy()

    def get_act_joint_positions(self):
        gear = self.model.actuator_gear[:,0]
        return self.data.actuator_length / gear

    def get_act_joint_velocities(self):
        gear = self.model.actuator_gear[:,0]
        return self.data.actuator_velocity / gear

    # ===============================
    # — контакты ступня–пол
    # ===============================
    def get_rfoot_floor_contacts(self):
        contacts = []
        if self.floor_body_name is None:
            return contacts
        floor_bid = self.model.body_name2id(self.floor_body_name)
        foot_bid = self.model.body_name2id(self.rfoot_body_name)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            # один из них — ступня, другой — пол
            if (b1==foot_bid and self.model.geom_bodyid[c.geom2]==floor_bid) or \
               (b2==foot_bid and self.model.geom_bodyid[c.geom1]==floor_bid):
                contacts.append((i, c))
        return contacts

    def get_lfoot_floor_contacts(self):
        contacts = []
        if self.floor_body_name is None:
            return contacts
        floor_bid = self.model.body_name2id(self.floor_body_name)
        foot_bid = self.model.body_name2id(self.lfoot_body_name)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if (b1==foot_bid and self.model.geom_bodyid[c.geom2]==floor_bid) or \
               (b2==foot_bid and self.model.geom_bodyid[c.geom1]==floor_bid):
                contacts.append((i, c))
        return contacts

    def get_rfoot_grf(self) -> float:
        grf = 0.0
        for i, _ in self.get_rfoot_floor_contacts():
            arr = np.zeros(6, dtype=np.float64)
            mj_contactForce(self.model, self.data, i, arr)
            grf += np.linalg.norm(arr)
        return grf

    def get_lfoot_grf(self) -> float:
        grf = 0.0
        for i, _ in self.get_lfoot_floor_contacts():
            arr = np.zeros(6, dtype=np.float64)
            mj_contactForce(self.model, self.data, i, arr)
            grf += np.linalg.norm(arr)
        return grf

    # ===============================
    # — скорость ступни в body frame
    # ===============================
    def get_rfoot_body_vel(self, frame=0):
        return self.get_body_vel(self.rfoot_body_name, frame)

    def get_lfoot_body_vel(self, frame=0):
        return self.get_body_vel(self.lfoot_body_name, frame)

    def get_body_vel(self, body_name: str, frame=0):
        body_id = mj_name2id(self.model, mjtObj.mjOBJ_BODY, body_name)
        buf = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mjtObj.mjOBJ_BODY, body_id, buf, frame)
        # возвращаем (linvel, angvel)
        return buf[:3].copy(), buf[3:].copy()

    # ===============================
    # — проверки коллизий
    # ===============================
    def check_self_collisions(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.body(self.model.geom_bodyid[c.geom1]).rootid
            b2 = self.model.body(self.model.geom_bodyid[c.geom2]).rootid
            if b1==b2==self.model.body_name2id(self.robot_root_name):
                return True
        return False

    # ===============================
    # — управление моторами через ctrl
    # ===============================
    def set_motor_torque(self, torque: np.ndarray, motor_dyn_fwd=False):
        ctrl = torque.copy()
        if motor_dyn_fwd:
            raise NotImplementedError("Motor nets in this build")
        np.copyto(self.data.ctrl, ctrl)
