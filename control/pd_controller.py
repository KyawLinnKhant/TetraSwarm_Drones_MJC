"""
Force-based PD position controller for the simplified drones.

Each drone is a free body; we command it by writing a 6D wrench to
data.xfrc_applied (force at COM + a small torque to damp rotation and keep it
level). This is the low-level execution layer. The RL formation-keeping policy
will later replace/augment this; for now it gives a reliable baseline and lets
the formation/navigation pipeline run end to end.
"""
import numpy as np
import mujoco


class SwarmPD:
    def __init__(self, model, n_drones, kp=6.0, kd=4.0, kd_ang=0.05, mass=0.30,
                 ff_mass=0.0, fmax=None):
        self.model = model
        self.n = n_drones
        self.kp, self.kd, self.kd_ang, self.mass = kp, kd, kd_ang, mass
        # ff_mass: extra mass each drone gravity-compensates for (e.g. its share
        # of a carried payload). Keeps the PD error small while lifting a load.
        self.ff_mass = ff_mass
        # fmax: optional cap on commanded force magnitude (N). Stops the PD from
        # ramming a drone through a wall when its target is on the far side.
        self.fmax = fmax
        self.g = 9.81
        self.body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"drone{i}")
            for i in range(n_drones)
        ]
        self.qadr = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_free")]
            for i in range(n_drones)
        ]
        self.vadr = [
            model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_free")]
            for i in range(n_drones)
        ]

    def positions(self, data):
        return np.array([data.xpos[bid].copy() for bid in self.body_ids])

    def apply(self, data, targets):
        """targets: (n_drones, 3) desired world positions."""
        for i, bid in enumerate(self.body_ids):
            pos = data.xpos[bid]
            linvel = data.qvel[self.vadr[i]: self.vadr[i] + 3]      # world-frame linear vel
            angvel = data.qvel[self.vadr[i] + 3: self.vadr[i] + 6]  # local angular vel
            force = self.kp * (targets[i] - pos) - self.kd * linvel
            force[2] += (self.mass + self.ff_mass) * self.g         # gravity + payload comp
            if self.fmax is not None:
                mag = np.linalg.norm(force)
                if mag > self.fmax:
                    force = force * (self.fmax / mag)
            data.xfrc_applied[bid, :3] = force
            data.xfrc_applied[bid, 3:6] = -self.kd_ang * angvel     # keep level
