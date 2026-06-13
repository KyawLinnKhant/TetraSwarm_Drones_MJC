"""
Adaptive cooperative-lift controller for an UNKNOWN payload.

Like the decentralized transport in Swarm_Drones_ROSCS, the carriers are given
*no* prior knowledge of the payload's mass or geometry. Each drone runs a PD
position law plus a per-drone **adaptive feedforward** w_i that it grows online
(an integral / MRAC-style update) until it cancels whatever steady load it turns
out to be carrying. The w_i therefore converge to each drone's true share, and
their sum recovers the total payload mass — estimated, never told.

    F_i = kp (x* - x_i) - kd v_i ;   F_i,z += w_i
    w_i <- w_i + gamma * (z*_i - z_i) * dt        (adapt to cancel the sag)

    estimated payload mass = (sum_i w_i)/g - N * drone_mass
"""
import numpy as np
import mujoco


class AdaptiveLift:
    def __init__(self, model, n_drones, kp=10.0, kd=6.0, gamma=12.0, mass=0.30,
                 fmax=None):
        self.model, self.n = model, n_drones
        self.kp, self.kd, self.gamma, self.mass, self.fmax = kp, kd, gamma, mass, fmax
        self.g = 9.81
        self.dt = model.opt.timestep
        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"drone{i}")
                         for i in range(n_drones)]
        self.vadr = [model.jnt_dofadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_free")] for i in range(n_drones)]
        # per-drone load estimate (N), starts at 0 -> the swarm knows nothing
        self.w = np.zeros(n_drones)
        self.active = False                       # only adapt once gripped

    def estimated_payload_mass(self):
        return float(self.w.sum() / self.g - self.n * self.mass)

    def share_estimates(self):
        """Per-drone payload-share estimate in kg (drone weight removed)."""
        return self.w / self.g - self.mass

    def apply(self, data, targets):
        for i, bid in enumerate(self.body_ids):
            pos = data.xpos[bid]
            linvel = data.qvel[self.vadr[i]: self.vadr[i] + 3]
            angvel = data.qvel[self.vadr[i] + 3: self.vadr[i] + 6]
            err = targets[i] - pos
            force = self.kp * err - self.kd * linvel
            if self.active:
                # grow the vertical feedforward to cancel the unknown sag
                self.w[i] += self.gamma * err[2] * self.dt
                self.w[i] = max(0.0, self.w[i])
            else:
                self.w[i] = self.mass * self.g    # before grip: just hold itself
            force[2] += self.w[i]
            if self.fmax is not None:
                m = np.linalg.norm(force)
                if m > self.fmax:
                    force *= self.fmax / m
            data.xfrc_applied[bid, :3] = force
            data.xfrc_applied[bid, 3:6] = -0.05 * angvel
