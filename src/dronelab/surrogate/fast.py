"""numba kernel of :meth:`dronelab.surrogate.models.Physics.step` for inference (no gradients).

At MPPI batch sizes (a few hundred rows of 3-vectors) every torch op costs 5-40 us of fixed
overhead, so the physics substeps are overhead-bound. One compiled loop over the batch does
the same arithmetic (same integration order as the torch version) in a fraction of the time.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def step_kernel(pos, vel, rot, omega, rpm, f_res, t_res, kf, km, kd_xy, kd_z, mass, grav,
                inertia, rotor_xy, yaw_sign, n_sub, h):  # pragma: no cover - compiled
    n = pos.shape[0]
    pos_o = np.empty_like(pos)
    vel_o = np.empty_like(vel)
    rot_o = np.empty_like(rot)
    om_o = np.empty_like(omega)
    ix, iy, iz = inertia[0], inertia[1], inertia[2]
    for b in range(n):
        thrust = tx = ty = tz = rsum = 0.0
        for i in range(4):
            w2 = rpm[b, i] * rpm[b, i]
            thrust += w2
            tx += w2 * rotor_xy[i, 1]
            ty -= w2 * rotor_xy[i, 0]
            tz += w2 * yaw_sign[i]
            rsum += rpm[b, i]
        fx, fy, fz = f_res[b, 0], f_res[b, 1], kf * thrust + f_res[b, 2]
        mx, my, mz = kf * tx + t_res[b, 0], kf * ty + t_res[b, 1], km * tz + t_res[b, 2]
        c = 1e-6 * 2.0 * math.pi / 60.0 * rsum
        dxy, dz = c * kd_xy, c * kd_z
        px, py, pz = pos[b, 0], pos[b, 1], pos[b, 2]
        vx, vy, vz = vel[b, 0], vel[b, 1], vel[b, 2]
        wx, wy, wz = omega[b, 0], omega[b, 1], omega[b, 2]
        r00, r01, r02 = rot[b, 0, 0], rot[b, 0, 1], rot[b, 0, 2]
        r10, r11, r12 = rot[b, 1, 0], rot[b, 1, 1], rot[b, 1, 2]
        r20, r21, r22 = rot[b, 2, 0], rot[b, 2, 1], rot[b, 2, 2]
        for _ in range(n_sub):
            # body velocity R^T v, body force, world acceleration R f_b / m - g
            bx = r00 * vx + r10 * vy + r20 * vz
            by = r01 * vx + r11 * vy + r21 * vz
            bz = r02 * vx + r12 * vy + r22 * vz
            gx, gy, gz = fx - dxy * bx, fy - dxy * by, fz - dz * bz
            ax = (r00 * gx + r01 * gy + r02 * gz) / mass
            ay = (r10 * gx + r11 * gy + r12 * gz) / mass
            az = (r20 * gx + r21 * gy + r22 * gz) / mass - grav
            # Euler: (tau - w x (I w)) / I
            cx = wy * iz * wz - wz * iy * wy
            cy = wz * ix * wx - wx * iz * wz
            cz = wx * iy * wy - wy * ix * wx
            vx, vy, vz = vx + h * ax, vy + h * ay, vz + h * az
            wx, wy, wz = wx + h * (mx - cx) / ix, wy + h * (my - cy) / iy, wz + h * (mz - cz) / iz
            px, py, pz = px + h * vx, py + h * vy, pz + h * vz
            # R <- R exp(h w), Rodrigues as in models.so3_exp
            x, y, z = h * wx, h * wy, h * wz
            t = math.sqrt(x * x + y * y + z * z + 1e-12)
            a = math.sin(t) / t
            s = math.sin(0.5 * t) / (0.5 * t)
            q = 0.5 * s * s
            e00, e01, e02 = 1 - q * (y * y + z * z), -a * z + q * x * y, a * y + q * x * z
            e10, e11, e12 = a * z + q * x * y, 1 - q * (x * x + z * z), -a * x + q * y * z
            e20, e21, e22 = -a * y + q * x * z, a * x + q * y * z, 1 - q * (x * x + y * y)
            r00, r01, r02 = (r00 * e00 + r01 * e10 + r02 * e20, r00 * e01 + r01 * e11 + r02 * e21,
                             r00 * e02 + r01 * e12 + r02 * e22)
            r10, r11, r12 = (r10 * e00 + r11 * e10 + r12 * e20, r10 * e01 + r11 * e11 + r12 * e21,
                             r10 * e02 + r11 * e12 + r12 * e22)
            r20, r21, r22 = (r20 * e00 + r21 * e10 + r22 * e20, r20 * e01 + r21 * e11 + r22 * e21,
                             r20 * e02 + r21 * e12 + r22 * e22)
        pos_o[b, 0], pos_o[b, 1], pos_o[b, 2] = px, py, pz
        vel_o[b, 0], vel_o[b, 1], vel_o[b, 2] = vx, vy, vz
        om_o[b, 0], om_o[b, 1], om_o[b, 2] = wx, wy, wz
        rot_o[b, 0, 0], rot_o[b, 0, 1], rot_o[b, 0, 2] = r00, r01, r02
        rot_o[b, 1, 0], rot_o[b, 1, 1], rot_o[b, 1, 2] = r10, r11, r12
        rot_o[b, 2, 0], rot_o[b, 2, 1], rot_o[b, 2, 2] = r20, r21, r22
    return pos_o, vel_o, rot_o, om_o
