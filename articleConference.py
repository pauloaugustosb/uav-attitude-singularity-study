#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quadrotor (reduced) simulation with:
A) Coupled 3D translation + attitude (Euler vs Quaternion) under PD control
B) Explicit Euler kinematic singularity experiments (attitude-only), now with TRUE quaternions:
   - pitch tracking near singularity
   - angular-rate / control-effort blow-up (log scale + zoom/inset)
   - forced crossing of singularity (Euler vs quaternion)
All figures are saved as EPS (and PNG) to OUT_DIR.

Author: Paulo Augusto Silva Borges et al.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# =========================================================
# OUTPUT
# =========================================================
OUT_DIR = "/home/paulo/Documentos/Academico/Rogerio_conference/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(base_name: str):
    """
    Save both EPS and PNG with tight layout.
    """
    png = os.path.join(OUT_DIR, f"{base_name}.png")
    eps = os.path.join(OUT_DIR, f"{base_name}.eps")
    plt.tight_layout()
    plt.savefig(png, dpi=250)
    plt.savefig(eps, format="eps")
    print(f"[saved] {png}")
    print(f"[saved] {eps}")

# =========================================================
# NUMERICS
# =========================================================
def sat(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

# =========================================================
# ROTATION: EULER (ZYX) <-> R
# =========================================================
def R_from_euler(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    # ZYX: R = Rz(psi) Ry(theta) Rx(phi)
    R = np.array([
        [cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi],
        [spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi],
        [-sth,     cth*sphi,                  cth*cphi]
    ])
    return R

def euler_from_R(R):
    # ZYX extraction (robust-ish)
    # theta = -asin(R[2,0])
    theta = np.arcsin(sat(-R[2, 0], -1.0, 1.0))
    cth = np.cos(theta)

    if abs(cth) < 1e-9:
        # Near singular: choose psi = 0, solve phi from R[0,1],R[1,1]
        psi = 0.0
        phi = np.arctan2(R[0, 1], R[1, 1])
    else:
        phi = np.arctan2(R[2, 1], R[2, 2])
        psi = np.arctan2(R[1, 0], R[0, 0])

    return phi, theta, psi

def E_matrix(phi, theta):
    """
    Maps body rates omega = [p,q,r] to Euler angle rates:
        [phi_dot, theta_dot, psi_dot]^T = E(phi,theta) * omega
    ZYX Euler has singularity at cos(theta)=0.
    """
    sphi, cphi = np.sin(phi), np.cos(phi)
    tth = np.tan(theta)
    cth = np.cos(theta)

    # Guard: keep cth away from 0 to avoid NaNs (for integration stability).
    # We DO NOT "fix" the singularity; we just prevent hard division-by-zero.
    if abs(cth) < 1e-9:
        cth = np.sign(cth) * 1e-9 if cth != 0 else 1e-9

    E = np.array([
        [1.0, sphi*tth, cphi*tth],
        [0.0, cphi,     -sphi],
        [0.0, sphi/cth, cphi/cth]
    ])
    return E

# =========================================================
# ROTATION: QUATERNIONS (true)
# Convention: q = [qw, qx, qy, qz] with qw scalar part.
# =========================================================
def q_norm(q):
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ])

def R_from_q(q):
    q = q_norm(q)
    qw, qx, qy, qz = q

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),         1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),         2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
    ])
    return R

def q_from_R(R):
    # Robust quaternion from rotation matrix (trace method)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    return q_norm(np.array([qw, qx, qy, qz]))

def Omega_mat(omega):
    # omega = [p,q,r] (body rates)
    p, q, r = omega
    return np.array([
        [0.0, -p,  -q,  -r],
        [p,   0.0, r,   -q],
        [q,   -r,  0.0, p],
        [r,   q,   -p,  0.0]
    ])

def euler_from_q(q):
    # Convert via R then extract ZYX
    R = R_from_q(q)
    return euler_from_R(R)

# =========================================================
# SIMPLE QUADROTOR MODEL (reduced rigid-body)
# States:
#   p in R3, v in R3
#   attitude: Euler angles OR quaternion
#   omega in R3
# Dynamics:
#   p_dot = v
#   v_dot = g*e3 + (T/m) * R * e3 + disturbances
#   omega_dot = I^{-1} (tau - omega x I omega)
#   attitude kinematics:
#       euler_dot = E(phi,theta) omega
#       q_dot = 0.5 * Omega(omega) q
# =========================================================
class Params:
    def __init__(self):
        self.m = 1.0
        self.g = 9.81
        self.I = np.diag([0.02, 0.02, 0.04])

        # Limits (keep simulation sane)
        self.T_min = 0.0
        self.T_max = 2.5 * self.m * self.g
        self.tau_max = np.array([1.5, 1.5, 1.0])  # N*m per axis

P = Params()

# =========================================================
# CONTROLLERS (PD) - OUTER (position) + INNER (attitude)
# =========================================================
def desired_attitude_from_acc(a_des, yaw_des=0.0):
    """
    Given desired world-frame acceleration a_des, design:
    - desired thrust direction b3_des aligned with (a_des - g e3)
    - desired yaw set to yaw_des
    Returns (R_des, T_des) ignoring rotor allocation.
    """
    e3 = np.array([0.0, 0.0, 1.0])

    # Required total acceleration including gravity compensation
    a_total = a_des + np.array([0.0, 0.0, P.g])

    # Desired body z axis in world:
    norm_a = np.linalg.norm(a_total)
    if norm_a < 1e-9:
        b3 = e3.copy()
        T = P.m * P.g
    else:
        b3 = a_total / norm_a
        T = P.m * norm_a

    # Desired yaw: build b1_des in xy-plane
    cpsi, spsi = np.cos(yaw_des), np.sin(yaw_des)
    b1_ref = np.array([cpsi, spsi, 0.0])

    # b2 = b3 x b1_ref (then normalize), b1 = b2 x b3
    b2 = np.cross(b3, b1_ref)
    nb2 = np.linalg.norm(b2)
    if nb2 < 1e-9:
        # fallback
        b2 = np.array([0.0, 1.0, 0.0])
    else:
        b2 = b2 / nb2

    b1 = np.cross(b2, b3)

    R_des = np.column_stack((b1, b2, b3))
    return R_des, T

def pd_position(p, v, p_ref, v_ref, Kp, Kd):
    return Kp*(p_ref - p) + Kd*(v_ref - v)

def pd_attitude_euler(phi_theta_psi, omega, phi_theta_psi_ref, Kp, Kd):
    e = phi_theta_psi_ref - phi_theta_psi
    # wrap yaw error to [-pi, pi] (optional)
    e[2] = (e[2] + np.pi) % (2*np.pi) - np.pi
    tau = Kp*e - Kd*omega
    return tau

def pd_attitude_quat(q, omega, q_des, Kp, Kd):
    """
    Quaternion attitude PD:
      q_err = q_des^{-1} ⊗ q   (or q_des*conj(q), both used in literature with sign conventions)
    We'll use: q_err = conj(q_des) ⊗ q
    Control uses vector part with shortest rotation (enforce qw >= 0).
    """
    q_err = q_mul(q_conj(q_des), q)
    q_err = q_norm(q_err)
    if q_err[0] < 0:
        q_err = -q_err

    e_vec = q_err[1:4]  # vector part
    tau = -Kp*e_vec - Kd*omega
    return tau

# =========================================================
# EXPERIMENT 0 — COUPLED 3D TRAJECTORY (Euler vs Quaternion)
# =========================================================
def run_coupled_3d(dt=0.002, T_end=8.0, seed=2):
    t = np.arange(0.0, T_end, dt)
    N = len(t)

    # Reference: smooth ramp to (1,1,1) then hold
    p_ref = np.zeros((N, 3))
    v_ref = np.zeros((N, 3))
    for k in range(N):
        if t[k] < 1.0:
            p_ref[k] = np.array([0.0, 0.0, 0.0])
            v_ref[k] = np.array([0.0, 0.0, 0.0])
        else:
            # line trajectory: 1m in x,y,z over ~4s, then hold
            s = sat((t[k]-1.0)/4.0, 0.0, 1.0)
            p_ref[k] = np.array([1.0, 1.0, 1.0]) * s
            v_ref[k] = np.array([1.0, 1.0, 1.0]) * (1.0/4.0 if (1.0 < t[k] < 5.0) else 0.0)

    # Disturbance (wind + gust)
    rng = np.random.default_rng(seed)
    wind = 0.10 * rng.standard_normal((N, 3))
    gust = np.zeros((N, 3))
    gust[(t > 3.0) & (t < 3.5)] = np.array([1.0, -0.6, 0.3])
    d_ext = wind + gust

    # Gains (tune mild, stable)
    Kp_pos = np.diag([2.0, 2.0, 3.0])
    Kd_pos = np.diag([1.6, 1.6, 2.2])

    Kp_att_e = np.array([8.0, 8.0, 4.0])
    Kd_att_e = np.array([2.0, 2.0, 1.5])

    Kp_att_q = 10.0
    Kd_att_q = 2.5

    # States Euler
    pE = np.zeros((N, 3))
    vE = np.zeros((N, 3))
    angE = np.zeros((N, 3))     # [phi,theta,psi]
    wE = np.zeros((N, 3))

    # States Quaternion
    pQ = np.zeros((N, 3))
    vQ = np.zeros((N, 3))
    qQ = np.zeros((N, 4))
    qQ[0] = np.array([1.0, 0.0, 0.0, 0.0])
    wQ = np.zeros((N, 3))

    e3 = np.array([0.0, 0.0, 1.0])

    for k in range(1, N):
        # ------------------------
        # OUTER LOOP (same design)
        # ------------------------
        a_des_E = pd_position(pE[k-1], vE[k-1], p_ref[k], v_ref[k], Kp_pos, Kd_pos)
        a_des_Q = pd_position(pQ[k-1], vQ[k-1], p_ref[k], v_ref[k], Kp_pos, Kd_pos)

        # Desired attitude from acceleration
        Rdes_E, Tdes_E = desired_attitude_from_acc(a_des_E, yaw_des=0.0)
        Rdes_Q, Tdes_Q = desired_attitude_from_acc(a_des_Q, yaw_des=0.0)

        # Clamp thrust
        Tdes_E = float(sat(Tdes_E, P.T_min, P.T_max))
        Tdes_Q = float(sat(Tdes_Q, P.T_min, P.T_max))

        # Convert desired to Euler / Quaternion
        ang_ref_E = np.array(euler_from_R(Rdes_E))
        q_des = q_from_R(Rdes_Q)

        # ------------------------
        # INNER LOOP (attitude PD)
        # ------------------------
        tauE = pd_attitude_euler(angE[k-1], wE[k-1], ang_ref_E,
                                 Kp_att_e, Kd_att_e)
        tauQ = pd_attitude_quat(qQ[k-1], wQ[k-1], q_des,
                                Kp_att_q, Kd_att_q)

        # Saturate torques
        tauE = sat(tauE, -P.tau_max, P.tau_max)
        tauQ = sat(tauQ, -P.tau_max, P.tau_max)

        # ------------------------
        # DYNAMICS UPDATE (Euler)
        # ------------------------
        RE = R_from_euler(*angE[k-1])
        aE = np.array([0.0, 0.0, -P.g]) + (Tdes_E / P.m) * (RE @ e3) + d_ext[k] / P.m
        vE[k] = vE[k-1] + aE * dt
        pE[k] = pE[k-1] + vE[k] * dt

        wdotE = np.linalg.inv(P.I) @ (tauE - np.cross(wE[k-1], P.I @ wE[k-1]))
        wE[k] = wE[k-1] + wdotE * dt

        Ed = E_matrix(angE[k-1][0], angE[k-1][1])
        angdotE = Ed @ wE[k]
        angE[k] = angE[k-1] + angdotE * dt

        # ------------------------
        # DYNAMICS UPDATE (Quat)
        # ------------------------
        RQ = R_from_q(qQ[k-1])
        aQ = np.array([0.0, 0.0, -P.g]) + (Tdes_Q / P.m) * (RQ @ e3) + d_ext[k] / P.m
        vQ[k] = vQ[k-1] + aQ * dt
        pQ[k] = pQ[k-1] + vQ[k] * dt

        wdotQ = np.linalg.inv(P.I) @ (tauQ - np.cross(wQ[k-1], P.I @ wQ[k-1]))
        wQ[k] = wQ[k-1] + wdotQ * dt

        qdot = 0.5 * (Omega_mat(wQ[k]) @ qQ[k-1])
        qQ[k] = q_norm(qQ[k-1] + qdot * dt)

    # Outputs
    posnorm_E = np.linalg.norm(pE, axis=1)
    posnorm_Q = np.linalg.norm(pQ, axis=1)
    posnorm_ref = np.linalg.norm(p_ref, axis=1)

    return t, p_ref, pE, pQ, posnorm_ref, posnorm_E, posnorm_Q

# =========================================================
# EXPERIMENT 1 — POSITION TRACKING (norm) + disturbances
# =========================================================
# def plot_position_tracking_norm(t, posnorm_ref, posnorm_E, posnorm_Q):
#     plt.figure(figsize=(9, 5))
#     plt.plot(t, posnorm_E, label="Euler Angles", linewidth=2)
#     plt.plot(t, posnorm_Q, label="Quaternion", linewidth=2)
#     plt.plot(t, posnorm_ref, "--", label="Reference", linewidth=2)
#     plt.xlabel("Time [s]")
#     plt.ylabel(r"$\Vert p \Vert$  [m]")
#     plt.title("Position Tracking (Norm) under PD Control with Disturbances")
#     plt.grid(True)
#     plt.legend()
#     savefig("01_Position_Tracking_Norm_PD")

# =========================================================
# EXPERIMENT 2 — 3D trajectory (coupled)
# =========================================================
# def plot_3d_trajectory(p_ref, pE, pQ):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")

#     ax.plot(pE[:, 0], pE[:, 1], pE[:, 2], linewidth=2.5, label="Euler Angles")
#     ax.plot(pQ[:, 0], pQ[:, 1], pQ[:, 2], linewidth=2.5, label="Quaternion")
#     ax.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], "--", linewidth=2.8, alpha=0.7, label="Reference")

#     ax.scatter(pE[0, 0], pE[0, 1], pE[0, 2], s=50)
#     ax.scatter(pQ[0, 0], pQ[0, 1], pQ[0, 2], s=50)

#     ax.set_xlabel("X [m]")
#     ax.set_ylabel("Y [m]")
#     ax.set_zlabel("Z [m]")
#     ax.set_title("Coupled 3D Quadrotor Trajectory under PD Control")

#     # Prettier viewpoint and aspect
#     ax.view_init(elev=22, azim=40)
#     ax.set_box_aspect([1, 1, 1])
#     ax.grid(True)
#     ax.legend(loc="upper left")
#     savefig("02_3D_Trajectory_Coupled_PD")

# =========================================================
# EXPERIMENTS 3–6 — Euler singularity vs true quaternion attitude
# Attitude-only (pitch command), but quaternion is REAL (q ∈ R4)
# =========================================================
def run_attitude_singularity(dt=0.001, T_end=5.0,
                             theta_target_deg=90.0,
                             ramp_start=0.5, ramp_end=2.0,
                             Kp=10.0, Kd=2.0):
    """
    Attitude-only: track a pitch command theta_ref(t) with PD on pitch error.
    Euler kinematics: theta_dot = omega_y / cos(theta)  (explicit singularity)
    Quaternion kinematics: q_dot = 0.5 Omega(omega) q, and we extract pitch for plotting.

    For fairness, both use the SAME body-rate dynamics:
      omega_dot = u  (simple), where u is "torque-like" PD
    """
    t = np.arange(0.0, T_end, dt)
    N = len(t)

    # reference: ramp to theta_target
    theta_ref = np.zeros(N)
    theta_tar = np.deg2rad(theta_target_deg)
    for k in range(N):
        if t[k] < ramp_start:
            theta_ref[k] = 0.0
        elif t[k] > ramp_end:
            theta_ref[k] = theta_tar
        else:
            s = (t[k] - ramp_start) / (ramp_end - ramp_start)
            theta_ref[k] = s * theta_tar

    # Euler states (pitch only)
    theta_E = np.zeros(N)
    omega_E = np.zeros(N)  # omega about body y (q rate)
    u_E = np.zeros(N)

    # Quaternion states (full q, but we only excite y-axis)
    q = np.zeros((N, 4))
    q[0] = np.array([1.0, 0.0, 0.0, 0.0])
    omega_Q = np.zeros(N)  # about body y
    u_Q = np.zeros(N)

    pitch_Q = np.zeros(N)  # extracted pitch from quaternion

    # Sim
    for k in range(1, N):
        # -----------------------------
        # Euler PD (pitch)
        # -----------------------------
        eE = theta_ref[k] - theta_E[k-1]
        uE = Kp*eE - Kd*omega_E[k-1]
        u_E[k] = uE
        omega_E[k] = omega_E[k-1] + uE * dt

        # Explicit singular kinematics:
        denom = np.cos(theta_E[k-1])
        if abs(denom) < 1e-9:
            denom = np.sign(denom) * 1e-9 if denom != 0 else 1e-9
        theta_dot = omega_E[k] / denom
        theta_E[k] = theta_E[k-1] + theta_dot * dt

        # -----------------------------
        # Quaternion PD (track pitch)
        # Make a desired quaternion for pure pitch (roll=yaw=0)
        # -----------------------------
        theta_d = theta_ref[k]
        Rdes = R_from_euler(0.0, theta_d, 0.0)
        q_des = q_from_R(Rdes)

        # Extract current pitch
        _, pitch, _ = euler_from_q(q[k-1])
        pitch_Q[k-1] = pitch

        # Quaternion error -> use vector part
        q_err = q_mul(q_conj(q_des), q[k-1])
        q_err = q_norm(q_err)
        if q_err[0] < 0:
            q_err = -q_err

        # For pure pitch, the dominant error is in y component (qx,qy,qz)
        e_vec = q_err[1:4]
        # PD "torque" on body rates (only y-axis is active)
        uQ = -Kp*e_vec[1] - Kd*omega_Q[k-1]
        u_Q[k] = uQ
        omega_Q[k] = omega_Q[k-1] + uQ * dt

        omega_vec = np.array([0.0, omega_Q[k], 0.0])
        qdot = 0.5 * (Omega_mat(omega_vec) @ q[k-1])
        q[k] = q_norm(q[k-1] + qdot * dt)

    # Final pitch extraction
    for k in range(N):
        _, pitch, _ = euler_from_q(q[k])
        pitch_Q[k] = pitch

    # Singular instant (Euler ill-conditioning proxy)
    # Identify max |omega_E / cos(theta)| or max |theta_dot|
    denom = np.cos(theta_E)
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    theta_dot_E = omega_E / denom
    idx_sing = int(np.argmax(np.abs(theta_dot_E)))
    t_sing = t[idx_sing]

    return (t, theta_ref, theta_E, pitch_Q, omega_E, omega_Q, u_E, u_Q, t_sing)

def plot_pitch_tracking_near_singularity(
        t, theta_ref, theta_E, pitch_Q, t_sing,
        inset_window=(2.0, 4.0),
        inset_ylim=(0.0, 2.0),
        inset_loc="lower left",
        base_name="03_Pitch_Tracking_Near_Singularity_PD"):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, theta_E, label="Euler Angles", linewidth=2)
    ax.plot(t, pitch_Q, label="Quaternion", linewidth=2)
    ax.plot(t, theta_ref, "--", label="Reference", linewidth=2)

    ax.axvline(t_sing, color="k", linestyle="--", alpha=0.6)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Pitch Angle $\theta$ [rad]")
    ax.set_title("Pitch Tracking under PD Control (Near Euler Singularity)")
    ax.grid(True)
    ax.legend(loc="upper right")

    # ------------------------
    # Inset (zoom)
    # ------------------------
    axins = inset_axes(
        ax,
        width="40%",
        height="45%",
        loc= "center left",
        borderpad=2
    )

    axins.plot(t, theta_E, linewidth=2)
    axins.plot(t, pitch_Q, linewidth=2)
    axins.plot(t, theta_ref, "--", linewidth=2)
    axins.axvline(t_sing, color="k", linestyle="--", alpha=0.6)

    axins.set_xlim(1, 3)
    axins.set_ylim(0, 3)
    axins.grid(True)

    # Caixa do zoom + conectores
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    savefig(base_name)
    plt.show()


    # Inset zoom
    # ax = plt.gca()
    # axins = inset_axes(ax, width="35%", height="50%", loc="lower right", borderpad=1.0)
    # axins.plot(t, theta_E, linewidth=1.5)
    # axins.plot(t, pitch_Q, linewidth=1.5)
    # axins.plot(t, theta_ref, "--", linewidth=1.2)
    # axins.axvline(t_sing, color="k", linestyle="--", alpha=0.6)

    # axins.set_xlim(inset_window[0], inset_window[1])
    # axins.set_ylim(inset_ylim[0], inset_ylim[1])
    # axins.grid(True)

    savefig(base_name)

# def plot_angular_rate_demand(t, omega_E, omega_Q, t_sing,
#                             inset_window=(0.40, 0.65),
#                             base_name="04_Angular_Rate_Demand_Near_Singularity_PD"):
#     plt.figure(figsize=(10, 5))
#     plt.plot(t, np.abs(omega_E), label="Euler Angles", linewidth=2)
#     plt.plot(t, np.abs(omega_Q), label="Quaternion", linewidth=2)

#     plt.axvline(t_sing, color="k", linestyle="--", alpha=0.6)
#     plt.yscale("log")
#     plt.xlabel("Time [s]")
#     plt.ylabel(r"$\|\omega\|$  [rad/s]")
#     plt.title("Angular Rate Demand near Euler Singularity (PD Control)")
#     plt.grid(True, which="both")
#     plt.legend(loc="upper right")

    # Inset zoom (linear y inside to see shape)
    # ax = plt.gca()
    # axins = inset_axes(ax, width="38%", height="50%", loc="lower left", borderpad=1.0)
    # axins.plot(t, np.abs(omega_E), linewidth=1.5)
    # axins.plot(t, np.abs(omega_Q), linewidth=1.5)
    # axins.axvline(t_sing, color="k", linestyle="--", alpha=0.6)
    # axins.set_xlim(inset_window[0], inset_window[1])
    # axins.set_yscale("log")
    # axins.grid(True, which="both")

    # savefig(base_name)

def plot_forced_crossing(dt=0.001, T_end=3.0, theta_target_deg=120.0,
                         Kp=10.0, Kd=2.0,
                         base_name_angle="05_Euler_vs_Quaternion_Forced_Crossing_PD",
                         base_name_effort="06_Control_Effort_Under_Singularity_PD"):

    t, theta_ref, theta_E, pitch_Q, omega_E, omega_Q, u_E, u_Q, t_sing = run_attitude_singularity(
        dt=dt, T_end=T_end, theta_target_deg=theta_target_deg,
        ramp_start=0.0, ramp_end=1.0, Kp=Kp, Kd=Kd
    )

    # =========================================================
    # ANGLE PLOT (COM INSET)
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, theta_E, label="Euler Angles", linewidth=2)
    ax.plot(t, pitch_Q, label="Quaternion", linewidth=2)
    ax.plot(t, theta_ref, "--", label="Reference", linewidth=2)

    ax.axvline(t_sing, color="k", linestyle="--", alpha=0.6)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Pitch Angle $\theta$ [rad]")
    ax.set_title("Euler Singularity versus Quaternion Representation (Forced Crossing)")
    ax.grid(True)
    ax.legend(loc="upper right")

    # -------------------------
    # INSET (ZOOM)
    # -------------------------
    axins = inset_axes(
        ax,
        width="40%",        # AJUSTE SE QUISER
        height="45%",       # AJUSTE SE QUISER
        loc="lower center",   # AJUSTE SE QUISER
        borderpad=2
    )

    axins.plot(t, theta_E, linewidth=2)
    axins.plot(t, pitch_Q, linewidth=2)
    axins.plot(t, theta_ref, "--", linewidth=2)
    axins.axvline(t_sing, color="k", linestyle="--", alpha=0.6)

    # ---- LIMITES DO ZOOM DO INSET (MEXE SÓ AQUI) ----
    axins.set_xlim(0.5, 1.25)
    axins.set_ylim(0.0, 5.0)

    axins.grid(True)

    # Caixa conectando o zoom ao gráfico principal
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    savefig(base_name_angle)
    plt.show()

    # =========================================================
    # CONTROL EFFORT PLOT (SEM INSET)
    # =========================================================
    plt.figure(figsize=(10, 5))
    plt.plot(t, np.abs(u_E), label="Euler Angles", linewidth=2)
    plt.plot(t, np.abs(u_Q), label="Quaternion", linewidth=2)

    plt.axvline(t_sing, color="k", linestyle="--", alpha=0.6)
    plt.yscale("log")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\|u\|$  [arb. units]")
    plt.title("Control Effort under Kinematic Singularity of Euler Angles (PD Control)")
    plt.grid(True, which="both")
    plt.legend(loc="upper right")

    savefig(base_name_effort)
    plt.show()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # -------------------------------
    # Coupled 3D (position + attitude)
    # -------------------------------
    t0, pref, pE, pQ, nref, nE, nQ = run_coupled_3d(dt=0.002, T_end=8.0, seed=2)

    # plot_position_tracking_norm(t0, nref, nE, nQ)
    # plot_3d_trajectory(pref, pE, pQ)
    plt.show()

    # ---------------------------------------
    # Near-singularity pitch tracking (PD)
    # ---------------------------------------
    t, theta_ref, theta_E, pitch_Q, omega_E, omega_Q, u_E, u_Q, t_sing = run_attitude_singularity(
        dt=0.001, T_end=5.0, theta_target_deg=90.0,
        ramp_start=0.5, ramp_end=2.0,
        Kp=10.0, Kd=2.0
    )

    plot_pitch_tracking_near_singularity(
        t, theta_ref, theta_E, pitch_Q, t_sing,
        inset_window=(t_sing - 0.08, t_sing + 0.10),
        inset_ylim=(-0.4, 0.6),
        base_name="03_Pitch_Tracking_Near_Euler_Singularity_PD"
    )
    # plot_angular_rate_demand(
    #     t, omega_E, omega_Q, t_sing,
    #     inset_window=(t_sing - 0.08, t_sing + 0.10),
    #     base_name="04_Angular_Rate_Demand_Near_Euler_Singularity_PD"
    # )
    plt.show()

    # ---------------------------------------
    # Forced crossing + definitive effort plot
    # ---------------------------------------
    plot_forced_crossing(
        dt=0.001, T_end=3.0, theta_target_deg=120.0,
        Kp=10.0, Kd=2.0,
        base_name_angle="05_Euler_vs_Quaternion_Forced_Crossing_PD",
        base_name_effort="06_Control_Effort_Under_Singularity_PD"
    )
    plt.show()

    print("\nDone. EPS+PNG saved in:", OUT_DIR)
