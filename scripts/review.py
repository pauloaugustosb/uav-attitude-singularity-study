#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quadrotor reduced simulation with:
1) Coupled 3D translation + attitude (Euler vs Quaternion) under PD/geometric-like setup
2) SO(3)/attitude-only singularity experiments:
   - Nominal 30 deg maneuver
   - Near-singularity 90 deg maneuver
   - Forced crossing 120 deg maneuver

Main output:
- One combined figure per scenario, generated directly in Python:
    * Nominal_30deg_combined.(png|pdf|eps)
    * NearSingularity_90deg_combined.(png|pdf|eps)
    * ForcedCrossing_120deg_combined.(png|pdf|eps)

This avoids oversized LaTeX figures and is better suited for 8-page conference papers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# OUTPUT
# =========================================================
OUT_DIR = "/home/paulo/Documentos/Academico/Rogerio_conference/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(base_name: str):
    png = os.path.join(OUT_DIR, f"{base_name}.png")
    pdf = os.path.join(OUT_DIR, f"{base_name}.pdf")
    eps = os.path.join(OUT_DIR, f"{base_name}.eps")
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(eps, format="eps", bbox_inches="tight")
    print(f"[saved] {png}")
    print(f"[saved] {pdf}")
    print(f"[saved] {eps}")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.0,
})

# =========================================================
# NUMERICS
# =========================================================
def sat(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# =========================================================
# ROTATION: EULER (ZYX) <-> R
# =========================================================
def R_from_euler(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    # ZYX: R = Rz(psi) Ry(theta) Rx(phi)
    return np.array([
        [cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi],
        [spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi],
        [-sth,     cth*sphi,                  cth*cphi]
    ])

def euler_from_R(R):
    theta = np.arcsin(sat(-R[2, 0], -1.0, 1.0))
    cth = np.cos(theta)

    if abs(cth) < 1e-9:
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

    if abs(cth) < 1e-9:
        cth = np.sign(cth) * 1e-9 if cth != 0 else 1e-9

    return np.array([
        [1.0, sphi*tth, cphi*tth],
        [0.0, cphi,     -sphi],
        [0.0, sphi/cth, cphi/cth]
    ])

# =========================================================
# ROTATION: QUATERNIONS
# Convention: q = [qw, qx, qy, qz]
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
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),         1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),         2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
    ])

def q_from_R(R):
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
    p, q, r = omega
    return np.array([
        [0.0, -p,  -q,  -r],
        [p,   0.0, r,   -q],
        [q,   -r,  0.0, p],
        [r,   q,   -p,  0.0]
    ])

def euler_from_q(q):
    return euler_from_R(R_from_q(q))

# =========================================================
# SIMPLE QUADROTOR MODEL
# =========================================================
class Params:
    def __init__(self):
        self.m = 1.0
        self.g = 9.81
        self.I = np.diag([0.02, 0.02, 0.04])

        self.T_min = 0.0
        self.T_max = 2.5 * self.m * self.g
        self.tau_max = np.array([1.5, 1.5, 1.0])

P = Params()

# =========================================================
# CONTROLLERS
# =========================================================
def desired_attitude_from_acc(a_des, yaw_des=0.0):
    e3 = np.array([0.0, 0.0, 1.0])
    a_total = a_des + np.array([0.0, 0.0, P.g])

    norm_a = np.linalg.norm(a_total)
    if norm_a < 1e-9:
        b3 = e3.copy()
        T = P.m * P.g
    else:
        b3 = a_total / norm_a
        T = P.m * norm_a

    cpsi, spsi = np.cos(yaw_des), np.sin(yaw_des)
    b1_ref = np.array([cpsi, spsi, 0.0])

    b2 = np.cross(b3, b1_ref)
    nb2 = np.linalg.norm(b2)
    if nb2 < 1e-9:
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
    e[2] = wrap_pi(e[2])
    return Kp*e - Kd*omega

def pd_attitude_quat(q, omega, q_des, Kp, Kd):
    q_err = q_mul(q_conj(q_des), q)
    q_err = q_norm(q_err)
    if q_err[0] < 0:
        q_err = -q_err
    e_vec = q_err[1:4]
    return -Kp*e_vec - Kd*omega

# =========================================================
# COUPLED 3D SIMULATION
# =========================================================
def run_coupled_3d(dt=0.002, T_end=8.0, seed=2):
    t = np.arange(0.0, T_end, dt)
    N = len(t)

    p_ref = np.zeros((N, 3))
    v_ref = np.zeros((N, 3))
    for k in range(N):
        if t[k] < 1.0:
            p_ref[k] = np.array([0.0, 0.0, 0.0])
            v_ref[k] = np.array([0.0, 0.0, 0.0])
        else:
            s = sat((t[k] - 1.0) / 4.0, 0.0, 1.0)
            p_ref[k] = np.array([1.0, 1.0, 1.0]) * s
            v_ref[k] = np.array([1.0, 1.0, 1.0]) * (1.0/4.0 if (1.0 < t[k] < 5.0) else 0.0)

    rng = np.random.default_rng(seed)
    wind = 0.10 * rng.standard_normal((N, 3))
    gust = np.zeros((N, 3))
    gust[(t > 3.0) & (t < 3.5)] = np.array([1.0, -0.6, 0.3])
    d_ext = wind + gust

    Kp_pos = np.diag([2.0, 2.0, 3.0])
    Kd_pos = np.diag([1.6, 1.6, 2.2])

    Kp_att_e = np.array([8.0, 8.0, 4.0])
    Kd_att_e = np.array([2.0, 2.0, 1.5])

    Kp_att_q = 10.0
    Kd_att_q = 2.5

    pE = np.zeros((N, 3))
    vE = np.zeros((N, 3))
    angE = np.zeros((N, 3))
    wE = np.zeros((N, 3))

    pQ = np.zeros((N, 3))
    vQ = np.zeros((N, 3))
    qQ = np.zeros((N, 4))
    qQ[0] = np.array([1.0, 0.0, 0.0, 0.0])
    wQ = np.zeros((N, 3))

    e3 = np.array([0.0, 0.0, 1.0])

    for k in range(1, N):
        a_des_E = pd_position(pE[k-1], vE[k-1], p_ref[k], v_ref[k], Kp_pos, Kd_pos)
        a_des_Q = pd_position(pQ[k-1], vQ[k-1], p_ref[k], v_ref[k], Kp_pos, Kd_pos)

        Rdes_E, Tdes_E = desired_attitude_from_acc(a_des_E, yaw_des=0.0)
        Rdes_Q, Tdes_Q = desired_attitude_from_acc(a_des_Q, yaw_des=0.0)

        Tdes_E = float(sat(Tdes_E, P.T_min, P.T_max))
        Tdes_Q = float(sat(Tdes_Q, P.T_min, P.T_max))

        ang_ref_E = np.array(euler_from_R(Rdes_E))
        q_des = q_from_R(Rdes_Q)

        tauE = pd_attitude_euler(angE[k-1], wE[k-1], ang_ref_E, Kp_att_e, Kd_att_e)
        tauQ = pd_attitude_quat(qQ[k-1], wQ[k-1], q_des, Kp_att_q, Kd_att_q)

        tauE = sat(tauE, -P.tau_max, P.tau_max)
        tauQ = sat(tauQ, -P.tau_max, P.tau_max)

        RE = R_from_euler(*angE[k-1])
        aE = np.array([0.0, 0.0, -P.g]) + (Tdes_E / P.m) * (RE @ e3) + d_ext[k] / P.m
        vE[k] = vE[k-1] + aE * dt
        pE[k] = pE[k-1] + vE[k] * dt

        wdotE = np.linalg.inv(P.I) @ (tauE - np.cross(wE[k-1], P.I @ wE[k-1]))
        wE[k] = wE[k-1] + wdotE * dt

        Ed = E_matrix(angE[k-1][0], angE[k-1][1])
        angdotE = Ed @ wE[k]
        angE[k] = angE[k-1] + angdotE * dt

        RQ = R_from_q(qQ[k-1])
        aQ = np.array([0.0, 0.0, -P.g]) + (Tdes_Q / P.m) * (RQ @ e3) + d_ext[k] / P.m
        vQ[k] = vQ[k-1] + aQ * dt
        pQ[k] = pQ[k-1] + vQ[k] * dt

        wdotQ = np.linalg.inv(P.I) @ (tauQ - np.cross(wQ[k-1], P.I @ wQ[k-1]))
        wQ[k] = wQ[k-1] + wdotQ * dt

        qdot = 0.5 * (Omega_mat(wQ[k]) @ qQ[k-1])
        qQ[k] = q_norm(qQ[k-1] + qdot * dt)

    posnorm_E = np.linalg.norm(pE, axis=1)
    posnorm_Q = np.linalg.norm(pQ, axis=1)
    posnorm_ref = np.linalg.norm(p_ref, axis=1)

    return t, p_ref, pE, pQ, posnorm_ref, posnorm_E, posnorm_Q

# =========================================================
# ATTITUDE-ONLY SINGULARITY SIMULATION
# =========================================================
def run_attitude_singularity(
    dt=0.001,
    T_end=5.0,
    theta_target_deg=90.0,
    ramp_start=0.5,
    ramp_end=2.0,
    Kp=10.0,
    Kd=2.0
):
    """
    Attitude-only experiment.
    Euler:
        theta_dot = omega_y / cos(theta)
    Quaternion:
        q_dot = 0.5 * Omega(omega) * q
    Both use the same simple rate dynamics:
        omega_dot = u
    """
    t = np.arange(0.0, T_end, dt)
    N = len(t)

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

    # Euler states
    theta_E = np.zeros(N)
    omega_E = np.zeros(N)
    u_E = np.zeros(N)

    # Quaternion states
    q = np.zeros((N, 4))
    q[0] = np.array([1.0, 0.0, 0.0, 0.0])
    omega_Q = np.zeros(N)
    u_Q = np.zeros(N)
    pitch_Q = np.zeros(N)

    for k in range(1, N):
        # Euler PD
        eE = theta_ref[k] - theta_E[k-1]
        uE = Kp * eE - Kd * omega_E[k-1]
        u_E[k] = uE
        omega_E[k] = omega_E[k-1] + uE * dt

        denom = np.cos(theta_E[k-1])
        if abs(denom) < 1e-9:
            denom = np.sign(denom) * 1e-9 if denom != 0 else 1e-9
        theta_dot = omega_E[k] / denom
        theta_E[k] = theta_E[k-1] + theta_dot * dt

        # Quaternion PD
        theta_d = theta_ref[k]
        Rdes = R_from_euler(0.0, theta_d, 0.0)
        q_des = q_from_R(Rdes)

        _, pitch, _ = euler_from_q(q[k-1])
        pitch_Q[k-1] = pitch

        q_err = q_mul(q_conj(q_des), q[k-1])
        q_err = q_norm(q_err)
        if q_err[0] < 0:
            q_err = -q_err

        e_vec = q_err[1:4]
        uQ = -Kp * e_vec[1] - Kd * omega_Q[k-1]
        u_Q[k] = uQ
        omega_Q[k] = omega_Q[k-1] + uQ * dt

        omega_vec = np.array([0.0, omega_Q[k], 0.0])
        qdot = 0.5 * (Omega_mat(omega_vec) @ q[k-1])
        q[k] = q_norm(q[k-1] + qdot * dt)

    for k in range(N):
        _, pitch, _ = euler_from_q(q[k])
        pitch_Q[k] = pitch

    denom = np.cos(theta_E)
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    theta_dot_E = omega_E / denom
    idx_sing = int(np.argmax(np.abs(theta_dot_E)))
    t_sing = t[idx_sing]

    condE = np.abs(1.0 / denom)

    return {
        "t": t,
        "theta_ref": theta_ref,
        "theta_E": theta_E,
        "pitch_Q": pitch_Q,
        "omega_E": omega_E,
        "omega_Q": omega_Q,
        "u_E": u_E,
        "u_Q": u_Q,
        "t_sing": t_sing,
        "condE_proxy": condE,
    }

# =========================================================
# COMBINED FIGURES
# =========================================================
def plot_nominal_combined(data, base_name="Nominal_30deg_combined"):
    t = data["t"]
    theta_ref = data["theta_ref"]
    theta_E = data["theta_E"]
    pitch_Q = data["pitch_Q"]
    u_E = np.abs(data["u_E"])
    u_Q = np.abs(data["u_Q"])
    condE = data["condE_proxy"]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))

    # (a) attitude error / pitch tracking
    ax[0].plot(t, theta_E, label="Euler")
    ax[0].plot(t, pitch_Q, label="Quaternion")
    ax[0].plot(t, theta_ref, "--", label="Reference")
    ax[0].set_title("(a) $e_{ang}$/pitch response")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Angle [rad]")
    ax[0].grid(True)
    ax[0].legend(loc="best")

    # (b) control effort
    ax[1].plot(t, u_E, label="Euler")
    ax[1].plot(t, u_Q, label="Quaternion")
    ax[1].set_title(r"(b) Control effort $\|u\|$")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Arb. units")
    ax[1].set_yscale("log")
    ax[1].grid(True, which="both")

    # (c) conditioning
    ax[2].plot(t, condE)
    ax[2].set_title(r"(c) Conditioning proxy $\kappa(E)$")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Proxy")
    ax[2].grid(True, which="both")
    ax[2].set_yscale("log")

    savefig(base_name)
    plt.show()

def plot_near_singularity_combined(data, base_name="NearSingularity_90deg_combined"):
    t = data["t"]
    theta_ref = data["theta_ref"]
    theta_E = data["theta_E"]
    pitch_Q = data["pitch_Q"]
    condE = data["condE_proxy"]
    t_sing = data["t_sing"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))

    ax[0].plot(t, theta_E, label="Euler")
    ax[0].plot(t, pitch_Q, label="Quaternion")
    ax[0].plot(t, theta_ref, "--", label="Reference")
    ax[0].axvline(t_sing, color="k", linestyle="--", alpha=0.6)
    ax[0].set_title(r"(a) Attitude response near $90^\circ$")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Angle [rad]")
    ax[0].grid(True)
    ax[0].legend(loc="best")

    ax[1].plot(t, condE)
    ax[1].axvline(t_sing, color="k", linestyle="--", alpha=0.6)
    ax[1].set_title(r"(b) Conditioning proxy $\kappa(E)$")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Proxy")
    ax[1].set_yscale("log")
    ax[1].grid(True, which="both")

    savefig(base_name)
    plt.show()

def plot_forced_crossing_combined(data, base_name="ForcedCrossing_120deg_combined"):
    t = data["t"]
    theta_ref = data["theta_ref"]
    theta_E = data["theta_E"]
    pitch_Q = data["pitch_Q"]
    u_E = np.abs(data["u_E"])
    u_Q = np.abs(data["u_Q"])
    condE = data["condE_proxy"]
    t_sing = data["t_sing"]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))

    ax[0].plot(t, theta_E, label="Euler")
    ax[0].plot(t, pitch_Q, label="Quaternion")
    ax[0].plot(t, theta_ref, "--", label="Reference")
    ax[0].axvline(t_sing, color="k", linestyle="--", alpha=0.6)
    ax[0].set_title(r"(a) Crossing response")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Angle [rad]")
    ax[0].grid(True)
    ax[0].legend(loc="best")

    ax[1].plot(t, u_E, label="Euler")
    ax[1].plot(t, u_Q, label="Quaternion")
    ax[1].axvline(t_sing, color="k", linestyle="--", alpha=0.6)
    ax[1].set_title(r"(b) Control effort $\|u\|$")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Arb. units")
    ax[1].set_yscale("log")
    ax[1].grid(True, which="both")

    ax[2].plot(t, condE)
    ax[2].axvline(t_sing, color="k", linestyle="--", alpha=0.6)
    ax[2].set_title(r"(c) Conditioning proxy $\kappa(E)$")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Proxy")
    ax[2].set_yscale("log")
    ax[2].grid(True, which="both")

    savefig(base_name)
    plt.show()

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # Optional coupled 3D simulation, kept for future use
    _ = run_coupled_3d(dt=0.002, T_end=8.0, seed=2)

    # 30 deg nominal case
    nominal = run_attitude_singularity(
        dt=0.001,
        T_end=4.0,
        theta_target_deg=30.0,
        ramp_start=0.5,
        ramp_end=1.5,
        Kp=10.0,
        Kd=2.0
    )
    plot_nominal_combined(nominal, base_name="Nominal_30deg_combined")

    # 90 deg near singularity
    near90 = run_attitude_singularity(
        dt=0.001,
        T_end=5.0,
        theta_target_deg=90.0,
        ramp_start=0.5,
        ramp_end=2.0,
        Kp=10.0,
        Kd=2.0
    )
    plot_near_singularity_combined(near90, base_name="NearSingularity_90deg_combined")

    # 120 deg crossing
    cross120 = run_attitude_singularity(
        dt=0.001,
        T_end=3.0,
        theta_target_deg=120.0,
        ramp_start=0.0,
        ramp_end=1.0,
        Kp=10.0,
        Kd=2.0
    )
    plot_forced_crossing_combined(cross120, base_name="ForcedCrossing_120deg_combined")

    print("\nDone. Combined figures saved in:", OUT_DIR)