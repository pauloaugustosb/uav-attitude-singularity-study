# UAV Attitude Representation Study

This repository contains the simulation scripts and generated results for the paper:

**"Impact of Euler Angle Singularities on UAV Attitude Control: A Comparison with Quaternion-Based Formulations"**

Submitted to **MEDER Conference 2026**.

## Overview

This work investigates how attitude parametrization influences the closed-loop rotational dynamics of unmanned aerial vehicles (UAVs).

Two attitude representations are compared:

- Euler angles (ZYX parametrization)
- Unit quaternions

Both implementations use the same geometric control law defined directly on **SO(3)**.  
This allows the study to isolate the effects of the internal attitude propagation mechanism.

The simulations evaluate three scenarios:

1. **Nominal maneuver** – 30° pitch
2. **Near-singularity maneuver** – 90° pitch
3. **Singularity-crossing maneuver** – 120° pitch

The results show that Euler-angle parametrizations can introduce dynamic amplification near kinematic singularities, while quaternion propagation remains well-conditioned.

---

# Repository Structure
uav-attitude-singularity-study
├── README.md
├── requirements.txt
├── scripts
│   ├── articleConference.py
│   └── review.py
│
├── figures
├── out
├── paper
│   ├── MEDER2026_submission.pdf
│   ├── MEDER2026_revision.pdf
│   └── revision_notes.pdf

---

# Installation

Clone the repository: git clone https://github.com/pauloaugustosb/uav-attitude-singularity-study.git

Enter the project directory: cd uav-attitude-singularity-study

# Install dependencies:
  pip install -r requirements.txt

---

# Running the Simulation

Run the main simulation script: python articleConference.py and review.py files for the review code

This script generates the results used in the paper, including the figures for the different maneuver scenarios.

---

# Reproducibility

All results and figures presented in the paper can be reproduced using the scripts available in this repository.

---

# Citation

If you use this code or results in your research, please cite the associated paper:  
Paulo Augusto Silva Borges et al.
"Impact of Euler Angle Singularities on UAV Attitude Control"


---

# Author

Paulo Augusto Silva Borges  
Federal University of Uberlândia (UFU)

---

# License

This project is released under the MIT License.
