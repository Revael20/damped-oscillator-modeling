# PINN Damped Harmonic Oscillator (Experimental Data)

This project uses a **Physics-Informed Neural Network (PINN)** to model the damped harmonic motion of a metal sphere in water, based on real experimental data. The neural network is trained to satisfy both physical laws (ODEs) and data measurements.

---

## 📊 Problem Description

We analyzed an experiment of a metal sphere oscillating vertically in water. The motion follows a second-order differential equation representing a **damped harmonic oscillator**:

\[
\frac{d^2y}{dt^2} + 2\zeta \omega \frac{dy}{dt} + \omega^2 y = 0
\]

Where:
- \( y(t) \): position over time
- \( \zeta \): damping ratio
- \( \omega \): natural frequency

---

## 🧠 PINN Model

We used PyTorch to define a fully connected neural network that:
- Takes time \( t \) as input
- Outputs predicted position \( y(t) \)
- Computes derivatives using `torch.autograd` for \( \frac{dy}{dt} \) and \( \frac{d^2y}{dt^2} \)
- Minimizes a **hybrid loss**:
  - Physics loss (ODE residual)
  - Data loss (difference between prediction and experimental data)
  - Optional regularization on learnable parameters

---

## 🧪 Files Included

| File | Description |
|------|-------------|
| `PINN_Damped_Oscillator.ipynb` | Full notebook including model definition, training, and plotting |
| `Sphere_in_Water.csv` | Experimental data (time and position) |
| `example_plot.png` | Sample output comparing PINN prediction with real data |
| `.gitignore` | Prevents unnecessary files (e.g., checkpoints, cache) from being tracked |
| `README.md` | You’re reading it! 🎉 |

---

## ⚙️ Learnable Parameters

The model learns:
- \( \zeta \): damping ratio (soft-bounded between 0 and 0.9)
- \( \omega \): frequency (positive)
- \( C \): vertical offset, since the data doesn’t start at 0

This makes the PINN both a data-fitting and physics-solving tool.

---

## 🧩 Notes

- We chose a hybrid loss to balance physics and data.
- A regularization term on \( \zeta, \omega \) is optional.
- The network sometimes converges to incorrect physical parameters if data is noisy or poorly scaled — this is an active challenge in PINN research.

---

## 📈 Results

- The final output shows close alignment between experimental measurements and the PINN prediction.
- The model attempts to discover the true damping ratio and frequency just from data + differential equations.

---

## 🤖 Tools

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 📌 Credits

- Data collected using LoggerPro from a real-world physics lab
- Code and modeling by Revael Pramudya Yudisthira

---
