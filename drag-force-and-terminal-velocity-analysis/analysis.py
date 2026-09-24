import numpy as np
import matplotlib.pyplot as plt

plt.close('all')


# ==========================================
# Part A: Setting up the system
# ==========================================
print("A correctly implemented")

m_cart = 0.2709           # kg
delta_m_cart = 0.0001     # kg

m_magnet = 0.073          # kg
delta_m_magnet = 0.0001   # kg

m_mass1 = 0.2523          # kg
delta_m_mass1 = 0.0001    # kg

m_mass2 = 0.2523          # kg
delta_m_mass2 = 0.0001    # kg

# theta used to compensate the friction
theta_comp = 0.25        # degree
delta_theta_comp = 0.05  # degree

g = 9.79127              # gravitational acceleration, m/s^2
delta_g = 0.000005       # uncertainty in g, m/s^2


# ==========================================
# Part B: Compensating for unwanted friction
# ==========================================
print("B correctly implemented")

# Load experimental data
data_before = np.loadtxt("B_before.txt", encoding="utf-8-sig")

data_after = np.loadtxt("B_after.txt", encoding="utf-8-sig")

# Extract time (1st col) and velocity columns (2nd)
time_before = data_before[:, 0]
velocity_before = data_before[:, 1]

time_after = data_after[:, 0]
velocity_after = data_after[:, 1]

# Graph before track adjustment
# Fig. 1
plt.figure()
plt.plot(time_before, velocity_before)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity before track adjustment")
plt.show()

# Graph after track adjustment
# Fig. 2
plt.figure()
plt.plot(time_after, velocity_after)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity after track adjustment")
plt.show()


# ==========================================
# Part C: Exploring the drag force
# ==========================================

# Total mass used in Part C
m_total = m_cart + m_magnet + m_mass1 + m_mass2
print("Total mass in Part C =", m_total, "kg")

# Extract time (1st col), force (2nd col) and velocity (3rd col) columns
data_C = np.loadtxt("c.txt", encoding="utf-8-sig")
time_C = data_C[:, 0]
force_C = data_C[:, 1]
velocity_C = data_C[:, 2]

# Clean outliers
idx_max = np.argmax(velocity_C)
velocity_C = velocity_C[idx_max:]
time_C = time_C[idx_max:]
force_C = force_C[idx_max:]
# Remove the first two measurements, which correspond to the initial abnormal data
velocity_C = velocity_C[2:]
time_C = time_C[2:]
force_C = force_C[2:]

# --------------------------------------------------
# C1. Graph of force as a function of velocity F(v)
# --------------------------------------------------
print("C1 correctly implemented")

# Fig. 3
plt.figure()
plt.plot(velocity_C, force_C, 'o')
plt.xlabel("Velocity (m/s)")
plt.ylabel("Force (N)")
plt.title("Force as a function of velocity")
plt.show()

# --------------------------------------------------
# C2. Does the force proportional to the velocity? 
# Use curve fitting to find the proportionally constant.
# --------------------------------------------------
print("C2 correctly implemented")

from scipy.optimize import curve_fit

# Linear model
def linear_model(v, k, c):
    return k * v + c

popt_force, pcov_force = curve_fit(linear_model, velocity_C, force_C)
k = popt_force[0]
c = popt_force[1]

# Uncertainty of fitted slope
delta_k = np.sqrt(pcov_force[0, 0])

# According to F_drag = -b*v
b_force = -k
delta_b_force = delta_k

print("Slope k =", k)
print("Intercept c =", c)
print("Drag coefficient from F(v) =", b_force, "kg/s")

v_line = np.linspace(np.min(velocity_C), np.max(velocity_C), 100)
F_line = linear_model(v_line, k, c)

# Fig. 4
plt.figure()
plt.plot(velocity_C, force_C, 'o')
plt.plot(v_line, F_line)
plt.xlabel("Velocity (m/s)")
plt.ylabel("Force (N)")
plt.title("Force as a function of velocity")
plt.legend(["Experimental data", "Linear fit"])

plt.show()

# --------------------------------------------------
# C3. Create a graph of the cart’s velocity as a function of time
# --------------------------------------------------
print("C3 correctly implemented")

# Fig. 5
plt.figure()
plt.plot(time_C, velocity_C)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity as a function of time")
plt.show()

# --------------------------------------------------
# C4. Linearize v(t) and find the drag coefficient
# --------------------------------------------------
print("C4 correctly implemented")

# ln(v) only works for positive velocity
positive = velocity_C > 0
time_log = time_C[positive]
velocity_log = velocity_C[positive]

# Shift time so that the beginning of the fitted motion is t = 0
time_log = time_log - time_log[0]
ln_velocity = np.log(velocity_log)

popt_time, pcov_time = curve_fit(linear_model, time_log, ln_velocity)

k_time = popt_time[0]
c_time = popt_time[1]

# Uncertainty of fitted slope
delta_k_time = np.sqrt(pcov_time[0, 0])

# Total mass uncertainty
delta_m_total = np.sqrt(
    delta_m_cart**2 +
    delta_m_magnet**2 +
    delta_m_mass1**2 +
    delta_m_mass2**2
)

# k = -b/m  ->  b = -k*m
b_time = -k_time * m_total

# Error propagation for b = -k*m
delta_b_time = np.sqrt((m_total * delta_k_time)**2 + (k_time * delta_m_total)**2)

print("Constant in exponent =", k_time, "+/-", delta_k_time, "1/s")
print("Drag coefficient from v(t) =", b_time, "+/-", delta_b_time, "kg/s")

# Create fitted line
t_line = np.linspace(np.min(time_log), np.max(time_log), 100)
ln_v_line = linear_model(t_line, k_time, c_time)

# Plot the linearized graph
# Fig. 6
plt.figure()
plt.plot(time_log, ln_velocity, 'o')
plt.plot(t_line, ln_v_line)
plt.xlabel("Time (s)")
plt.ylabel("ln(Velocity)")
plt.title("Linearized velocity decay")
plt.legend(["Experimental data", "Linear fit"])
plt.show()

# --------------------------------------------------
# C5. Compare the two values of b
# --------------------------------------------------
print("C5 correctly implemented")

# R^2 for F(v) method
force_pred = linear_model(velocity_C, k, c)
ss_res_force = np.sum((force_C - force_pred)**2)
ss_tot_force = np.sum((force_C - np.mean(force_C))**2)
r2_force = 1 - ss_res_force / ss_tot_force

# R^2 for ln(v) vs t method
ln_velocity_pred = linear_model(time_log, k_time, c_time)
ss_res_time = np.sum((ln_velocity - ln_velocity_pred)**2)
ss_tot_time = np.sum((ln_velocity - np.mean(ln_velocity))**2)
r2_time = 1 - ss_res_time / ss_tot_time

# Residuals
residual_force = force_C - force_pred
residual_time = ln_velocity - ln_velocity_pred

# Fig. 7: Residual plot for F(v)
plt.figure()
plt.plot(velocity_C, residual_force, 'o')
plt.plot([np.min(velocity_C), np.max(velocity_C)], [0, 0])

x_min_force = np.min(velocity_C)
x_max_force = np.max(velocity_C)
x_range_force = x_max_force - x_min_force
plt.xlim(x_min_force - 0.1 * x_range_force, x_max_force + 0.1 * x_range_force)

residual_limit_force = 5 * np.max(np.abs(residual_force))
plt.ylim(-residual_limit_force, residual_limit_force)

plt.xlabel("Velocity (m/s)")
plt.ylabel("Residual (N)")
plt.title("Residual plot for F(v) linear fit")
plt.legend(["Residuals", "Zero line"])
plt.show()

# Fig. 8: Residual plot for ln(v) vs t
plt.figure()
plt.plot(time_log, residual_time, 'o')
plt.plot([np.min(time_log), np.max(time_log)], [0, 0])
x_min_time = np.min(time_log)
x_max_time = np.max(time_log)
x_range_time = x_max_time - x_min_time
plt.xlim(x_min_time - 0.1 * x_range_time, x_max_time + 0.1 * x_range_time)
residual_limit_time = 5 * np.max(np.abs(residual_time))
plt.ylim(-residual_limit_time, residual_limit_time)
plt.xlabel("Time (s)")
plt.ylabel("Residual")
plt.title("Residual plot for ln(v) vs t linear fit")
plt.legend(["Residuals", "Zero line"])
plt.show()

# Print comparison
print()
print("Comparison of drag coefficients:")
print("b from F(v) =", b_force, "+/-", delta_b_force, "kg/s")
print("b from v(t) =", b_time, "+/-", delta_b_time, "kg/s")

print()
print("R^2 for F(v) method =", r2_force)
print("R^2 for v(t) method =", r2_time)

# Select b according to R^2
if r2_force > r2_time:
    b = b_force
    delta_b = delta_b_force
    print("F(v) gives the better linear fit based on R^2.")
else:
    b = b_time
    delta_b = delta_b_time
    print("v(t) gives the better linear fit based on R^2.")

print("Final b =", b, "+/-", delta_b, "kg/s")
print("Researcher: inspect Fig. 7 and Fig. 8. If the residuals show a clear systematic pattern, reconsider the fitting model.")
# R^2 gives a quantitative measure of the linear fit.
# Residual plots are inspected for random scatter around zero.
# Reference: https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd44.htm?utm_source=chatgpt.com


# ==========================================
# Part D: Terminal velocity of the cart
# on an inclined track
# ==========================================

# mass1 and mass2 are used to raise the track,
# so they are no longer part of the moving system
m_D = m_cart + m_magnet
delta_m_D = np.sqrt(delta_m_cart**2 + delta_m_magnet**2)

# Inclination angle used in Part D
theta_D_deg = 2.00
delta_theta_D_deg = 0.05

theta_D = np.deg2rad(theta_D_deg)
delta_theta_D = np.deg2rad(delta_theta_D_deg)


# --------------------------------------------------
# Load and clean Part D data
# --------------------------------------------------

data_D = np.loadtxt("d.txt", encoding="utf-8-sig")
time_D = data_D[:, 0]
velocity_D = data_D[:, 1]

# Zero-velocity data at the beginning are removed because recording started
# before the cart was released.
# Recording was stopped before the cart reached the bumper, so no extra
# data at the end need to be removed.
nonzero_D = velocity_D != 0
time_D = time_D[nonzero_D]
velocity_D = velocity_D[nonzero_D]


# --------------------------------------------------
# D1. Graph of velocity as a function of time
# --------------------------------------------------
print("D1 correctly implemented")

# Fig. 9
plt.figure()
plt.plot(time_D, velocity_D)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity of the cart on an inclined track")
plt.show()


# --------------------------------------------------
# Find the experimental terminal velocity
# --------------------------------------------------

# From visual inspection of the velocity-time graph,
# the velocity becomes approximately constant after t = 2.0 s.
terminal_region = time_D >= 2.0
time_terminal = time_D[terminal_region]
velocity_terminal = velocity_D[terminal_region]


# Mean terminal velocity
v_terminal_exp = np.mean(velocity_terminal)

# Standard deviation
N_terminal = len(velocity_terminal)
s_terminal = np.sqrt(np.sum((velocity_terminal - v_terminal_exp)**2) / (N_terminal - 1))

# Standard error
delta_v_terminal_exp = s_terminal / N_terminal**0.5

print()
print("Experimental terminal velocity =", v_terminal_exp, "+/-", delta_v_terminal_exp, "m/s")


# --------------------------------------------------
# Show the selected terminal-velocity region
# --------------------------------------------------

# Fig. 10
plt.figure()
plt.plot(time_D, velocity_D)
plt.plot(time_terminal, velocity_terminal, 'o')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Selected terminal-velocity region")
plt.legend(["Experimental data", "Terminal region"])
plt.show()

print("Terminal-velocity region selected as t >= 2.0 s based on visual inspection.")


# --------------------------------------------------
# D2. Theoretical terminal velocity
# Eq. 7: v_T = m*g*sin(theta)/b
# --------------------------------------------------
print("D2 correctly implemented")

v_terminal_theory = m_D * g * np.sin(theta_D) / b


# --------------------------------------------------
# Uncertainty of theoretical terminal velocity
# --------------------------------------------------

# v_T = m*g*sin(theta)/b

dv_dm = g * np.sin(theta_D) / b
dv_dg = m_D * np.sin(theta_D) / b
dv_dtheta = m_D * g * np.cos(theta_D) / b
dv_db = -m_D * g * np.sin(theta_D) / b**2

delta_v_terminal_theory = np.sqrt(
    (dv_dm * delta_m_D)**2 +
    (dv_dg * delta_g)**2 +
    (dv_dtheta * delta_theta_D)**2 +
    (dv_db * delta_b)**2
)

print("Theoretical terminal velocity =", v_terminal_theory, "+/-", delta_v_terminal_theory, "m/s")


# --------------------------------------------------
# Compare experimental and theoretical terminal velocity
# --------------------------------------------------

difference_terminal = abs(v_terminal_exp - v_terminal_theory)

combined_uncertainty = np.sqrt(delta_v_terminal_exp**2 + delta_v_terminal_theory**2)

criterion_95 = 2 * combined_uncertainty

print()
print("Comparison of terminal velocities:")
print("Experimental v_T =", v_terminal_exp, "+/-", delta_v_terminal_exp, "m/s")
print("Theoretical v_T =", v_terminal_theory, "+/-", delta_v_terminal_theory, "m/s")
print("Absolute difference =", difference_terminal, "m/s")
print("Combined uncertainty =", combined_uncertainty, "m/s")
print("95% criterion =", criterion_95, "m/s")

if difference_terminal <= criterion_95:
    print("The experimental and theoretical values agree within the 95% uncertainty criterion.")
else:
    print("The experimental and theoretical values do not agree within the 95% uncertainty criterion.")
