import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

plt.close('all')


# ==========================================
# Part 1: Experiment log and spring constants
# ==========================================

m_cart = 0.8492           # kg
delta_m_cart = 0.0001     # kg

g = 9.79127               # m/s^2
delta_g = 0.000005        # m/s^2

# Spring 1
m1_initial = 0.1603
delta_m1_initial = 0.0001
m1_final = 0.1240
delta_m1_final = 0.0001
delta_x1 = 0.099
delta_delta_x1 = 0.0005

# Spring 2
m2_initial = 0.1713
delta_m2_initial = 0.0001
m2_final = 0.1340
delta_m2_final = 0.0001
delta_x2 = 0.099
delta_delta_x2 = 0.0005

# Spring constants
delta_m1 = m1_initial - m1_final
delta_delta_m1 = np.sqrt(delta_m1_initial**2 + delta_m1_final**2)
k1 = delta_m1 * g / delta_x1
delta_k1 = k1 * np.sqrt((delta_delta_m1 / delta_m1)**2 + (delta_g / g)**2 + (delta_delta_x1 / delta_x1)**2)

delta_m2 = m2_initial - m2_final
delta_delta_m2 = np.sqrt(delta_m2_initial**2 + delta_m2_final**2)
k2 = delta_m2 * g / delta_x2
delta_k2 = k2 * np.sqrt((delta_delta_m2 / delta_m2)**2 + (delta_g / g)**2 + (delta_delta_x2 / delta_x2)**2)

# Theoretical natural angular frequency
omega_0 = np.sqrt((k1 + k2) / m_cart)
delta_omega_0 = 0.5 * omega_0 * np.sqrt(((delta_k1**2 + delta_k2**2) / (k1 + k2)**2) + (delta_m_cart / m_cart)**2)

print("k1 =", k1, "+/-", delta_k1, "N/m")
print("k2 =", k2, "+/-", delta_k2, "N/m")
print("Theoretical omega_0 =", omega_0, "+/-", delta_omega_0, "rad/s")


# ==========================================
# Part 2: Import and clean experimental data
# ==========================================

data_1 = np.loadtxt("exp3_1.txt", encoding="utf-8-sig")
data_2 = np.loadtxt("exp3_2.txt", encoding="utf-8-sig")

time_1 = data_1[:, 0]
position_1 = data_1[:, 1]
time_2 = data_2[:, 0]
position_2 = data_2[:, 1]

# Based on visual inspection, the cart had already stopped during the final 10% of each recording.
# This region is used to estimate the final equilibrium position.
N_eq_1 = int(0.1 * len(position_1))
N_eq_2 = int(0.1 * len(position_2))
x_eq_1 = np.mean(position_1[-N_eq_1:])
x_eq_2 = np.mean(position_2[-N_eq_2:])

# Shift the final equilibrium position to x = 0
position_1 = position_1 - x_eq_1
position_2 = position_2 - x_eq_2

# In both experiments, the cart was first displaced in the positive direction and then released.
# Therefore, the maximum positive displacement is used as the initial released position.
idx_start_1 = np.argmax(position_1)
idx_start_2 = np.argmax(position_2)

time_1 = time_1[idx_start_1:]
position_1 = position_1[idx_start_1:]
time_2 = time_2[idx_start_2:]
position_2 = position_2[idx_start_2:]

# Reset the release time to t = 0
time_1 = time_1 - time_1[0]
time_2 = time_2 - time_2[0]


# ==========================================
# Part 3: Harmonic oscillation experiment
# ==========================================

def analyse_experiment(time, position, number):
    # Find positive turning points only
    peaks, _ = find_peaks(position)

    # Include the initial positive release point because find_peaks does not detect endpoints
    peaks = np.insert(peaks, 0, 0)

    t_turn = time[peaks]
    A = position[peaks]

    # Based on visual inspection, the final five detected peaks occur after
    # the oscillation has nearly stopped and are dominated by small measurement fluctuations.
    t_turn = t_turn[:-5]
    A = A[:-5]

    plt.figure()
    plt.plot(time, position)
    plt.plot(t_turn, A, 'o')
    plt.xlabel("Time (s)")
    plt.ylabel("Position from equilibrium (m)")
    plt.title("Position as a function of time - Experiment " + str(number))
    plt.legend(["Experimental data", "Positive turning points"])
    plt.show()

    return t_turn, A


t_turn_1, A_1 = analyse_experiment(time_1, position_1, 1)
t_turn_2, A_2 = analyse_experiment(time_2, position_2, 2)


# ==========================================
# Part 4: Determine the damping mechanism
# ==========================================

def linear_model(t, k, c):
    return k * t + c

def calculate_r2(y, y_fit):
    return 1 - np.sum((y - y_fit)**2) / np.sum((y - np.mean(y))**2)

def fit_line(x, y):
    fit, cov = curve_fit(linear_model, x, y)
    y_fit = linear_model(x, fit[0], fit[1])
    return fit[0], fit[1], y_fit, cov, calculate_r2(y, y_fit)

def plot_fit(x, y, y_fit, ylabel, title):
    plt.figure()
    plt.plot(x, y, 'o')
    plt.plot(x, y_fit)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(["Experimental data", "Linear fit"])
    plt.show()

def plot_residual(x, residual, ylabel, title):
    plt.figure()
    plt.plot(x, residual, 'o')
    plt.plot([np.min(x), np.max(x)], [0, 0])
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


# Experiment 1
k_linear_1, c_linear_1, A_fit_1, cov_linear_1, r2_linear_1 = fit_line(t_turn_1, A_1)
ln_A_1 = np.log(A_1)
k_log_1, c_log_1, ln_A_fit_1, cov_log_1, r2_log_1 = fit_line(t_turn_1, ln_A_1)

# Experiment 2
k_linear_2, c_linear_2, A_fit_2, cov_linear_2, r2_linear_2 = fit_line(t_turn_2, A_2)
ln_A_2 = np.log(A_2)
k_log_2, c_log_2, ln_A_fit_2, cov_log_2, r2_log_2 = fit_line(t_turn_2, ln_A_2)

# Fit graphs
plot_fit(t_turn_1, A_1, A_fit_1, "Amplitude (m)", "Amplitude as a function of time - Experiment 1")
plot_fit(t_turn_1, ln_A_1, ln_A_fit_1, "ln(Amplitude)", "ln(Amplitude) as a function of time - Experiment 1")
plot_fit(t_turn_2, A_2, A_fit_2, "Amplitude (m)", "Amplitude as a function of time - Experiment 2")
plot_fit(t_turn_2, ln_A_2, ln_A_fit_2, "ln(Amplitude)", "ln(Amplitude) as a function of time - Experiment 2")

# Residuals
residual_linear_1 = A_1 - A_fit_1
residual_log_1 = ln_A_1 - ln_A_fit_1
residual_linear_2 = A_2 - A_fit_2
residual_log_2 = ln_A_2 - ln_A_fit_2

plot_residual(t_turn_1, residual_linear_1, "Residual (m)", "Residuals of A(t) fit - Experiment 1")
plot_residual(t_turn_1, residual_log_1, "Residual", "Residuals of ln(A) fit - Experiment 1")
plot_residual(t_turn_2, residual_linear_2, "Residual (m)", "Residuals of A(t) fit - Experiment 2")
plot_residual(t_turn_2, residual_log_2, "Residual", "Residuals of ln(A) fit - Experiment 2")

print()
print("Experiment 1:")
print("R^2 for A(t) =", r2_linear_1)
print("R^2 for ln(A) vs t =", r2_log_1)

print()
print("Experiment 2:")
print("R^2 for A(t) =", r2_linear_2)
print("R^2 for ln(A) vs t =", r2_log_2)

print()
print("Inspect the residual plots together with R^2 to determine which damping model is more appropriate.")


# ==========================================
# Part 5: Damping strength
# ==========================================

# Experiment 1: sliding-friction model
delta_k_linear_1 = np.sqrt(cov_linear_1[0, 0])
mu_k_1 = -np.pi * omega_0 * k_linear_1 / (2 * g)
delta_mu_k_1 = abs(mu_k_1) * np.sqrt((delta_k_linear_1 / k_linear_1)**2 + (delta_omega_0 / omega_0)**2 + (delta_g / g)**2)

print()
print("Experiment 1:")
print("mu_k =", mu_k_1, "+/-", delta_mu_k_1)

# Experiment 2: drag-model estimate.
# The model should be evaluated using both R^2 and the residual plot.
gamma_2 = -k_log_2
delta_gamma_2 = np.sqrt(cov_log_2[0, 0])
tau_2 = 1 / gamma_2
delta_tau_2 = delta_gamma_2 / gamma_2**2

omega_1_theory = np.sqrt(omega_0**2 - gamma_2**2)
delta_omega_1_theory = np.sqrt((omega_0 * delta_omega_0)**2 + (gamma_2 * delta_gamma_2)**2) / omega_1_theory

print()
print("Experiment 2 - conditional pure-drag estimate:")
print("gamma =", gamma_2, "+/-", delta_gamma_2, "1/s")
print("tau =", tau_2, "+/-", delta_tau_2, "s")
print("Note: the pure-drag model is not supported by the current residual analysis.")


# ==========================================
# Part 6: Experimental angular frequency
# ==========================================

def experimental_frequency(t_turn):
    # Consecutive positive turning points are one full period apart.
    n = np.arange(len(t_turn))
    fit_period, cov_period = curve_fit(linear_model, n, t_turn)
    T = fit_period[0]
    delta_T = np.sqrt(cov_period[0, 0])
    omega_exp = 2 * np.pi / T
    delta_omega_exp = omega_exp * delta_T / T
    return omega_exp, delta_omega_exp


omega_exp_1, delta_omega_exp_1 = experimental_frequency(t_turn_1)
omega_exp_2, delta_omega_exp_2 = experimental_frequency(t_turn_2)


# ==========================================
# Part 7: 95% comparison
# ==========================================

def compare_95(omega_exp, delta_omega_exp, omega_theory, delta_omega_theory, number):
    difference = abs(omega_exp - omega_theory)
    combined_uncertainty = np.sqrt(delta_omega_exp**2 + delta_omega_theory**2)
    criterion_95 = 2 * combined_uncertainty

    print()
    print("Experiment", number)
    print("Experimental omega =", omega_exp, "+/-", delta_omega_exp, "rad/s")
    print("Theoretical omega =", omega_theory, "+/-", delta_omega_theory, "rad/s")
    print("Absolute difference =", difference, "rad/s")
    print("95% criterion =", criterion_95, "rad/s")

    if difference <= criterion_95:
        print("The experimental and theoretical values agree within the 95% uncertainty criterion.")
    else:
        print("The experimental and theoretical values do not agree within the 95% uncertainty criterion.")


compare_95(omega_exp_1, delta_omega_exp_1, omega_0, delta_omega_0, 1)

# This comparison is conditional because omega_1_theory assumes a pure-drag model,
# which is not supported by the current Experiment 2 residual plots.
compare_95(omega_exp_2, delta_omega_exp_2, omega_1_theory, delta_omega_1_theory, 2)

print()
print("Experiment 2 frequency comparison is conditional on the pure-drag model.")
