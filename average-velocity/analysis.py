import numpy as np 
from scipy.stats import norm 
import matplotlib.pyplot as plt 
 
plt.close('all') # start fresh 
 
# Helper functions for formatting uncertainties to one significant digit
def one_sig(x):
    """Return x rounded to one significant digit."""
    if x == 0 or not np.isfinite(x):
        return x
    sign = np.sign(x)
    abs_x = abs(x)
    scale = 10 ** np.floor(np.log10(abs_x))
    mant = int(np.floor(abs_x / scale + 0.5))
    return sign * mant * scale

def decimals_for_error(x):
    """Number of decimal places needed to show a one-significant-digit error."""
    err = abs(one_sig(x))
    if err == 0 or not np.isfinite(err) or err >= 1:
        return 0
    return -int(np.floor(np.log10(err)))

def fmt_err(x):
    """Format an error value to one significant digit."""
    err = one_sig(x)
    dp = decimals_for_error(x)
    if err >= 1:
        return f"{err:.0f}"
    return f"{err:.{dp}f}"

def fmt_pair(value, error):
    """Format a value and its error with the same number of decimal places."""
    err = one_sig(error)
    dp = decimals_for_error(error)
    return f"{value:.{dp}f} +/- {err:.{dp}f}" 
 
 
# Part A: Experiment log and theoretical model 
 
# Directly measured / given parameters 
g = 9.79127          # gravitational acceleration, m/s^2 
delta_g = 0.000005   # uncertainty in g, m/s^2 
 
l = 0.05             # length of picket fence, m 
delta_l = 0.005      # uncertainty in l, m 
 
s = 0.415             # distance from release point to photogate, m 
delta_s = 0.005       # uncertainty in s, m 
 
theta_deg = 2           # inclination angle, degree 
delta_theta_deg = 0.05  # uncertainty in angle, degree 
 
# Convert angle to radians 
theta = np.deg2rad(theta_deg) 
delta_theta = np.deg2rad(delta_theta_deg) 
 
# 2. Theoretical average velocity through the photogate 
v_theory = np.sqrt(g * np.sin(theta) / 2) * (np.sqrt(s + l) + np.sqrt(s)) 

print("Theoretical average velocity =", v_theory, "m/s") 
 
 
# 3. Uncertainty in theoretical velocity 
dv_dg = v_theory / (2 * g) 
 
dv_dtheta = (v_theory / 2) * (np.cos(theta) / np.sin(theta)) 
 
dv_dl = np.sqrt(g * np.sin(theta) / 2) / (2 * np.sqrt(s + l)) 
 
dv_ds = np.sqrt(g * np.sin(theta) / 2) * (1 / (2 * np.sqrt(s + l)) + 1 / (2 * np.sqrt(s))) 
 
delta_v = np.sqrt( 
    (dv_dg * delta_g)**2 + 
    (dv_dtheta * delta_theta)**2 + 
    (dv_dl * delta_l)**2 + 
    (dv_ds * delta_s)**2 
) 
 
print(f"Uncertainty in theoretical velocity = {fmt_err(delta_v)} m/s") 
print(f"Theoretical result = {fmt_pair(v_theory, delta_v)} m/s") 
 
 
# part B 
 
N = 150 # 150 trails 
 
# Generate some data for this  
# demonstration. 
data = np.loadtxt("exp_raw_data_1.txt") 
 
# Fit a normal distribution to 
# the data: 
# mean and standard deviation 
mu, std = norm.fit(data)  
 
# Plot the histogram. 
plt.hist(data, bins=12, density=True, alpha=0.6, color='b', label='Experimental data') 
 
# Plot the PDF. 
xmin, xmax = plt.xlim() 
x = np.linspace(xmin, xmax, 100) 
p = norm.pdf(x, mu, std) 
 
plt.plot(x, p, 'k', linewidth=2, label='Normal fit') 
plt.xlabel("Time (s)")                # Added x-axis label with unit
plt.ylabel("Probability density")     # Added y-axis label
title = f"Fit: mean = {mu:.4f} s, std = {std:.4f} s"   # Changed title
plt.title(title) 
plt.legend()
plt.show() 
 
# Calculate mean & std deviation 
mean_t = np.mean(data) 
std_t = np.sqrt( 
    np.sum((data - mean_t)**2) / (N - 1) 
) 
std_err_t = std_t / N**0.5 
 
# Calculate outliers and clean them 
lower = mean_t - 3 * std_t 
upper = mean_t + 3 * std_t 
outliers = data[(data < lower) | (data > upper)] 
clean_data = data[(data >= lower) & (data <= upper)] 
print(f"Number of outliers removed: {len(outliers)}")
 
# 7. Outcome of the experiment 
 
# Calculate average velocity for every valid measurement 
velocity = l / clean_data 
 
# Number of measurements after removing outliers 
N_clean = len(velocity) 
 
# Mean average velocity 
mean_v = np.mean(velocity) 
 
# Standard deviation of velocity 
std_v = np.sqrt( 
    np.sum((velocity - mean_v)**2) / (N_clean - 1) 
) 
 
# Standard error of mean velocity 
std_err_v = std_v / N_clean**0.5 
 
# 8. Histogram without outliers + theoretical normal distribution 
 
plt.figure() 
 
plt.hist(velocity, bins=12, density=True, alpha=0.6, color='b', label='Measured velocity') 
 
xmin, xmax = plt.xlim() 
x = np.linspace(xmin, xmax, 100) 
 
p = norm.pdf(x, mean_v, std_v) 
 
plt.plot(x, p, 'k', linewidth=2, label='Normal fit') 
 
plt.xlabel("Average velocity (m/s)") 
plt.ylabel("Probability density") 
 
title = f"Fit: mean = {mean_v:.4f} m/s, std = {std_v:.4f} m/s"   # Changed title
plt.title(title) 
plt.legend()
plt.show() 
 
# Part B - 9 
# Compare experimental result with theoretical value 
 
difference = abs(mean_v - v_theory) 
 
ratio = difference / std_err_v 
 
print(f"Experimental value = {fmt_pair(mean_v, std_err_v)} m/s") 
print("|experimental - theoretical| / standard error =", ratio) 
 
if ratio <= 2: 
    print("The experiment AGREES with the theory.") 
else: 
    print("The experiment DISAGREES with the theory.") 
     
     
# -------------------------------------------------- 
# Part C: Statistical theory 
# -------------------------------------------------- 
 
# N = 20 
 
data_20 = clean_data[:20] 
velocity_20 = l / data_20 
 
mean_v_20 = np.mean(velocity_20) 
 
std_v_20 = np.sqrt( 
    np.sum((velocity_20 - mean_v_20)**2) / (20 - 1) 
) 
 
std_err_v_20 = std_v_20 / 20**0.5 
 
dp20 = decimals_for_error(std_err_v_20)
err20 = one_sig(std_err_v_20)

print("N = 20") 
print(f"Mean velocity = {mean_v_20:.{dp20}f} m/s") 
print(f"Standard deviation = {std_v_20:.{dp20}f} m/s") 
print(f"Standard error = {err20:.{dp20}f} m/s") 
 
plt.figure() 
 
plt.hist(velocity_20, bins=7, density=True, alpha=0.6, color='b', label='Measured velocity') 
 
xmin, xmax = plt.xlim() 
x = np.linspace(xmin, xmax, 100) 
 
p = norm.pdf(x, mean_v_20, std_v_20) 
 
plt.plot(x, p, 'k', linewidth=2, label='Normal fit') 
 
plt.xlabel("Average velocity (m/s)") 
plt.ylabel("Probability density") 
plt.title("N = 20") 
plt.legend()
plt.show() 
 
 
# N = 60 
 
data_60 = clean_data[:60] 
velocity_60 = l / data_60 
 
mean_v_60 = np.mean(velocity_60) 
 
std_v_60 = np.sqrt( 
    np.sum((velocity_60 - mean_v_60)**2) / (60 - 1) 
) 
 
std_err_v_60 = std_v_60 / 60**0.5 
 
dp60 = decimals_for_error(std_err_v_60)
err60 = one_sig(std_err_v_60)

print("N = 60") 
print(f"Mean velocity = {mean_v_60:.{dp60}f} m/s") 
print(f"Standard deviation = {std_v_60:.{dp60}f} m/s") 
print(f"Standard error = {err60:.{dp60}f} m/s") 
 
plt.figure() 
 
plt.hist(velocity_60, bins=5, density=True, alpha=0.6, color='b', label='Measured velocity') 
 
xmin, xmax = plt.xlim() 
x = np.linspace(xmin, xmax, 100) 
 
p = norm.pdf(x, mean_v_60, std_v_60) 
 
plt.plot(x, p, 'k', linewidth=2, label='Normal fit') 
 
plt.xlabel("Average velocity (m/s)") 
plt.ylabel("Probability density") 
plt.title("N = 60") 
plt.legend()
plt.show() 
 
dp150 = decimals_for_error(std_err_v)
err150 = one_sig(std_err_v)

print(f"N = 20: mean = {mean_v_20:.{dp20}f}, SD = {std_v_20:.{dp20}f}, SE = {err20:.{dp20}f}") 
print(f"N = 60: mean = {mean_v_60:.{dp60}f}, SD = {std_v_60:.{dp60}f}, SE = {err60:.{dp60}f}") 
print(f"N = 150: mean = {mean_v:.{dp150}f}, SD = {std_v:.{dp150}f}, SE = {err150:.{dp150}f}")
