import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(42)




# functions for EM algorithm

def gaussian_pdf(x, mean, std):
    variance = std ** 2
    coefficient = 1.0 / np.sqrt(2 * np.pi * variance)
    exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
    return coefficient * exponent

def e_step(alpha, y_hat, y_I_est, y_F_est, sigma_v):
    L = len(y_I_est)
    M = len(alpha)
    probs = np.zeros((L, M))
    
    for l in range(L):
        mu_l = (1 - alpha) * y_I_est[l] + alpha * y_F_est[l]
        probs[l, :] = gaussian_pdf(y_hat, mu_l, sigma_v)
        
    # Normalize across lines (axis=0) to get responsibilities
    gamma = probs / np.sum(probs, axis=0)
    return gamma

def m_step(alpha, y_hat, gamma):
    L = gamma.shape[0]
    y_I_new = np.zeros(L)
    y_F_new = np.zeros(L)
    
    for l in range(L):
        gamma_l = gamma[l, :]
        
        A11 = np.sum(gamma_l * (1 - alpha)**2)
        A12 = np.sum(gamma_l * (1 - alpha) * alpha)
        A22 = np.sum(gamma_l * alpha**2)
        A = np.array([[A11, A12], [A12, A22]])
        
        b1 = np.sum(gamma_l * (1 - alpha) * y_hat)
        b2 = np.sum(gamma_l * alpha * y_hat)
        b = np.array([b1, b2])
        
        theta_new = np.linalg.solve(A, b)
        y_I_new[l], y_F_new[l] = theta_new
        
    return y_I_new, y_F_new

# EM Execution

# values we choose
L = 5
M = 1000
sigma_v = 0.5

x_I, x_F = 0, 10

y_I_ = np.linspace(2.0, 7.0, L)
y_F_ = np.linspace(8.0, 3.0, L)

# y_I_est = np.random.uniform(1.0, 9.0, size=L)
# y_F_est = np.random.uniform(1.0, 9.0, size=L)

y_I_est = np.linspace(1.0, 9.0, L)
y_F_est = np.linspace(9.0, 1.0, L)

# actual data
z = np.random.randint(0, L, size=M)

alpha = np.random.uniform(0, 1, size=M)

y_points = (1 - alpha) * y_I_[z] + alpha * y_F_[z]

v = np.random.normal(0, sigma_v, size=M)
y_hat = y_points + v

x_m = (1 - alpha) * x_I + alpha * x_F



max_iters = 100
tolerance = 1e-4
log_filename = "em_execution_log.txt"

with open(log_filename, "w") as log_file:
    
    print("=== STARTING EM ALGORITHM ===", file=log_file)
    print(f"Initial Guesses:", file=log_file)
    for l in range(L):
        print(f"  Line {l+1}: y_I = {y_I_est[l]:.4f}, y_F = {y_F_est[l]:.4f}", file=log_file)
    print("", file=log_file)

    for i in range(max_iters):
        print(f"Processing Iteration {i+1}...", end='\r')
        
        print(f"=======================================================", file=log_file)
        print(f"                     ITERATION {i+1}                   ", file=log_file)
        print(f"=======================================================", file=log_file)
        
        # E-Step
        gamma = e_step(alpha, y_hat, y_I_est, y_F_est, sigma_v)
        
        print("  [E-Step] Responsibilities (Gamma) for ALL points:", file=log_file)
        print("  Pt_idx | alpha  | y_hat  ||", end='', file=log_file)
        for l in range(L):
            print(f" P(Line {l+1}) |", end='', file=log_file)
        print("  -------------------------------------------------", file=log_file)
        for pt_idx in range(M):
            print(f"  {pt_idx:<6} | {alpha[pt_idx]:.4f} | {y_hat[pt_idx]:.4f} ||", end='', file=log_file)
            for l in range(L):
                print(f" {gamma[l, pt_idx]:.4f}    |", end='', file=log_file)
            print("", file=log_file)
        
        # M-Step
        y_I_new, y_F_new = m_step(alpha, y_hat, gamma)
        
        print(f"\n  [M-Step] New calculated line parameters:", file=log_file)
        for l in range(L):
            print(f"      Line {l+1}: y_I = {y_I_new[l]:.4f}, y_F = {y_F_new[l]:.4f}", file=log_file)
        
        # Check Convergence
        diff = np.max(np.abs(y_I_new - y_I_est)) + np.max(np.abs(y_F_new - y_F_est))
        print(f"  Total parameter shift this iteration: {diff:.6f}\n", file=log_file)
        
        if diff < tolerance:
            print(f"=== CONVERGED at iteration {i+1} ===", file=log_file)
            break
            
        y_I_est, y_F_est = y_I_new, y_F_new
        
        if i == max_iters - 1:
            print(f"\nFinished {max_iters} iterations without full convergence.")

# plotting results
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(x_m, y_hat, color='#ff4d4d', alpha=0.3, s=15, edgecolors='none', label='Measurements $\\hat{y}^{[m]}$')
x_line = np.array([x_I, x_F])
colors = cm.rainbow(np.linspace(0, 1, L))
for l in range(L):
    # True lines
    y_line_true = np.array([y_I_[l], y_F_[l]])
    ax.plot(x_line, y_line_true, color=colors[l], linestyle='--', alpha=0.5, linewidth=2, 
            label='True Lines' if l == 0 else "")
    
    # Estimated lines
    y_line_est = np.array([y_I_new[l], y_F_new[l]])
    ax.plot(x_line, y_line_est, color=colors[l], linewidth=3, 
            label='Estimated Lines' if l == 0 else "")

ax.set_xlabel('$x(\\alpha)$', fontsize=12)
ax.set_ylabel('$y$', fontsize=12)
ax.set_title(f'EM Algorithm Convergence: {L} Latent Lines', fontsize=14, pad=15)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

ax.grid(True, linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()