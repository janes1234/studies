import numpy as np
import pandas as pd

# 1. Define the objective function (Maximisation)
def f(x):
    return -(x[0]**3) + 3*x[0] - (x[1]**2)

# 2. Initialise parameters
positions = np.array([
    [0.3, -0.2],
    [-0.9, 0.5],
    [-0.4, 0.5],
    [0.9, 0.7],
    [-0.4, -0.5]
])
velocities = np.zeros_like(positions)

# Hyperparameters
w = 0.8
c1 = 0.2
c2 = 0.2
rho1 = 0.5
rho2 = 0.5
v_max = 0.25
v_min = -0.25

# Initialise personal bests (pbest) and global best (gbest)
pbests = np.copy(positions)
pbest_vals = np.array([f(p) for p in pbests])

gbest_idx = np.argmax(pbest_vals)
gbest = np.copy(pbests[gbest_idx])
gbest_val = pbest_vals[gbest_idx]

# List to store tabular data
table_data = []

def record_state(t, idx, pos, vel, pbest, f_val, gbest):
    table_data.append({
        't': t,
        'i': idx + 1,
        'Position x(t)': f"[{pos[0]:.3f}, {pos[1]:.3f}]",
        'Velocity v(t)': f"[{vel[0]:.3f}, {vel[1]:.3f}]",
        'pbest p(i)': f"[{pbest[0]:.3f}, {pbest[1]:.3f}]",
        'f(x)': f"{f_val:.3f}",
        'gbest x*': f"[{gbest[0]:.3f}, {gbest[1]:.3f}]"
    })

# Record Iteration 0 (Initial State)
for i in range(5):
    record_state(0, i, positions[i], velocities[i], pbests[i], pbest_vals[i], gbest)

# 3. Perform 3 PSO Iterations
for t in range(1, 4):
    for i in range(5):
        # Calculate cognitive and social components
        r1c1 = rho1 * c1
        r2c2 = rho2 * c2
        cognitive = r1c1 * (pbests[i] - positions[i])
        social = r2c2 * (gbest - positions[i])
        
        # Update velocity
        velocities[i] = w * velocities[i] + cognitive + social
        
        # Apply velocity clamping
        velocities[i] = np.clip(velocities[i], v_min, v_max)
        
        # Update position
        positions[i] = positions[i] + velocities[i]
        
        # Evaluate new position
        current_val = f(positions[i])
        
        # Update personal best (Maximisation)
        if current_val > pbest_vals[i]:
            pbests[i] = np.copy(positions[i])
            pbest_vals[i] = current_val
            
        # Update global best (Maximisation)
        if current_val > gbest_val:
            gbest = np.copy(positions[i])
            gbest_val = current_val
            
        record_state(t, i, positions[i], velocities[i], pbests[i], current_val, gbest)

# 4. Display Results
df = pd.DataFrame(table_data)
print(df.to_string(index=False))