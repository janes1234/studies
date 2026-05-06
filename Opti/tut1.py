import numpy as np

# Define the vectors
v = np.array([[-1], [0], [2]])
w = np.array([[3], [2], [-1]])

# Compute inner products
v_T_w = v.T @ w
w_T_v = w.T @ v

print("v^T w =", v_T_w[0][0])
print("w^T v =", w_T_v[0][0])

# Compute outer products
v_w_T = v @ w.T
w_v_T = w @ v.T

print("v w^T =\n", v_w_T)
print("w v^T =\n", w_v_T)


A = np.array([[2, 1, 1], 
              [0, 2, 1], 
              [-2, 4, 2]])

# Calculate determinant
det_A = np.linalg.det(A)
print(f"Determinant of A: {round(det_A)}")

# Compute inverse
A_inv = np.linalg.inv(A)
print("Inverse of A:\n", A_inv)

# Verify A * A_inv = Identity
# We use np.round to clean up tiny floating point inaccuracies (e.g., 0.000000000000004)
identity_check = np.round(A @ A_inv)
print("A @ A_inv:\n", identity_check)

# Assuming v, w, A, and A_inv are already defined from Q1 and Q2
# 1. Compute using the Sherman-Morrison formula
numerator = A_inv @ v @ w.T @ A_inv
denominator = 1 + (w.T @ A_inv @ v)
formula_result = A_inv - (numerator / denominator)

# 2. Compute directly by taking the inverse of (A + vw^T)
direct_result = np.linalg.inv(A + v @ w.T)

print("Sherman-Morrison Formula Result:\n", np.round(formula_result, 4))
print("\nDirect Inverse Result:\n", np.round(direct_result, 4))

# Check if they match closely
print("\nDo the matrices match?", np.allclose(formula_result, direct_result))

import numpy as np

B = np.array([[2, -1, 0], 
              [-1, 2, 0], 
              [0, 0, 2]])

# (b) Verify Rank using determinant
det_B = np.linalg.det(B)
rank_B = np.linalg.matrix_rank(B)
print(f"Determinant of B: {round(det_B)}")
print(f"Rank of B: {rank_B}")

# (h) Verify Positive Definiteness
# We can check if all eigenvalues are strictly greater than zero
eigenvalues = np.linalg.eigvals(B)
is_positive_definite = np.all(eigenvalues > 0)

print(f"Eigenvalues: {eigenvalues}")
print(f"Is B Positive Definite? {is_positive_definite}")


B = np.array([[2, -1, 0], 
              [-1, 2, 0], 
              [0, 0, 2]])

# Calculate eigenvalues and eigenvectors
eigenvalues, _ = np.linalg.eig(B)

# Note: NumPy might return them in a different order (e.g., 3, 1, 2)
print("Eigenvalues of B:", np.round(eigenvalues))

# Calculate both eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(B)

# The eigenvectors are returned as columns in the matrix
print("Eigenvectors of B (each column is an eigenvector):\n", np.round(eigenvectors, 4))