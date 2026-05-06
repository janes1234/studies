import numpy as np

def f(x):
    """The objective function: f(x1, x2) = 4x1*x2 - 2x1^2 - x2^4"""
    x1, x2 = x[0], x[1]
    return 4 * x1 * x2 - 2 * x1**2 - x2**4

def grad_f(x):
    """The gradient vector of the objective function."""
    x1, x2 = x[0], x[1]
    return np.array([
        4 * x2 - 4 * x1,
        4 * x1 - 4 * x2**3
    ])

def fmt(val):
    """Helper to format numbers cleanly to 3 decimal places."""
    return f"{val:8.3f}"

def print_row(k, x, grad, B, p, alpha, s, y, norm_grad):
    """Prints the variables in the multi-line vector/matrix format."""
    if k == 3: # Final iteration styling
        line1 = f"{k:<2} [\033[91m{fmt(x[0])}\033[0m]      ---           ---                  ---          ---        ---          ---             ---"
        line2 = f"   [\033[91m{fmt(x[1])}\033[0m]"
    else:
        line1 = f"{k:<2} [{fmt(x[0])}] [{fmt(grad[0])}] [{fmt(B[0,0])} {fmt(B[0,1])}] [{fmt(p[0])}] {alpha:8.3f} [{fmt(s[0])}] [{fmt(y[0])}]     {norm_grad:8.3f}"
        line2 = f"   [{fmt(x[1])}] [{fmt(grad[1])}] [{fmt(B[1,0])} {fmt(B[1,1])}] [{fmt(p[1])}]          [{fmt(s[1])}] [{fmt(y[1])}]"
    
    print(line1)
    print(line2)
    print()

def sr1_maximize_formatted(x0, B0, a=0.45, b=0.95, iterations=3):
    x = np.array(x0, dtype=float)
    B = np.array(B0, dtype=float)
    
    print("k    x^k         Grad f(x^k)         B^k                  p^k        alpha^k    s^k         y^k           ||Grad||_2")
    print("-" * 116)
    
    for k in range(iterations):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        
        # Step 3: Compute search direction p^k
        p = np.linalg.solve(B, -grad)
        
        # Step 4: Armijo-Goldstein Line Search (for MAXIMIZATION)
        alpha = 1.0
        directional_derivative = np.dot(grad, p)
        
        while f(x + alpha * p) < f(x) + a * alpha * directional_derivative:
            alpha *= b
            
        # Step 5: Update x
        s = alpha * p
        x_next = x + s
        
        # Step 6 & 7: Calculate y^k
        y = grad_f(x_next) - grad
        
        # Print the current formatted row BEFORE updating B
        print_row(k, x, grad, B, p, alpha, s, y, grad_norm)
        
        # Step 8: SR1 Update for B^{k+1}
        Bs = B @ s
        diff = y - Bs
        denominator = np.dot(diff, s)
        
        # Safety check for SR1: if denominator is close to zero, the update can blow up
        if abs(denominator) > 1e-8:
            numerator = np.outer(diff, diff)
            B_next = B + (numerator / denominator)
        else:
            B_next = B # Skip update if denominator is too small
            
        # Prepare for next iteration
        x = x_next
        B = B_next
        
    # Print the final x^k value (iteration 3) highlighted in red
    print_row(3, x, None, None, None, None, None, None, None)

if __name__ == '__main__':
    # Initial setup
    x0 = [-2.0, -2.0]
    B0 = [[-3.0, 4.0],
          [ 4.0, -45.0]]
    
    sr1_maximize_formatted(x0, B0)