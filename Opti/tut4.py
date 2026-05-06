import math

def f(x):
    # The objective function to maximize
    return -2 * x**3 + x + math.exp(x)

def golden_section_search():
    # Initial parameters
    a = 0.0
    b = 1.5
    eps = 0.05
    phi = (math.sqrt(5) - 1) / 2  # The Golden Ratio ~ 0.618034
    
    # Calculate required iterations (Omega)
    omega = math.ceil(math.log(eps / (b - a), phi))
    print(f"Calculated number of iterations (Omega): {omega}\n")
    
    # Define table header
    header = (f"{'k':<3} | {'a^k':<9} | {'b^k':<9} | {'alpha^k':<9} | {'beta^k':<9} | "
              f"{'f(a^k)':<9} | {'f(b^k)':<9} | {'f(alpha^k)':<10} | {'f(beta^k)':<10} | {'b^k - a^k':<9}")
    
    print(header)
    print("-" * len(header))
    
    # Run the iterations (including the final state where width < eps)
    for k in range(omega + 1):
        width = b - a
        
        # Calculate test points
        alpha = b - phi * width
        beta = a + phi * width
        
        # Evaluate function
        f_a = f(a)
        f_b = f(b)
        f_alpha = f(alpha)
        f_beta = f(beta)
        
        # Print row data
        print(f"{k:<3} | {a:<9.6f} | {b:<9.6f} | {alpha:<9.6f} | {beta:<9.6f} | "
              f"{f_a:<9.6f} | {f_b:<9.6f} | {f_alpha:<10.6f} | {f_beta:<10.6f} | {width:<9.6f}")
        
        if k == omega:
            break  # Stop after displaying the final state
            
        # Update interval bounds for MAXIMIZATION
        if f_alpha > f_beta:
            b = beta  # the maximum lies in [a, beta]
        else:
            a = alpha # the maximum lies in [alpha, b]

if __name__ == "__main__":
    golden_section_search()