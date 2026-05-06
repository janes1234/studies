def f(x1, x2):
    # The objective function to maximise from Q2
    return -(x1**3) + 3*x1 - (x2**2)

def local_search():
    # Initial parameters
    x = [1.0, -0.2]
    eps = 0.3
    k = 0
    
    # Define the correctly ordered table header
    header = (f"{'k':<3} | {'eps':<5} | {'x^k':<16} | "
              f"{'N_eps(x^k) [Right, Left, Up, Down]':<43} | {'f(x^k)':<8} | {'f in N_eps':<35}")
    
    print(header)
    print("-" * len(header))
    
    # Run until lattice resolution drops below 0.1
    while eps >= 0.1:
        current_f = f(x[0], x[1])
        
        # Calculate the 4 neighbors: Right (+x1), Left (-x1), Up (+x2), Down (-x2)
        neighbors = [
            [x[0] + eps, x[1]],
            [x[0] - eps, x[1]],
            [x[0], x[1] + eps],
            [x[0], x[1] - eps]
        ]
        
        # Evaluate the function at all 4 neighbors
        f_vals = [f(n[0], n[1]) for n in neighbors]
        
        # Format the arrays as strings for clean terminal printing
        str_x = f"[{x[0]:.3f}, {x[1]:.3f}]"
        str_N = "[" + ", ".join([f"[{n[0]:.2f}, {n[1]:.2f}]" for n in neighbors]) + "]"
        str_fN = "[" + ", ".join([f"{v:.3f}" for v in f_vals]) + "]"
        
        # Print the row with f(x^k) sandwiched between the neighbors and neighbor values
        print(f"{k:<3} | {eps:<5.3f} | {str_x:<16} | {str_N:<43} | {current_f:<8.3f} | {str_fN:<35}")
        
        # Decision logic: Find the best neighbor
        best_f = max(f_vals)
        
        if best_f > current_f:
            # If the best neighbor is an improvement, move there
            best_idx = f_vals.index(best_f)
            x = neighbors[best_idx]
        else:
            # If no neighbor is better, stay put and shrink the lattice resolution
            eps /= 2.0
        
        k += 1

if __name__ == "__main__":
    local_search()