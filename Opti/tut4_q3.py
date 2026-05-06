def f(x1, x2):
    # The objective function to minimise from Q3
    return x1**2 + x2**2 - 2*x1 + 4*x2 + 1

def tabu_search():
    # Initial parameters
    x = [0.8, -1.5]
    eps = 0.3
    k = 0
    
    # Trackers for Tabu rules and termination
    incumbent_val = f(x[0], x[1])
    tabu_list = {} # Maps the forbidden coordinate tuple to its expiration iteration
    tenure = 2
    stagnation = 0
    
    # Define the correctly ordered table header
    header = (f"{'k':<3} | {'eps':<5} | {'x^k':<17} | "
              f"{'N_eps(x^k) [Right, Up, Left, Down]':<64} | {'f(x^k)':<8} | {'Values of f in N_eps':<40}")
    
    print(header)
    print("-" * len(header))
    
    # Run until the incumbent hasn't improved for 2 consecutive iterations
    while stagnation < 2:
        current_f = f(x[0], x[1])
        
        # Calculate the 4 neighbors: Right (+x1), Up (+x2), Left (-x1), Down (-x2)
        neighbors = [
            [round(x[0] + eps, 3), round(x[1], 3)],
            [round(x[0], 3), round(x[1] + eps, 3)],
            [round(x[0] - eps, 3), round(x[1], 3)],
            [round(x[0], 3), round(x[1] - eps, 3)]
        ]
        
        # Evaluate function at all 4 neighbors
        f_vals = [f(n[0], n[1]) for n in neighbors]
        
        # Formatting elements for the terminal table
        str_x = f"[{x[0]:.3f}, {x[1]:.3f}]"
        str_N_list = []
        str_f_list = []
        
        best_valid_f = float('inf')
        best_valid_idx = -1
        
        # Evaluate each neighbor against Tabu and Aspiration rules
        for i, n in enumerate(neighbors):
            n_tuple = (n[0], n[1])
            val = f_vals[i]
            
            # Check if this neighbor is currently on the Tabu list
            is_tabu = False
            if n_tuple in tabu_list and k <= tabu_list[n_tuple]:
                is_tabu = True
            
            # ASPIRATION CRITERION: Override Tabu status if the move beats the Incumbent
            if is_tabu and val < incumbent_val:
                is_tabu = False 
            
            str_N_list.append(f"[{n[0]:.3f}, {n[1]:.3f}]")
            
            if is_tabu:
                str_f_list.append("Tabu")
            else:
                str_f_list.append(f"{val:.3f}")
                # Track the best valid move
                if val < best_valid_f:
                    best_valid_f = val
                    best_valid_idx = i
        
        # Format strings for the terminal
        str_N = "[" + ", ".join(str_N_list) + "]"
        str_fN = "[" + ", ".join(str_f_list) + "]"
        
        # Print the row
        print(f"{k:<3} | {eps:<5.3f} | {str_x:<17} | {str_N:<64} | {current_f:<8.3f} | {str_fN:<40}")
        
        # If all moves are somehow tabu and don't meet aspiration (rare), break safety
        if best_valid_idx == -1:
            break 
            
        next_x = neighbors[best_valid_idx]
        
        # Update Incumbent and Stagnation Tracker
        if best_valid_f < incumbent_val:
            incumbent_val = best_valid_f
            stagnation = 0 # Reset stagnation since we improved
        else:
            stagnation += 1 # Increment stagnation since we didn't improve
            
        # Update Tabu List: 
        # By moving from x to next_x, the reverse move (returning to x) becomes forbidden
        tabu_list[(round(x[0], 3), round(x[1], 3))] = k + tenure
        
        # Commit the move
        x = next_x
        k += 1

if __name__ == "__main__":
    tabu_search()