import numpy as np
import math
import random

# --- Problem Definition ---
# (Same as before)
G = np.array([[10, 12, 11], [8, 8, 10]])
C = np.array([[3, 4, 5], [2, 2, 3]])
P_profit_coeffs = G - C
MHA = np.array([120, 80])
D_dimensions = 6
lb_bounds = np.zeros(D_dimensions)
ub_bounds = np.array([MHA[0], MHA[0], MHA[0], MHA[1], MHA[1], MHA[1]])

def objective_function(solution):
    return np.sum(P_profit_coeffs.flatten() * solution)

# --- MHO Algorithm Parameters ---
N_pop = 5
Max_iter = 5 # For a detailed run, even 1-2 iterations will be very long. Let's aim for 1 detailed iteration.
             # The user asked for 5, so the code will run 5, but the log for 1 will be very verbose.

# For reproducibility
random.seed(42)
np.random.seed(42)

output_lines = []

def log_print(message, indent=0):
    prefix = "  " * indent
    full_message = prefix + message
    print(full_message)
    output_lines.append(full_message)

# --- MHO Enhanced Components ---

# Sine Chaotic Map for Initialization (MHO Eq. 1, 4, 5)
# Eq 1: x_{k+1} = alpha * sin(pi * x_k) (MHO paper text form for reference)
# Eq 5 in MHO paper: Sine_chaos = alpha * sin(k_pi_x) (where x is an initial value, k is a parameter)
# For this implementation, we generate a sequence of chaotic values.
# alpha: chaos coefficient control parameter (close to 1 for superior chaotic properties)
# k_factor: parameter 'k' in MHO Eq. 5
def sine_chaotic_map_init(num_values, D_dim, current_chaos_val, alpha=0.99, k_factor=1.0):
    chaotic_sequence = np.zeros(num_values * D_dim)
    val = current_chaos_val
    for i in range(num_values * D_dim):
        val = alpha * math.sin(k_factor * math.pi * val)
        chaotic_sequence[i] = abs(val) # Use abs to keep in [0, alpha], typically mapped to [0,1]
    return chaotic_sequence, val # Return sequence and last chaos value

def initialize_population_mho(N_pop, D_dim, lb, ub, initial_chaos_seed=0.5):
    log_print("Initializing population using Sine Chaotic Map concept:", 1)
    log_print(f"  Initial chaos seed: {initial_chaos_seed:.3f}, alpha=0.99, k_factor=1.0 (for Eq.5 variant)", 2)
    
    population = np.zeros((N_pop, D_dim))
    # Generate enough chaotic values for the entire population
    total_dims_to_init = N_pop * D_dim
    
    # For detailed logging, we'll show chaotic sequence generation conceptually for the first hippo's first dim
    # In practice, a single sequence would be generated and then used.
    
    # Use a single persistent chaos_val for the entire initialization process
    # to mimic a continuous chaotic sequence
    current_chaos_val_for_all = initial_chaos_seed 
    
    for i in range(N_pop):
        log_print(f"  Initializing Hippo {i+1}:", 2)
        # Generate chaotic values for this specific hippo
        # In a strict sense, one long chaotic sequence should be used and reshaped.
        # For simplicity of generation per hippo here while maintaining the idea:
        hippo_chaotic_values, current_chaos_val_for_all = sine_chaotic_map_init(1, D_dim, current_chaos_val_for_all)
        
        for j in range(D_dim):
            # MHO Eq. 4: x_ij = lb_j + Sine_chaos_val * (ub_j - lb_j)
            # Sine_chaos_val is the chaotic value in [0,1] (abs already taken)
            sine_chaos_val = hippo_chaotic_values[j] # Use the j-th value from the hippo's sequence
            population[i, j] = lb[j] + sine_chaos_val * (ub[j] - lb[j])
            if i == 0 and j < 2 : # Log first few for detail
                 log_print(f"    Dim {j+1}: prev_chaos_val for this sequence chunk start (conceptual) = {hippo_chaotic_values[0]:.4f} -> used_chaos_val = {sine_chaos_val:.4f}", 3)
                 log_print(f"      x_{i+1},{j+1} = {lb[j]:.1f} + {sine_chaos_val:.4f} * ({ub[j]:.1f} - {lb[j]:.1f}) = {population[i,j]:.3f}", 3)

        population[i] = np.clip(population[i], lb, ub)
        log_print(f"    Hippo {i+1} Initial Raw Position (after chaos map & bounds): {np.round(population[i],3).tolist()}", 3)
    log_print(f"  Final chaos_val after all initializations: {current_chaos_val_for_all:.4f}",2)
    return population


# Helper to apply bounds
def apply_bounds(solution, lb, ub):
    return np.clip(solution, lb, ub)

# --- MHO Algorithm Implementation (Detailed) ---
def MHO_detailed(N_pop, Max_iter, lb, ub, D_dim, obj_func):
    log_print("Modified Hippopotamus Optimization (MHO) Detailed Sample Run\n")
    log_print(f"Problem: Maximize Profit for 2 crops over 3 months")
    log_print(f"Profit Coefficients (P = G-C): \n  Crop1: {P_profit_coeffs[0].tolist()}, Crop2: {P_profit_coeffs[1].tolist()}")
    log_print(f"Variable Bounds (lower): {lb.tolist()}")
    log_print(f"Variable Bounds (upper based on MHA): {ub.tolist()}\n")
    log_print(f"Parameters: N_pop = {N_pop}, Max_iter = {Max_iter}, Dimensions = {D_dim}\n")

    # Initialization (MHO Eq. 1, 4, 5)
    initial_chaos_seed = random.random() # Start with a random seed for chaos
    positions = initialize_population_mho(N_pop, D_dim, lb, ub, initial_chaos_seed=initial_chaos_seed)
    fitness = np.array([obj_func(pos) for pos in positions])

    best_idx = np.argmax(fitness)
    D_hippo_pos = positions[best_idx].copy()
    D_hippo_fitness = fitness[best_idx]
    convergence_curve = np.zeros(Max_iter)

    log_print("\n--- Iteration 0: Initial Population Details ---", 0)
    for i in range(N_pop):
        log_print(f"Hippo {i+1}: Position = {np.round(positions[i], 3).tolist()}, Fitness (Profit) = {fitness[i]:.3f}", 1)
    log_print(f"Initial Dominant Hippo (Hippo {best_idx+1}): Position = {np.round(D_hippo_pos, 3).tolist()}, Fitness = {D_hippo_fitness:.3f}\n", 1)

    # Main loop
    for t_iter_idx in range(Max_iter): # t_iter_idx is 0-indexed
        current_iter_num = t_iter_idx + 1 # Iteration number (1 to Max_iter)
        log_print(f"--- Iteration {current_iter_num}/{Max_iter} ---", 0)

        # Calculate convergence factor T (MHO Eq. 13)
        T_convergence = 1 - ((current_iter_num / Max_iter) ** 6)
        log_print(f"Convergence Factor T = 1 - (({current_iter_num}/{Max_iter})^6) = {T_convergence:.6f}", 1)

        # Store positions before phase updates for logging clarity
        positions_before_phases = positions.copy()
        fitness_before_phases = fitness.copy()

        # --- Phase 1 & 2 Loop ---
        for i_hippo_idx in range(N_pop): # i_hippo_idx is 0-indexed
            log_print(f"  Processing Hippo {i_hippo_idx+1}: Current Pos = {np.round(positions[i_hippo_idx], 3).tolist()}, Fitness = {fitness[i_hippo_idx]:.3f}", 1)
            
            original_pos_for_hippo_iter = positions[i_hippo_idx].copy() # Store pos before any phase in this iteration for this hippo
            original_fit_for_hippo_iter = fitness[i_hippo_idx]

            # Phase 1: Exploration (MHO Eq. 14, 15)
            if i_hippo_idx < N_pop / 2: # e.g., indices 0, 1 for N_pop=5
                log_print(f"    Phase 1: Exploration for Hippo {i_hippo_idx+1}", 2)
                
                # MG_i: average of some randomly selected hippos
                num_to_select = random.randint(1, N_pop)
                rand_indices = np.array(random.sample(range(N_pop), num_to_select))
                MG_i = np.mean(positions_before_phases[rand_indices], axis=0) # Use population state at start of iteration t
                log_print(f"      MG_{i_hippo_idx+1} (avg of {len(rand_indices)} random hippos {rand_indices+1} from iter start) = {np.round(MG_i,3).tolist()}", 3)

                # h factor (MHO Eq. 7)
                r1_h = random.random()
                r2_h = random.random()
                e1_h = random.randint(0,1)
                h_choice = random.choice(['v1', 'v2', 'v3'])
                if h_choice == 'v1':
                    # h = f2 * r1^t + (-e1) -> Using r2_h for f2, r1_h for r1
                    h_factor_val = r2_h * (r1_h**current_iter_num) + (-e1_h)
                    log_print(f"      h_factor (Eq.7 v1: r2*r1^iter + (-e1)): r1={r1_h:.3f}, r2={r2_h:.3f}, e1={e1_h}, iter={current_iter_num} => {h_factor_val:.4f}", 3)
                elif h_choice == 'v2':
                    # h = 2 * r1^t - 1
                    h_factor_val = 2 * (r1_h**current_iter_num) - 1
                    log_print(f"      h_factor (Eq.7 v2: 2*r1^iter - 1): r1={r1_h:.3f}, iter={current_iter_num} => {h_factor_val:.4f}", 3)
                else: # v3
                    # h = r1 * r2^t + (-e1)
                    h_factor_val = r1_h * (r2_h**current_iter_num) + (-e1_h)
                    log_print(f"      h_factor (Eq.7 v3: r1*r2^iter + (-e1)): r1={r1_h:.3f}, r2={r2_h:.3f}, e1={e1_h}, iter={current_iter_num} => {h_factor_val:.4f}", 3)
                
                I1_rand_val = random.randint(1,2)
                log_print(f"      I1_rand_val = {I1_rand_val}", 3)

                if T_convergence > 0.95: # MHO Eq. 14
                    log_print(f"      T_conv ({T_convergence:.4f}) > 0.95, using Eq. 14: X_new = X_old + h * (D_hippo - I1*MG_i)", 3)
                    delta_X_P1 = h_factor_val * (D_hippo_pos - I1_rand_val * MG_i)
                    new_pos_p1 = positions[i_hippo_idx] + delta_X_P1
                    log_print(f"        D_hippo_pos = {np.round(D_hippo_pos,3)}",4)
                    log_print(f"        I1*MG_i = {np.round(I1_rand_val * MG_i,3)}",4)
                    log_print(f"        (D_hippo - I1*MG_i) = {np.round(D_hippo_pos - I1_rand_val * MG_i,3)}",4)
                    log_print(f"        delta_X = {h_factor_val:.4f} * prev_vector = {np.round(delta_X_P1,3)}",4)
                else: # MHO Eq. 15
                    log_print(f"      T_conv ({T_convergence:.4f}) <= 0.95, using Eq. 15", 3)
                    r6_rand = random.random()
                    log_print(f"      r6_rand = {r6_rand:.3f}", 3)
                    if r6_rand > 0.5:
                        log_print(f"        r6_rand > 0.5, using X_new = X_old + h * (MG_i - D_hippo)", 4) # h2 chosen similarly to h1
                        delta_X_P1 = h_factor_val * (MG_i - D_hippo_pos)
                        new_pos_p1 = positions[i_hippo_idx] + delta_X_P1
                    else:
                        log_print(f"        r6_rand <= 0.5, random exploration: X_new_j = lb_j + r7*(ub_j-lb_j)", 4)
                        r7_vec = np.random.rand(D_dim)
                        new_pos_p1 = lb_bounds + r7_vec * (ub_bounds - lb_bounds)
                        log_print(f"          r7_vec = {np.round(r7_vec,3)}",5)
                
                new_pos_p1_bounded = apply_bounds(new_pos_p1, lb_bounds, ub_bounds)
                fit_p1 = obj_func(new_pos_p1_bounded)
                log_print(f"      Cand. Pos (P1) before bounds = {np.round(new_pos_p1,3)}", 3)
                log_print(f"      Cand. Pos (P1) after bounds  = {np.round(new_pos_p1_bounded,3)}, Fitness = {fit_p1:.3f}", 3)

                if fit_p1 > fitness[i_hippo_idx]:
                    positions[i_hippo_idx] = new_pos_p1_bounded
                    fitness[i_hippo_idx] = fit_p1
                    log_print(f"      Hippo {i_hippo_idx+1} updated in Phase 1.", 3)
            
            else: # Phase 2: Defense Against Predators
                log_print(f"    Phase 2: Defense for Hippo {i_hippo_idx+1} (MHO: 'consistent with original HO', simplified here)", 2)
                predator_pos = lb_bounds + np.random.rand(D_dim) * (ub_bounds - lb_bounds)
                log_print(f"      Random Predator Position = {np.round(predator_pos,3)}", 3)
                
                # Simplified defense: small random perturbation, potentially influenced by dominant hippo (safety)
                # This is a placeholder as MHO paper does not specify new detailed mechanics for this phase.
                r_def_strat = random.random()
                step_size_factor = 0.05 * (ub_bounds - lb_bounds) # Small step
                
                if r_def_strat < 0.33: # Random walk
                    perturbation = (np.random.rand(D_dim) - 0.5) * 2 * step_size_factor 
                    log_print(f"      Strategy: Random walk, perturbation = {np.round(perturbation,3)}",4)
                elif r_def_strat < 0.66: # Move towards dominant (safety)
                    direction_to_dominant = D_hippo_pos - positions[i_hippo_idx]
                    perturbation = random.random() * step_size_factor * (direction_to_dominant / (np.linalg.norm(direction_to_dominant) + 1e-6))
                    log_print(f"      Strategy: Move towards dominant, perturbation = {np.round(perturbation,3)}",4)
                else: # Move away from predator (conceptual)
                    direction_from_predator = positions[i_hippo_idx] - predator_pos
                    perturbation = random.random() * step_size_factor * (direction_from_predator / (np.linalg.norm(direction_from_predator) + 1e-6))
                    log_print(f"      Strategy: Move away from predator, perturbation = {np.round(perturbation,3)}",4)

                new_pos_p2 = positions[i_hippo_idx] + perturbation
                new_pos_p2_bounded = apply_bounds(new_pos_p2, lb_bounds, ub_bounds)
                fit_p2 = obj_func(new_pos_p2_bounded)
                log_print(f"      Cand. Pos (P2) before bounds = {np.round(new_pos_p2,3)}", 3)
                log_print(f"      Cand. Pos (P2) after bounds  = {np.round(new_pos_p2_bounded,3)}, Fitness = {fit_p2:.3f}", 3)

                if fit_p2 > fitness[i_hippo_idx]:
                    positions[i_hippo_idx] = new_pos_p2_bounded
                    fitness[i_hippo_idx] = fit_p2
                    log_print(f"      Hippo {i_hippo_idx+1} updated in Phase 2.", 3)
            
            if np.array_equal(positions[i_hippo_idx], original_pos_for_hippo_iter) and fitness[i_hippo_idx] == original_fit_for_hippo_iter:
                log_print(f"    Hippo {i_hippo_idx+1} position unchanged after Phase 1/2.", 2)


        # --- Phase 3: Escaping from Predator (Development) --- (MHO Eq. 19, 20, 21)
        # Applied to all hippos after Phase 1 & 2 are done for all for this iteration t
        log_print(f"\n    Phase 3: Escaping/Development for ALL Hippos", 1)
        for i_hippo_idx in range(N_pop):
            log_print(f"      Processing Hippo {i_hippo_idx+1} for Phase 3:", 2)
            log_print(f"        Current Pos = {np.round(positions[i_hippo_idx], 3).tolist()}, Fitness = {fitness[i_hippo_idx]:.3f}", 3)
            
            # MHO Eq. 20: Local bounds shrink with iteration number
            # Using current_iter_num (1-based) for 't' in Eq. 20
            lb_local = lb_bounds / current_iter_num 
            ub_local = ub_bounds / current_iter_num
            log_print(f"        lb_local (lb/{current_iter_num}) = {np.round(lb_local,3)}", 3)
            log_print(f"        ub_local (ub/{current_iter_num}) = {np.round(ub_local,3)}", 3)

            r10_esc = random.random() # scalar
            
            # s1_factor from MHO Eq. 21 (Randomly select one variant)
            s_choice = random.choice(['v1', 'v2', 'v3'])
            if s_choice == 'v1': # 2 * r_bar_11 - 1
                r_bar_11_esc = np.random.rand(D_dim) # vector
                s1_factor = 2 * r_bar_11_esc - 1
                log_print(f"        s1_factor (Eq.21 v1: 2*r_bar_11-1): r_bar_11={np.round(r_bar_11_esc,3)} => {np.round(s1_factor,3)}", 3)
            elif s_choice == 'v2': # r12 (normally distributed)
                s1_factor = np.random.randn(D_dim) # vector
                log_print(f"        s1_factor (Eq.21 v2: r12 normal dist) => {np.round(s1_factor,3)}", 3)
            else: # r13 (random [0,1])
                s1_factor = np.random.rand(D_dim) # vector
                log_print(f"        s1_factor (Eq.21 v3: r13 random [0,1]) => {np.round(s1_factor,3)}", 3)
            log_print(f"        r10_esc = {r10_esc:.3f}", 3)

            # MHO Eq. 19
            term_in_parenthesis = lb_local + s1_factor * (ub_local - lb_local)
            delta_X_P3 = r10_esc * term_in_parenthesis
            new_pos_p3 = positions[i_hippo_idx] + delta_X_P3
            log_print(f"          lb_local + s1*(ub_local-lb_local) = {np.round(term_in_parenthesis,3)}",4)
            log_print(f"          delta_X_P3 = r10 * prev_vector = {np.round(delta_X_P3,3)}",4)

            new_pos_p3_bounded = apply_bounds(new_pos_p3, lb_bounds, ub_bounds)
            fit_p3 = obj_func(new_pos_p3_bounded)
            log_print(f"        Cand. Pos (P3) before bounds = {np.round(new_pos_p3,3)}", 3)
            log_print(f"        Cand. Pos (P3) after bounds  = {np.round(new_pos_p3_bounded,3)}, Fitness = {fit_p3:.3f}", 3)

            # MHO Eq. 22 update rule
            if fit_p3 > fitness[i_hippo_idx]:
                positions[i_hippo_idx] = new_pos_p3_bounded
                fitness[i_hippo_idx] = fit_p3
                log_print(f"        Hippo {i_hippo_idx+1} updated in Phase 3.", 3)
            else:
                log_print(f"        Hippo {i_hippo_idx+1} not updated in Phase 3.",3)

        # --- Small-Hole Imaging Reverse Learning Strategy (MHO Eq. 18) ---
        # Applied to the current global best solution (D_hippo_pos)
        log_print(f"\n    Small-Hole Imaging Reverse Learning for current Dominant Hippo:", 1)
        log_print(f"      Current D_hippo_pos = {np.round(D_hippo_pos, 3).tolist()}, Fitness = {D_hippo_fitness:.3f}", 2)
        
        # MHO Eq. 18: X'_best,j = (lb_j + ub_j) - X_best,j
        # lb_bounds and ub_bounds are the original problem bounds
        reversed_D_hippo_pos = (lb_bounds + ub_bounds) - D_hippo_pos
        reversed_D_hippo_pos_bounded = apply_bounds(reversed_D_hippo_pos, lb_bounds, ub_bounds)
        reversed_D_hippo_fitness = obj_func(reversed_D_hippo_pos_bounded)
        log_print(f"      Reversed Cand. Pos before bounds = {np.round(reversed_D_hippo_pos,3)}",2)
        log_print(f"      Reversed Cand. Pos after bounds  = {np.round(reversed_D_hippo_pos_bounded,3)}, Fitness = {reversed_D_hippo_fitness:.3f}", 2)

        if reversed_D_hippo_fitness > D_hippo_fitness:
            D_hippo_pos = reversed_D_hippo_pos_bounded.copy()
            D_hippo_fitness = reversed_D_hippo_fitness
            log_print(f"      Dominant Hippo updated by Small-Hole Imaging! New D_hippo Fitness = {D_hippo_fitness:.3f}", 2)
            
            # Strategy: Replace worst hippo in population with this new global best if it's better
            worst_idx_pop = np.argmin(fitness)
            if D_hippo_fitness > fitness[worst_idx_pop]:
                 positions[worst_idx_pop] = D_hippo_pos.copy()
                 fitness[worst_idx_pop] = D_hippo_fitness
                 log_print(f"      Worst hippo in population (Hippo {worst_idx_pop+1}, fitness {fitness[worst_idx_pop]:.3f}) replaced by new D_hippo.", 2)
        else:
            log_print(f"      Reversed solution not better. Dominant Hippo remains unchanged by SHR.",2)

        # Update the overall best solution (D_hippo) by checking current population again
        # (in case a regular hippo update in Phase 3 became better than SHR-updated D_hippo)
        current_pop_best_idx = np.argmax(fitness)
        if fitness[current_pop_best_idx] > D_hippo_fitness:
            D_hippo_pos = positions[current_pop_best_idx].copy()
            D_hippo_fitness = fitness[current_pop_best_idx]
            log_print(f"    New Dominant Hippo found from population (Hippo {current_pop_best_idx+1}) after all phases.", 1)
        
        convergence_curve[t_iter_idx] = D_hippo_fitness
        log_print(f"  End of Iteration {current_iter_num}: Best Fitness So Far = {D_hippo_fitness:.3f}", 1)
        log_print(f"  Best Position So Far = {np.round(D_hippo_pos, 3).tolist()}", 1)
        log_print(f"  Current Population status:",1)
        for i_pop in range(N_pop):
            log_print(f"    Hippo {i_pop+1}: Pos = {np.round(positions[i_pop],3)}, Fit = {fitness[i_pop]:.3f}",2)
        log_print("",0) # Newline for readability

    log_print("\n--- MHO Algorithm Finished ---", 0)
    log_print(f"Final Best Position: {np.round(D_hippo_pos, 3).tolist()}", 1)
    log_print(f"Final Best Fitness (Profit): {D_hippo_fitness:.3f}", 1)
    log_print("\nConvergence Curve (Best fitness per iteration):", 0)
    for i in range(len(convergence_curve)):
        log_print(f"Iter {i+1}: {convergence_curve[i]:.3f}", 1)

    return D_hippo_fitness, D_hippo_pos, convergence_curve

# --- Run MHO Detailed ---
best_fitness, best_position, curve = MHO_detailed(N_pop, Max_iter, lb_bounds, ub_bounds, D_dimensions, objective_function)

# --- Generate Text File Output ---
file_name = "mho_detailed_run_output.txt"
with open(file_name, "w") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"\nDetailed sample run output saved to {file_name}")