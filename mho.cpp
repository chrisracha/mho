#include <cmath>
#include <vector>
#include <iostream>
#include <stdlib.h> // for rand(), srand() - consider replacing with <random>
#include <iomanip>  // for std::setw, std::setprecision
#include <algorithm> // for std::min_element, std::max, std::min, std::shuffle
#include <limits>    // for std::numeric_limits
#include <numeric>   // for std::iota, std::accumulate
#include <random>    // for std::mt19937, std::uniform_real_distribution, std::normal_distribution
#include <ctime>     // for time()
#include <functional> // for std::function

using namespace std;

#define M_PI 3.14159265358979323846

// --- Random Number Generation Setup ---
std::mt19937 rng(static_cast<unsigned int>(time(0))); // Mersenne Twister RNG seeded with time

// Helper to get a random double between min and max
double rand_double(double min_val, double max_val) {
    if (min_val > max_val) std::swap(min_val, max_val);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    return dist(rng);
}

// Helper to get a random integer between min and max (inclusive)
int rand_int(int min_val, int max_val) {
     if (min_val > max_val) std::swap(min_val, max_val);
     if (min_val == max_val) return min_val;
    std::uniform_int_distribution<int> dist(min_val, max_val); // Use uniform_int_distribution
    return dist(rng);
}

// Helper for standard normal distribution (mean 0, stddev 1)
double rand_normal() {
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}
// --- End Random Number Generation Setup ---


// --- Problem Definition (Example: Crop Allocation) ---
int crops = 2;                      // i
int months = 3;                     // j
int t_growth = 3;                   // growth period in months
vector<double> TLA = {300, 200};    // total land allocated per crop i
vector<double> MHA = {120,  80};    // max harvest per month per crop i
vector<double> ub_vec;              // per-variable upper bound, size n*m
vector<double> lb_vec;              // per-variable lower bound, size n*m
int dim;                            // Problem dimension


// gross returns and costs
vector<vector<double>> G = {
    {10, 12, 11},
    { 8,  9, 10}
};
vector<vector<double>> C = {
    {3, 4, 5},
    {2, 2, 3}
};

// --- End Problem Definition ---

// --- Helper Functions ---
// 2d xij to 1d array index
inline int idx(int i, int j) {
    return i*months + j;
}

// MHO Modification 1: Sine chaotic map initialization
double sine_map(double x, double alpha) {
    // Keep seed in [0, 1] for consistent chaotic behavior
    if (x <= 0.0 || x >= 1.0) {
        x = rand_double(0.01, 0.99); // Re-seed if it goes out of bounds
    }
    return alpha * sin(M_PI * x); // MHO paper Eq 1 uses alpha*sin(pi*x)
    // Note: Some sources use sin(alpha * pi * x). Using the paper's formula.
}

// Objective function (Minimization)
double objective(const vector<double>& X) {
    double profit = 0.0;
    if (X.empty() || X.size() != dim) {
        cerr << "Error: Invalid dimension in objective function. Expected " << dim << ", Got " << X.size() << endl;
        return std::numeric_limits<double>::max();
     }
    for (int i = 0; i < crops; ++i) {
        for (int j = 0; j < months; ++j) {
            profit += X[idx(i,j)] * (G[i][j] - C[i][j]);
        }
    }
    return -profit; // Minimize negative profit
}

// --- Constraints ---
void apply_constraints(vector<double>& X) {
     if (X.empty() || lb_vec.empty() || ub_vec.empty() || X.size() != dim) {
         cerr << "Error: Invalid input to apply_constraints." << endl;
         return;
     }
     // Apply bounds first
     for (int k = 0; k < dim; ++k) {
         X[k] = max(lb_vec[k], min(X[k], ub_vec[k]));
     }

     // Apply problem-specific constraints (Land allocation, Double cropping)
     // These might need adjustment based on the actual optimization problem
     // Land allocation
     for (int i = 0; i < crops; ++i) {
         for (int m0 = 0; m0 < months; ++m0) {
             double sum = 0;
             vector<int> indices_to_scale; // Store indices involved in this constraint check
             for (int dt = 0; dt < t_growth; ++dt) {
                 int mj = (m0 - dt + months) % months;
                 int k = idx(i, mj);
                 sum += X[k];
                 indices_to_scale.push_back(k);
             }
             if (sum > TLA[i] + 1e-9) {
                 if (abs(sum) < 1e-9) continue; // Avoid division by zero
                 double scale = TLA[i] / sum;
                 for (int k : indices_to_scale) {
                     X[k] *= scale;
                 }
             }
         }
     }

     // Double cropping
     for (int i = 0; i < crops; ++i) {
         double total = 0;
          vector<int> indices_to_scale;
         for (int j = 0; j < months; ++j) {
             int k = idx(i,j);
             total += X[k];
             indices_to_scale.push_back(k);
         }
         double cap = 2.0 * TLA[i];
         if (total > cap + 1e-9) {
              if (abs(total) < 1e-9) continue;
             double scale = cap / total;
             for (int k : indices_to_scale) {
                 X[k] *= scale;
             }
         }
     }

     // Re-apply bounds after constraint adjustments
     for (int k = 0; k < dim; ++k) {
         X[k] = max(lb_vec[k], min(X[k], ub_vec[k]));
     }
}
// --- End Constraints ---


// --- Utility Functions ---
// Print population state
void print_population(const string& algo_name, int t_iter, const vector<vector<double>>& pop, const vector<double>& fit, int leader_idx) {
     int N = pop.size();
     if (N == 0 || pop[0].empty()) {
         cout << "\n--- " << algo_name << " Population State (Iter " << t_iter << ") --- EMPTY\n";
         return;
     }
     int current_dim = pop[0].size();
     cout << "\n--- " << algo_name << " Population State (Iter " << t_iter << ") --- \n";
     cout << "Hippo ";
     for (int j = 0; j < current_dim; ++j) cout << setw(10) << "Var" << j+1;
     cout << setw(15) << "Fitness";
     cout << "\n" << string(6 + 10*current_dim + 15, '-') << "\n";
     for (int h = 0; h < N; ++h) {
         cout << setw(5) << h << " |";
         if (h >= pop.size() || pop[h].size() != current_dim) { cout << " Error size/dim\n"; continue; }
         for (double x : pop[h]) cout << setw(10) << fixed << setprecision(3) << x;
         if (h >= fit.size()) { cout << " Error fit size\n"; continue; }
         cout << setw(15) << fixed << setprecision(5) << fit[h];
         if (h == leader_idx) cout << " (Leader)";
         cout << "\n";
     }
      if (leader_idx >= 0 && leader_idx < N && leader_idx < fit.size()) {
        cout << "Best Fitness (" << algo_name << "): " << fit[leader_idx] << endl;
     } else {
         cout << "Best Fitness (" << algo_name << "): Invalid leader index." << endl;
     }
}

// Levy flight vector (Used in original HO Phase 2)
vector<double> levy_vector(int dimension) {
    double beta = 1.5;
    vector<double> step(dimension);
    double sigma_num = tgamma(1.0 + beta) * sin(M_PI * beta / 2.0);
    double sigma_den = tgamma((1.0 + beta) / 2.0) * beta * pow(2.0, (beta - 1.0) / 2.0);
    if (abs(sigma_den) < 1e-10 || std::isnan(sigma_num) || std::isnan(sigma_den) || std::isinf(sigma_num) || std::isinf(sigma_den)) {
         std::fill(step.begin(), step.end(), 0.0); // Return zero vector if calculation fails
         return step;
     }
    double sigma = pow(sigma_num / sigma_den, 1.0 / beta);
    std::normal_distribution<double> dist_u(0.0, sigma);
    std::normal_distribution<double> dist_v(0.0, 1.0);
    for (int k = 0; k < dimension; ++k) {
        double u = dist_u(rng);
        double v = dist_v(rng);
        double v_abs_pow = pow(abs(v), 1.0 / beta);
        if (abs(v_abs_pow) < 1e-10) {
            step[k] = u / ((v >= 0 ? 1.0 : -1.0) * 1e-10);
        } else {
            step[k] = u / v_abs_pow;
        }
        step[k] *= 0.01; // Apply scaling factor
    }
    return step;
}
// --- End Utility Functions ---


int main() {
    // --- Algorithm Parameters ---
    int N      = 10;            // Population size
    int T_max  = 500;           // Max iterations
    double alpha_chaos = 0.99;  // Chaos coeff for MHO initialization
    // --- End Algorithm Parameters ---

    // --- Initialization ---
    dim = crops * months;
    if (dim <= 0) { cerr << "Error: Problem dimension must be positive." << endl; return 1; }

    lb_vec.resize(dim, 0.0);
    ub_vec.resize(dim);
    for (int i = 0; i < crops; ++i) {
        for (int j = 0; j < months; ++j) ub_vec[idx(i,j)] = MHA[i];
    }

    vector<vector<double>> pop(N, vector<double>(dim));
    vector<double> fit(N);

    // MHO Modification 1: Sine Chaotic Map Initialization
    cout << "Initializing MHO population using Sine Chaotic Map...\n";
    double chaos_seed = rand_double(0.01, 0.99);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < dim; ++k) {
            chaos_seed = sine_map(chaos_seed, alpha_chaos);
            // Ensure chaos_seed stays within [0, 1] for the formula lb + seed * (ub - lb)
            if (chaos_seed < 0.0 || chaos_seed > 1.0 || std::isnan(chaos_seed) || std::isinf(chaos_seed)) {
                chaos_seed = rand_double(0.01, 0.99); // Reset if it escapes [0, 1]
            }
            pop[i][k] = lb_vec[k] + chaos_seed * (ub_vec[k] - lb_vec[k]);
        }
        apply_constraints(pop[i]);
        fit[i] = objective(pop[i]);
    }

    int leader_idx = min_element(fit.begin(), fit.end()) - fit.begin();
    double best_fit_global = fit[leader_idx];
    vector<double> best_sol_global = pop[leader_idx];

    cout << "\nInitial State:";
    print_population("MHO", 0, pop, fit, leader_idx);
    cout << "-----------------------------------------\n";
    // --- End Initialization ---

    // --- Main Optimization Loop ---
    for (int t_iter = 1; t_iter <= T_max; ++t_iter) {
        cout << "\n<<<<<<<< Iteration " << t_iter << " >>>>>>>>\n";

        vector<vector<double>> pop_next = pop; // Work on a copy for updates
        vector<double> fit_next = fit;

        // --- Base HO Phases (with MHO Modifications) ---

        // MHO Modification 2: Changed Convergence Factor
        double T_mho_conv = 1.0 - pow((double(t_iter) / T_max), 6.0); // MHO Eq 13

        // HO Phase 1: Position Update (Modified with T_mho_conv)
        cout << " MHO Phase 1 (Position Update)...\n";
        vector<vector<double>> X_P1(N, vector<double>(dim)); // Potential "Male" update
        vector<vector<double>> X_P2(N, vector<double>(dim)); // Potential "Female/Immature" update

        for (int i = 0; i < N; ++i) {
            vector<double> X_current = pop[i];
            int I1 = rand_int(1, 2);
            int I2 = rand_int(1, 2);

            // Calculate MG (Mean of Random Group)
            vector<double> MG(dim, 0.0);
             if (N > 1) {
                 int rand_group_size = rand_int(1, max(1, N - 1)); // Ensure size is at least 1 if N > 1
                 vector<int> indices(N);
                 iota(indices.begin(), indices.end(), 0);
                 std::shuffle(indices.begin(), indices.end(), rng);
                 int count = 0;
                 for (int j = 0; j < N && count < rand_group_size; ++j) {
                     if (indices[j] == i) continue;
                     for (int k = 0; k < dim; ++k) MG[k] += pop[indices[j]][k];
                     count++;
                 }
                 if (count > 0) for (int k = 0; k < dim; ++k) MG[k] /= count;
                 else MG = X_current;
             } else MG = X_current;


            // Calculate random factors A and B for Female/Immature update
             vector<double> factor_A(dim), factor_B(dim);
             int scenario_A = rand_int(0, 4); int scenario_B = rand_int(0, 4);
             vector<double> r_vec(dim); for(int k=0; k<dim; ++k) r_vec[k] = rand_double(0,1); // Temp random vector
             // Function to calculate factor based on scenario
             auto calculate_factor = [&](int scenario, int I_val) -> vector<double> {
                 vector<double> factor(dim);
                 if (scenario == 0) for(int k=0; k<dim; ++k) factor[k] = I_val * r_vec[k] + double(!rand_int(0,1));
                 else if (scenario == 1) for(int k=0; k<dim; ++k) factor[k] = 2.0 * r_vec[k] - 1.0;
                 else if (scenario == 2) for(int k=0; k<dim; ++k) factor[k] = r_vec[k];
                 else if (scenario == 3) for(int k=0; k<dim; ++k) factor[k] = I_val * r_vec[k] + double(!rand_int(0,1));
                 else { double r_scalar = rand_double(0,1); fill(factor.begin(), factor.end(), r_scalar); }
                 return factor;
             };
            factor_A = calculate_factor(scenario_A, I2);
            factor_B = calculate_factor(scenario_B, I1); // Note: HO.m seems to use same factors for P1/P2, but paper might imply separate


            // Calculate potential update X_P1 ("Male"-like)
            double r_scalar_p1 = rand_double(0,1);
            for (int k = 0; k < dim; ++k) {
                 X_P1[i][k] = X_current[k] + r_scalar_p1 * (pop[leader_idx][k] - I1 * X_current[k]);
            }

            // Calculate potential update X_P2 ("Female/Immature"-like, uses MHO convergence logic)
            // MHO Modification 2: Use T_mho_conv > 0.95 condition
            if (T_mho_conv > 0.95) { // MHO Eq 14 case 1
                 for (int k = 0; k < dim; ++k) {
                     X_P2[i][k] = X_current[k] + factor_A[k] * (pop[leader_idx][k] - I2 * MG[k]);
                 }
            } else { // MHO Eq 15 cases
                 double r6 = rand_double(0, 1);
                 if (r6 > 0.5) { // MHO Eq 15 case 1
                      for (int k = 0; k < dim; ++k) {
                          X_P2[i][k] = X_current[k] + factor_B[k] * (MG[k] - pop[leader_idx][k]);
                      }
                 } else { // MHO Eq 15 case 2: Random exploration
                     for (int k = 0; k < dim; ++k) {
                         X_P2[i][k] = lb_vec[k] + rand_double(0, 1) * (ub_vec[k] - lb_vec[k]);
                     }
                 }
            }

            // Apply constraints and evaluate fitness for potential updates
            apply_constraints(X_P1[i]);
            apply_constraints(X_P2[i]);
            double f_p1 = objective(X_P1[i]);
            double f_p2 = objective(X_P2[i]);

            // Greedy Selection for Phase 1: Update pop_next if P1 or P2 is better
            if (f_p1 < fit_next[i]) {
                pop_next[i] = X_P1[i];
                fit_next[i] = f_p1;
            }
            if (f_p2 < fit_next[i]) { // Check P2 independently
                pop_next[i] = X_P2[i];
                fit_next[i] = f_p2;
            }
        }
        // Update pop/fit before next phase if needed (or update at the end)
        // Let's update at the end of all phases + reverse learning

        // HO Phase 2: Defense against predators
        cout << " MHO Phase 2 (Defense)...\n";
        vector<double> predator_pos(dim);
        for (int k = 0; k < dim; ++k) predator_pos[k] = lb_vec[k] + rand_double(0, 1) * (ub_vec[k] - lb_vec[k]);
        double F_predator = objective(predator_pos);

        for (int i = 0; i < N; ++i) { // Apply to all agents? HO.m applied to second half. MHO paper doesn't specify split. Let's apply to all.
            vector<double> X_current = pop_next[i]; // Use potentially updated pop_next
            vector<double> X_defend(dim);
            vector<double> D_vec(dim);
            for (int k = 0; k < dim; ++k) D_vec[k] = abs(predator_pos[k] - X_current[k]);

            vector<double> RL_vec = levy_vector(dim);
            double b_factor = rand_double(2, 4);
            double c_factor = rand_double(1, 1.5);
            double d_factor = rand_double(2, 3);
            double l_angle = rand_double(-2.0*M_PI, 2.0*M_PI);
            double cos_l = cos(l_angle);
            double denom_factor = c_factor - d_factor * cos_l;
            if (abs(denom_factor) < 1e-9) denom_factor = (denom_factor >= 0 ? 1.0 : -1.0) * 1e-9;

            // Using fit_next[i] as the comparison fitness for the *current* state after Phase 1 updates
            if (fit_next[i] > F_predator) { // Predator is 'better'
                for (int k = 0; k < dim; ++k) {
                    double dist_k = max(abs(D_vec[k]), 1e-9); // Avoid division by zero
                    X_defend[k] = RL_vec[k] * predator_pos[k] + (b_factor / denom_factor) * (1.0 / dist_k);
                }
            } else { // Hippo is 'better'
                 for (int k = 0; k < dim; ++k) {
                    double rand_val = rand_double(0,1);
                    double denom_term = max(abs(2.0 * D_vec[k] + rand_val), 1e-9); // Avoid division by zero
                    X_defend[k] = RL_vec[k] * predator_pos[k] + (b_factor / denom_factor) * (1.0 / denom_term);
                 }
            }

            apply_constraints(X_defend);
            double f_defend = objective(X_defend);
            if (f_defend < fit_next[i]) {
                pop_next[i] = X_defend;
                fit_next[i] = f_defend;
            }
        }

        // HO Phase 3: Escaping from the Predator (Exploitation)
        cout << " MHO Phase 3 (Escape)...\n";
        double t_factor_ho = max(1.0, (double)t_iter);
        vector<double> lb_local(dim), ub_local(dim);
        for(int k=0; k<dim; ++k) {
            lb_local[k] = lb_vec[k] / t_factor_ho;
            ub_local[k] = ub_vec[k] / t_factor_ho;
        }

        for (int i = 0; i < N; ++i) {
            vector<double> X_current = pop_next[i]; // Use potentially updated pop_next
            vector<double> X_escape(dim);

            vector<double> factor_D(dim);
            int s_scenario = rand_int(0, 2);
            if (s_scenario == 0) for(int k=0; k<dim; ++k) factor_D[k] = 2.0 * rand_double(0, 1) - 1.0;
            else if (s_scenario == 1) { double r=rand_double(0,1); fill(factor_D.begin(), factor_D.end(), r); }
            else { double rn=rand_normal(); fill(factor_D.begin(), factor_D.end(), rn); }

            double r10 = rand_double(0, 1);

            for (int k = 0; k < dim; ++k) {
                X_escape[k] = X_current[k] + r10 * (lb_local[k] + factor_D[k] * (ub_local[k] - lb_local[k]));
            }

            apply_constraints(X_escape);
            double f_escape = objective(X_escape);
            if (f_escape < fit_next[i]) {
                pop_next[i] = X_escape;
                fit_next[i] = f_escape;
            }
        }


        // MHO Modification 3: Small-hole imaging reverse learning
        cout << " MHO Reverse Learning Step...\n";
        for (int i = 0; i < N; ++i) {
            vector<double> X_current_updated = pop_next[i]; // Position after HO phases 1-3
            vector<double> X_rev(dim);

            // Calculate reversed solution (MHO Eq. 18)
            for (int k = 0; k < dim; ++k) {
                X_rev[k] = (lb_vec[k] + ub_vec[k]) - X_current_updated[k];
            }
            apply_constraints(X_rev);
            double f_rev = objective(X_rev);

            // Update if reversed solution is better than the one from HO phases
            if (f_rev < fit_next[i]) {
                 pop_next[i] = X_rev;
                 fit_next[i] = f_rev;
                 // cout << "  Hippo " << i << " updated (Reversed). Fitness: " << f_rev << endl; // Verbose
            }
        }


        // Final update of population and fitness for this iteration
        pop = pop_next;
        fit = fit_next;

        // Find and update global best
        leader_idx = min_element(fit.begin(), fit.end()) - fit.begin();
        if (fit[leader_idx] < best_fit_global) {
            best_fit_global = fit[leader_idx];
            best_sol_global = pop[leader_idx];
             cout << " MHO New Global Best Found: " << best_fit_global << endl;
        }
        print_population("MHO", t_iter, pop, fit, leader_idx);
        // --- End MHO Update Cycle ---


        // --- Iteration Summary ---
        if (t_iter % 50 == 0 || t_iter == T_max) {
            cout << "\n-----------------------------------------";
            cout << "\nIteration " << t_iter << " Summary:";
            cout << "\n MHO Best Fitness: " << fixed << setprecision(8) << best_fit_global;
            cout << "\n-----------------------------------------\n";
        }
        // --- End Iteration Summary ---

    }
    // --- End Main Optimization Loop ---

    // --- Final Output ---
    cout << "\n=========================================";
    cout << "\nOptimization Finished!";
    cout << "\n=========================================\n";

    leader_idx = min_element(fit.begin(), fit.end()) - fit.begin(); // Ensure leader is current
    print_population("MHO Final", T_max, pop, fit, leader_idx);
    cout << "MHO Optimal Fitness: " << fixed << setprecision(8) << best_fit_global << "\n";
    cout << "MHO Optimal Solution (Variables): ";
    for(double val : best_sol_global) cout << fixed << setprecision(5) << val << " ";
    cout << endl;

    return 0;
}