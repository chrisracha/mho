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
vector<vector<double>> ub_vec;      // per-variable upper bound, size [crops][months]
vector<vector<double>> lb_vec;      // per-variable lower bound, size [crops][months]
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
// idx function removed

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
double objective(const vector<vector<double>>& X) {
    double profit = 0.0;
    if (X.empty() || X.size() != crops || X[0].size() != months) {
        cerr << "Error: Invalid dimension in objective function. Expected [" << crops << "," << months << "], Got [" << X.size() << "," << (X.empty() ? 0 : X[0].size()) << "]" << endl;
        return std::numeric_limits<double>::max();
    }
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            profit += X[i][j] * (G[i][j] - C[i][j]);
        }
    }
    return -profit; // Minimize negative profit
}

// --- Constraints ---
void apply_constraints(vector<vector<double>>& X) {
    if (X.empty() || lb_vec.empty() || ub_vec.empty() || X.size() != crops || X[0].size() != months) {
        cerr << "Error: Invalid input to apply_constraints." << endl;
        return;
    }
    // Apply bounds first
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            X[i][j] = max(lb_vec[i][j], min(X[i][j], ub_vec[i][j]));
        }
    }
    // Land allocation
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t m0 = 0; m0 < X[i].size(); ++m0) {
            double sum = 0;
            vector<pair<size_t, size_t>> indices_to_scale;
            for (int dt = 0; dt < t_growth; ++dt) {
                size_t mj = (m0 - dt + months) % months;
                sum += X[i][mj];
                indices_to_scale.push_back({i, mj});
            }
            if (sum > TLA[i] + 1e-9) {
                if (abs(sum) < 1e-9) continue;
                double scale = TLA[i] / sum;
                for (auto& idx : indices_to_scale) {
                    X[idx.first][idx.second] *= scale;
                }
            }
        }
    }
    // Double cropping
    for (size_t i = 0; i < X.size(); ++i) {
        double total = 0;
        vector<pair<size_t, size_t>> indices_to_scale;
        for (size_t j = 0; j < X[i].size(); ++j) {
            total += X[i][j];
            indices_to_scale.push_back({i, j});
        }
        double cap = 2.0 * TLA[i];
        if (total > cap + 1e-9) {
            if (abs(total) < 1e-9) continue;
            double scale = cap / total;
            for (auto& idx : indices_to_scale) {
                X[idx.first][idx.second] *= scale;
            }
        }
    }
    // Re-apply bounds after constraint adjustments
    for (size_t i = 0; i < X.size(); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            X[i][j] = max(lb_vec[i][j], min(X[i][j], ub_vec[i][j]));
        }
    }
}
// --- End Constraints ---


// --- Utility Functions ---
// Print population state
void print_population(const string& algo_name, int t_iter, const vector<vector<vector<double>>>& hippo, const vector<double>& fit, int leader_idx) {
    int N = hippo.size();
    if (N == 0 || hippo[0].empty() || hippo[0][0].empty()) {
        cout << "\n--- " << algo_name << " Population State (Iter " << t_iter << ") --- EMPTY\n";
        return;
    }
    cout << "\n--- " << algo_name << " Population State (Iter " << t_iter << ") --- \n";
    cout << "Hippo ";
    for (size_t i = 0; i < hippo[0].size(); ++i)
        for (size_t j = 0; j < hippo[0][i].size(); ++j)
            cout << setw(10) << "C" << i+1 << "M" << j+1;
    cout << setw(15) << "Fitness";
    cout << "\n" << string(6 + 10*crops*months + 15, '-') << "\n";
    for (size_t h = 0; h < hippo.size(); ++h) {
        cout << setw(5) << h << " |";
        for (const auto& crop : hippo[h])
            for (const auto& val : crop)
                cout << setw(10) << fixed << setprecision(3) << val;
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

    lb_vec.resize(crops, vector<double>(months, 0.0));
    ub_vec.resize(crops, vector<double>(months));
    for (int i = 0; i < crops; ++i) {
        for (int j = 0; j < months; ++j) ub_vec[i][j] = MHA[i];
    }

    vector<vector<vector<double>>> hippo(N, vector<vector<double>>(crops, vector<double>(months)));
    vector<double> fit(N);

    // MHO Modification 1: Sine Chaotic Map Initialization
    cout << "Initializing MHO population using Sine Chaotic Map...\n";
    double chaos_seed = rand_double(0.01, 0.99);
    for (int h = 0; h < N; ++h) {
        for (int i = 0; i < crops; ++i) {
            for (int j = 0; j < months; ++j) {
                chaos_seed = sine_map(chaos_seed, alpha_chaos);
                if (chaos_seed < 0.0 || chaos_seed > 1.0 || std::isnan(chaos_seed) || std::isinf(chaos_seed)) {
                    chaos_seed = rand_double(0.01, 0.99);
                }
                hippo[h][i][j] = lb_vec[i][j] + chaos_seed * (ub_vec[i][j] - lb_vec[i][j]);
            }
        }
        apply_constraints(hippo[h]);
        fit[h] = objective(hippo[h]);
    }

    int leader_idx = min_element(fit.begin(), fit.end()) - fit.begin();
    double best_fit_global = fit[leader_idx];
    vector<vector<double>> best_sol_global = hippo[leader_idx];

    cout << "\nInitial State:";
    print_population("MHO", 0, hippo, fit, leader_idx);
    cout << "-----------------------------------------\n";
    // --- End Initialization ---

    // --- Main Optimization Loop ---
    for (int t_iter = 1; t_iter <= T_max; ++t_iter) {
        cout << "\n<<<<<<<< Iteration " << t_iter << " >>>>>>>>>\n";

        vector<vector<vector<double>>> hippo_next = hippo;
        vector<double> fit_next = fit;

        // --- Base HO Phases (with MHO Modifications) ---
        double T_mho_conv = 1.0 - pow((double(t_iter) / T_max), 6.0);
        cout << " MHO Phase 1 (Position Update)...\n";
        vector<vector<vector<double>>> X_P1(N, vector<vector<double>>(crops, vector<double>(months)));
        vector<vector<vector<double>>> X_P2(N, vector<vector<double>>(crops, vector<double>(months)));

        for (int h = 0; h < N; ++h) {
            vector<vector<double>> X_current = hippo[h];
            int I1 = rand_int(1, 2);
            int I2 = rand_int(1, 2);
            vector<vector<double>> MG(crops, vector<double>(months, 0.0));
            if (N > 1) {
                int rand_group_size = rand_int(1, max(1, N - 1));
                vector<int> indices(N);
                iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), rng);
                int count = 0;
                for (int j = 0; j < N && count < rand_group_size; ++j) {
                    if (indices[j] == h) continue;
                    for (int i = 0; i < crops; ++i)
                        for (int m = 0; m < months; ++m)
                            MG[i][m] += hippo[indices[j]][i][m];
                    count++;
                }
                if (count > 0) for (int i = 0; i < crops; ++i) for (int m = 0; m < months; ++m) MG[i][m] /= count;
                else MG = X_current;
            } else MG = X_current;
            vector<vector<double>> factor_A(crops, vector<double>(months)), factor_B(crops, vector<double>(months));
            int scenario_A = rand_int(0, 4); int scenario_B = rand_int(0, 4);
            vector<vector<double>> r_vec(crops, vector<double>(months));
            for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) r_vec[i][j] = rand_double(0,1);
            auto calculate_factor = [&](int scenario, int I_val) -> vector<vector<double>> {
                vector<vector<double>> factor(crops, vector<double>(months));
                if (scenario == 0) for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) factor[i][j] = I_val * r_vec[i][j] + double(!rand_int(0,1));
                else if (scenario == 1) for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) factor[i][j] = 2.0 * r_vec[i][j] - 1.0;
                else if (scenario == 2) for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) factor[i][j] = r_vec[i][j];
                else if (scenario == 3) for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) factor[i][j] = I_val * r_vec[i][j] + double(!rand_int(0,1));
                else { double r_scalar = rand_double(0,1); for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) factor[i][j] = r_scalar; }
                return factor;
            };
            factor_A = calculate_factor(scenario_A, I2);
            factor_B = calculate_factor(scenario_B, I1);
            double r_scalar_p1 = rand_double(0,1);
            for (int i = 0; i < crops; ++i) {
                for (int j = 0; j < months; ++j) {
                    X_P1[h][i][j] = X_current[i][j] + r_scalar_p1 * (hippo[leader_idx][i][j] - I1 * X_current[i][j]);
                }
            }
            if (T_mho_conv > 0.95) {
                for (int i = 0; i < crops; ++i) {
                    for (int j = 0; j < months; ++j) {
                        X_P2[h][i][j] = X_current[i][j] + factor_A[i][j] * (hippo[leader_idx][i][j] - I2 * MG[i][j]);
                    }
                }
            } else {
                double r6 = rand_double(0, 1);
                if (r6 > 0.5) {
                    for (int i = 0; i < crops; ++i) {
                        for (int j = 0; j < months; ++j) {
                            X_P2[h][i][j] = X_current[i][j] + factor_B[i][j] * (MG[i][j] - hippo[leader_idx][i][j]);
                        }
                    }
                } else {
                    for (int i = 0; i < crops; ++i) {
                        for (int j = 0; j < months; ++j) {
                            X_P2[h][i][j] = lb_vec[i][j] + rand_double(0, 1) * (ub_vec[i][j] - lb_vec[i][j]);
                        }
                    }
                }
            }
            apply_constraints(X_P1[h]);
            apply_constraints(X_P2[h]);
            double f_p1 = objective(X_P1[h]);
            double f_p2 = objective(X_P2[h]);
            if (f_p1 < fit_next[h]) {
                hippo_next[h] = X_P1[h];
                fit_next[h] = f_p1;
            }
            if (f_p2 < fit_next[h]) {
                hippo_next[h] = X_P2[h];
                fit_next[h] = f_p2;
            }
        }
        cout << " MHO Phase 2 (Defense)...\n";
        vector<vector<double>> predator_pos(crops, vector<double>(months));
        for (int i = 0; i < crops; ++i) for (int j = 0; j < months; ++j) predator_pos[i][j] = lb_vec[i][j] + rand_double(0, 1) * (ub_vec[i][j] - lb_vec[i][j]);
        double F_predator = objective(predator_pos);
        for (int h = 0; h < N; ++h) {
            vector<vector<double>> X_current = hippo_next[h];
            vector<vector<double>> X_defend(crops, vector<double>(months));
            vector<vector<double>> D_vec(crops, vector<double>(months));
            for (int i = 0; i < crops; ++i) for (int j = 0; j < months; ++j) D_vec[i][j] = abs(predator_pos[i][j] - X_current[i][j]);
            vector<double> RL_vec = levy_vector(crops * months);
            double b_factor = rand_double(2, 4);
            double c_factor = rand_double(1, 1.5);
            double d_factor = rand_double(2, 3);
            double l_angle = rand_double(-2.0*M_PI, 2.0*M_PI);
            double cos_l = cos(l_angle);
            double denom_factor = c_factor - d_factor * cos_l;
            if (abs(denom_factor) < 1e-9) denom_factor = (denom_factor >= 0 ? 1.0 : -1.0) * 1e-9;
            if (fit_next[h] > F_predator) {
                int rl_idx = 0;
                for (int i = 0; i < crops; ++i) {
                    for (int j = 0; j < months; ++j, ++rl_idx) {
                        double dist_k = max(abs(D_vec[i][j]), 1e-9);
                        X_defend[i][j] = RL_vec[rl_idx] * predator_pos[i][j] + (b_factor / denom_factor) * (1.0 / dist_k);
                    }
                }
            } else {
                int rl_idx = 0;
                for (int i = 0; i < crops; ++i) {
                    for (int j = 0; j < months; ++j, ++rl_idx) {
                        double rand_val = rand_double(0,1);
                        double denom_term = max(abs(2.0 * D_vec[i][j] + rand_val), 1e-9);
                        X_defend[i][j] = RL_vec[rl_idx] * predator_pos[i][j] + (b_factor / denom_factor) * (1.0 / denom_term);
                    }
                }
            }
            apply_constraints(X_defend);
            double f_defend = objective(X_defend);
            if (f_defend < fit_next[h]) {
                hippo_next[h] = X_defend;
                fit_next[h] = f_defend;
            }
        }
        cout << " MHO Phase 3 (Escape)...\n";
        double t_factor_ho = max(1.0, (double)t_iter);
        vector<vector<double>> lb_local(crops, vector<double>(months)), ub_local(crops, vector<double>(months));
        for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) {
            lb_local[i][j] = lb_vec[i][j] / t_factor_ho;
            ub_local[i][j] = ub_vec[i][j] / t_factor_ho;
        }
        for (int h = 0; h < N; ++h) {
            vector<vector<double>> X_current = hippo_next[h];
            vector<vector<double>> X_escape(crops, vector<double>(months));
            vector<vector<double>> factor_D(crops, vector<double>(months));
            int s_scenario = rand_int(0, 2);
            if (s_scenario == 0) for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) factor_D[i][j] = 2.0 * rand_double(0, 1) - 1.0;
            else if (s_scenario == 1) { double r=rand_double(0,1); for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) factor_D[i][j] = r; }
            else { double rn=rand_normal(); for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) factor_D[i][j] = rn; }
            double r10 = rand_double(0, 1);
            for (int i = 0; i < crops; ++i) {
                for (int j = 0; j < months; ++j) {
                    X_escape[i][j] = X_current[i][j] + r10 * (lb_local[i][j] + factor_D[i][j] * (ub_local[i][j] - lb_local[i][j]));
                }
            }
            apply_constraints(X_escape);
            double f_escape = objective(X_escape);
            if (f_escape < fit_next[h]) {
                hippo_next[h] = X_escape;
                fit_next[h] = f_escape;
            }
        }
        cout << " MHO Reverse Learning Step...\n";
        for (int h = 0; h < N; ++h) {
            vector<vector<double>> X_current_updated = hippo_next[h];
            vector<vector<double>> X_rev(crops, vector<double>(months));
            for (int i = 0; i < crops; ++i) {
                for (int j = 0; j < months; ++j) {
                    X_rev[i][j] = (lb_vec[i][j] + ub_vec[i][j]) - X_current_updated[i][j];
                }
            }
            apply_constraints(X_rev);
            double f_rev = objective(X_rev);
            if (f_rev < fit_next[h]) {
                hippo_next[h] = X_rev;
                fit_next[h] = f_rev;
            }
        }
        hippo = hippo_next;
        fit = fit_next;
        leader_idx = min_element(fit.begin(), fit.end()) - fit.begin();
        if (fit[leader_idx] < best_fit_global) {
            best_fit_global = fit[leader_idx];
            best_sol_global = hippo[leader_idx];
            cout << " MHO New Global Best Found: " << best_fit_global << endl;
        }
        print_population("MHO", t_iter, hippo, fit, leader_idx);
        if (t_iter % 50 == 0 || t_iter == T_max) {
            cout << "\n-----------------------------------------";
            cout << "\nIteration " << t_iter << " Summary:";
            cout << "\n MHO Best Fitness: " << fixed << setprecision(8) << best_fit_global;
            cout << "\n-----------------------------------------\n";
        }
    }
    cout << "\n=========================================";
    cout << "\nOptimization Finished!";
    cout << "\n=========================================\n";
    leader_idx = min_element(fit.begin(), fit.end()) - fit.begin();
    print_population("MHO Final", T_max, hippo, fit, leader_idx);
    cout << "MHO Optimal Fitness: " << fixed << setprecision(8) << best_fit_global << "\n";
    cout << "MHO Optimal Solution (Variables): ";
    for(int i=0; i<crops; ++i) for(int j=0; j<months; ++j) cout << fixed << setprecision(5) << best_sol_global[i][j] << " ";
    cout << endl;
    return 0;
}