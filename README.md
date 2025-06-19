# Modified hippopotamus optimization (mho) algorithm

## overview

this repository contains the implementation of the modified hippopotamus optimization (mho) algorithm, a metaheuristic for constrained optimization. mho is an enhanced version of the original hippopotamus optimization (ho) algorithm, with modifications for improved convergence and solution quality.

## 🎯 Research Objective

The primary goal of this research is to enhance the performance of the original Hippopotamus Optimization algorithm through strategic modifications, particularly focusing on:

1. **Chaotic Initialization**: Using sine chaotic maps for improved population diversity
2. **Enhanced Convergence**: Modified convergence factors for better exploration-exploitation balance
3. **Reverse Learning**: Small-hole imaging reverse learning mechanism for escaping local optima
4. **Real-world Application**: Applied to a sample land allocation and scheduling problem

## 🚀 Key Features

### MHO Modifications

1. **Sine Chaotic Map Initialization** (MHO Eq. 1, 4, 5)
   - Uses `x_{k+1} = α * sin(π * x_k)` for chaotic sequence generation
   - Improves initial population diversity and distribution

2. **Modified Convergence Factor** (MHO Eq. 13)
   - `T_mho_conv = 1 - (t/T_max)^6`
   - Enhanced convergence control compared to original HO

3. **Small-hole Imaging Reverse Learning** (MHO Eq. 18)
   - `X_rev = (lb + ub) - X_current`
   - Helps escape local optima and explore solution space more effectively

### Problem Domain

The algorithm is tested on **crop allocation optimization** problems with the following characteristics:
- **Objective**: Maximize profit from crop harvesting
- **Constraints**: Land allocation limits, double cropping restrictions
- **Variables**: Harvest amounts per crop per month
- **Dimensions**: Configurable (default: 2 crops × 3 months = 6 variables)

## 📁 Repository Structure

```
├── mho.cpp                    # Main C++ implementation (503 lines)
├── mho.py                     # Python implementation with detailed logging
├── a.cpp                      # Additional implementation variant
├── ho.pdf                     # Original HO algorithm paper
├── mancul files/
│   └── obj func.pdf          # Objective function documentation
└── README.md                 # This file
```

## problem formulation

maximize profit: sum x[i][j] * (g[i][j] - c[i][j])
subject to:
- 0 ≤ x[i][j] ≤ mha[i]
- sum over growth period ≤ tla[i]
- sum over months ≤ 2 * tla[i]

## usage

c++:
```
g++ -std=c++11 -O2 mho.cpp -o mho
./mho
```
python:
```
python mho.py
```

## 📈 Performance Analysis

The algorithm includes comprehensive logging and output generation:
- Population state tracking
- Fitness evolution
- Convergence analysis
- Detailed execution traces


## 🔬 Research Contributions

1. **Novel Initialization Strategy**: Chaotic map-based population initialization
2. **Enhanced Convergence**: Modified convergence factor for better performance
3. **Reverse Learning Mechanism**: Small-hole imaging for local optima escape
4. **Comprehensive Analysis**: Detailed implementation and performance evaluation

## 📚 References

- Original Hippopotamus Optimization (HO) algorithm
- Modified Hippopotamus Optimization (MHO) algorithm

## 👨‍🎓 Thesis Information

This work is part of a thesis research project focusing on metaheuristic optimization algorithms and their applications to agricultural planning problems.

## 📄 License

This project is part of academic research. Please cite appropriately if used in your work.

## 🤝 Contributing

This is a thesis research project. For academic collaboration or questions, please contact the author.

---

*For detailed algorithm analysis and implementation specifics, refer to the source code and output files provided in this repository.* 
