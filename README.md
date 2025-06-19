# Modified Hippopotamus Optimization (MHO) Algorithm

## Overview

This repository provides an implementation of the Modified Hippopotamus Optimization (MHO) algorithm, a metaheuristic for constrained optimization. The MHO algorithm is an enhanced version of the original Hippopotamus Optimization (HO) algorithm, with modifications for improved convergence and solution quality. **The modifications are based on the work by Han et al. (2025) [1].**

The optimization problem addressed in this repository—crop harvesting and land allocation—is adapted from Custodio et al. (2024) [2].

## Table of Contents

- [Research Objective](#research-objective)
- [Key Features](#key-features)
- [Problem Domain](#problem-domain)
- [Repository Structure](#repository-structure)
- [Problem Formulation](#problem-formulation)
- [Usage](#usage)
- [Performance Analysis](#performance-analysis)
- [Research Contributions](#research-contributions)
- [Citations](#citations)
- [License](#license)
- [Contributing](#contributing)

## Research Objective

The primary goal of this research is to enhance the performance of the original Hippopotamus Optimization algorithm through strategic modifications, particularly focusing on:

1. **Chaotic Initialization**: Using sine chaotic maps for improved population diversity.
2. **Enhanced Convergence**: Modified convergence factors for better exploration-exploitation balance.
3. **Reverse Learning**: Small-hole imaging reverse learning mechanism for escaping local optima.
4. **Real-world Application**: Application to a sample land allocation and scheduling problem.

## Key Features

### MHO Modifications (from [1])

1. **Sine Chaotic Map Initialization**
   - Uses \( x_{k+1} = \alpha \cdot \sin(\pi x_k) \) for chaotic sequence generation.
   - Improves initial population diversity and distribution.

2. **Modified Convergence Factor**
   - \( T_{\text{mho\_conv}} = 1 - (t/T_{\max})^6 \)
   - Enhanced convergence control compared to the original HO.

3. **Small-hole Imaging Reverse Learning**
   - \( X_{\text{rev}} = (lb + ub) - X_{\text{current}} \)
   - Helps escape local optima and explore the solution space more effectively.

## Problem Domain

The algorithm is tested on **crop allocation optimization** problems with the following characteristics (adapted from [2]):
- **Objective**: Maximize profit from crop harvesting.
- **Constraints**: Land allocation limits, double cropping restrictions.
- **Variables**: Harvest amounts per crop per month.
- **Dimensions**: Configurable (default: 2 crops × 3 months = 6 variables).

## Repository Structure

```
├── mho.cpp                    # Main C++ implementation (503 lines)
├── mho.py                     # Python implementation with detailed logging
├── a.cpp                      # Additional implementation variant
├── ho.pdf                     # Original HO algorithm paper
├── mancul files/
│   └── obj func.pdf           # Objective function documentation
└── README.md                  # This file
```

## Problem Formulation

**Maximize profit:**

    maximize:  sum x[i][j] * (g[i][j] - c[i][j])

**Subject to:**
- \( 0 \leq x[i][j] \leq mha[i] \)
- \( \sum_{\text{growth period}} x[i][j] \leq tla[i] \)
- \( \sum_{\text{months}} x[i][j] \leq 2 \cdot tla[i] \)

*This formulation is adapted from Custodio et al. (2024) [2].*

## Usage

**C++:**
```sh
g++ -std=c++11 -O2 mho.cpp -o mho
./mho
```

**Python:**
```sh
python mho.py
```

## Performance Analysis

The algorithm includes comprehensive logging and output generation:
- Population state tracking
- Fitness evolution
- Convergence analysis
- Detailed execution traces

## Research Contributions

1. **Novel Initialization Strategy**: Chaotic map-based population initialization.
2. **Enhanced Convergence**: Modified convergence factor for better performance.
3. **Reverse Learning Mechanism**: Small-hole imaging for local optima escape.
4. **Comprehensive Analysis**: Detailed implementation and performance evaluation.

## Citations

1. **MHO Algorithm Modifications:**
   - Han, T., Wang, H., Li, T., Liu, Q., & Huang, Y. (2025). MHO: A Modified Hippopotamus Optimization Algorithm for Global Optimization and Engineering Design Problems. Biomimetics (Basel), 10(2), 90. [https://pubmed.ncbi.nlm.nih.gov/39997113/](https://pubmed.ncbi.nlm.nih.gov/39997113/)

2. **Optimization Problem Formulation:**
   - Custodio, J. M., et al. (2024). Optimization of Crop Harvesting Schedules and Land Allocation Through Linear Programming. Process Integration and Optimization for Sustainability. [https://link.springer.com/article/10.1007/s41660-023-00357-4](https://link.springer.com/article/10.1007/s41660-023-00357-4)

## License

This project is part of academic research. Please cite appropriately if used in your work.

## Contributing

This is a thesis research project. For academic collaboration or questions, please contact the author.

---

*For detailed algorithm analysis and implementation specifics, refer to the source code and output files provided in this repository.* 
