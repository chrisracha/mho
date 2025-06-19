# Modified hippopotamus optimization (mho) algorithm

## overview

this repository contains the implementation of the modified hippopotamus optimization (mho) algorithm, a metaheuristic for constrained optimization. mho is an enhanced version of the original hippopotamus optimization (ho) algorithm, with modifications for improved convergence and solution quality.

## features

- sine chaotic map initialization
- modified convergence factor
- small-hole imaging reverse learning
- tested on crop allocation optimization

## file structure

```
├── mho.cpp           # main c++ implementation
├── mho.py            # python implementation
├── requirements.txt  # python dependencies
├── README.md         # this file
└── mancul files/
    └── obj func.pdf  # objective function documentation
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

## references

- custio, r. et al. (2024). crop allocation optimization problem. *journal of agricultural optimization*.
- han, y., li, x., & zhang, w. (2025). modified hippopotamus optimization algorithm. *applied soft computing*.
- original hippopotamus optimization (ho) algorithm
- chaotic map applications in metaheuristics
- reverse learning mechanisms in optimization

---

for details, see the source code and obj func.pdf. 