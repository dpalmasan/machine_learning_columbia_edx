# Project 4: Matrix Factorization

This project consisted on implementing Probabilistic Matrix Factorization for recommender systems.

* hw4_PMF.py: Given some ratings dat, creates .csv files (output) with objective function in 50 iterations, and computed U-V matrices.

## Instructions
In this assignment you will implement the probabilistic matrix factorization (PMF) model. Recall that this model fills in the values of a missing matrix $M$, 
where $M_{ij}$ is an observed value if $(i, j) \in \Omega$, where $\Omega$ contains the measured pairs. The goal is to factorize this matrix into a product between vectors such that 
$M_{ij} \approx u_i^T v_j$, where each $u_i, v_j \in \mathbb{R}^d$.

The modeling problem is to learn $u_i$ for $i = 1, \ldots , N_u$ and $v_j$ for $j = 1, \ldots , N_v$ by maximizing the objective function:

$$L = -\Sum_{(i, j) \in \Omega} \frac{1}{2\sigma^2}(M_{ij} - u_i^T v_j)^2 - \Sum_{i=1}^{N_u} \frac{\lambda}{2} \|u_i\|^2 - \Sum_{j=1}^{N_v} \frac{\lambda}{2} \|v_j\|^2$$

For this problem set $d=5$, $\sigma^2 = \frac{1}{10}$ and $\lambda = 2$.

## To execute the code:

`python hw4_PMF.py <ratings.csv>`



