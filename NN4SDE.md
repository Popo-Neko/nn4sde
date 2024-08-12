[toc]

# NN4SDE

## Experiments on one dimension

### Whether Monte Carlo together with Euler-Maruyama a good simulation? (BS Call Option as example)

- Base parameters

  | params | $\mu$ | $\sigma$ | $K$: strike | $r_{f}$ | $T$: terminal time | $x_{0}$ |
  | ------ | ----- | -------- | ----------- | ------- | ------------------ | ------- |
  | value  | 0.05  | 0.2      | 1.25        | 0.05    | 1                  | 1       |

- Relative Error ($\frac{|y_{exact}-y_{simulate}|}{y_{exact}}$) , present in (mean, std)

  | M: paths; N: time steps | $N=100$      | $N=1000$              | $N=10000$           | $N=100000$        |
  | ----------------------- | ------------ | --------------------- | ------------------- | ----------------- |
  | $M=100$                 | 0.253, 0.176 | 0.252, 0.248          |                     |                   |
  | $M=1000$                |              | 0.084, 0.0563         |                     |                   |
  | $M=10000$               |              | 0.0257, 0.0197        | 0.0789, 0.0598      |                   |
  | $M=100000$              |              | **0.00778, 0.005586** | **0.00323, 0.0021** | 0.00819,  0.00621 |

  Conclusion: 