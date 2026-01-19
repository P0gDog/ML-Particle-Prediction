# ML-Particle-Prediction Basic Section

# Abstract
How accuratly can a trained model predict particle energy in a set box? The model only sees $n$ and $L$ in $$E_n = \frac{n^2 h^2}{8 m L^2}$$, and tries to predict $E$ based on training. This project shows limitations of ML when it comes to calculating things like this. It tries to predict based on only patterns, not actual physics. I sectioned this project off into different versions, each getting more in depth. Use the dropdown below to access them.
<details>
  <summary>Version 1</summary>
  
# Introduction
Over time, Machine Learning has become an increasingly powerful and popular tool in physics for analysis and patern recognition. The only issue with this, however, is how many physical systems are governed by exact mathmatical laws. This raises the question of when machine learning is useful, and when it fails. In this project, I explored this question using one of the simplest of quantum systems: a particle trapped in a 1D infinite potential well. Because the energy levels of this simulation are known analytically, it provides an ideal test simulation for comparing machine learning predictions to exact physical results.

# Physics
A particle in a 1D infinite potential well is confined between two unpassable walls seperated by a distance, $L$. Quantum mechanically, the particle is described by a wavefunction that must halt at the walls, allowing only standing wave solutions. Each wavefunction corresponds to a discrete energy level. The energy levels of this system are given by
$$E_n = \frac{n^2 h^2}{8 m L^2}$$, where $n$ is a positive integer representing the energy level. This system was chosen because of how well understood it's behavior is and because of how it provides a clear physical interpretation of quantization.

# Where ML comes in
To model the relationship between quantum number, box size, and enery, a feedforward neural network was trained using supervised learning. The input features were the energy level $n$ and box length $L$, whilst the target output was energy $E$. Training data was generated using an analythical energy formual over randomly sampled values of $n$ and $L$. The neural netowrk consisted of two layers that were hidden with the ReLU activation functions and was trained using mean squared error loss and the Adam optimizer. 

# Results
The first figure shows the prediction error of the trained neural network as a function of energy level. Whilst the error generally increases ant higher energy levels, this trend seems to not be strictly monotonic. This behavior arises because the neural network does not directly learn physical laws but instead approximates patterns within the training data. Since the energy level corresponds to a range of box lengths, prediction difficulty varies across different regions of paramater space. The rest of the figures are just used to aid in visualizing the varying energy levels of the wavefucntions. 

Real energy vs Predicted energy:

<img width="765" height="553" alt="Real Energy vs Predicted Energy" src="https://github.com/user-attachments/assets/112b0313-9776-49cf-b6d3-728bba8ccac8" />

Error based on Energy Level:

<img width="784" height="573" alt="Error based on Energy Level" src="https://github.com/user-attachments/assets/f08c057a-514d-4079-98b3-11e8bff25151" />

I found the above image quite odd. Why were the chart of errors so random? Shouldn't they be linear? After looking into it, I realized why. The machine was trained on different energly levels at random amounts. The chart below helps visualize this.
| n | Training samples |
|:-:|:----------------:|
| 1 | 140              |
| 2 | 98               |
| 3 | 121              |
| 4 | 87               |
| 5 | 156              |
| 6 | 72               |
| 7 | 103              |

In the next version, this will be fixed to aid in accuracy.

# Conclusion
The results show that while neural netowkrs can approximate simple quantum systems, their accuracy depends strongly on the distribution of training data and the curvature of the underlying physical relationship. At high energies and small box sizes, small changes in input lead to large changes in energy. This makes percise approximation more difficult. In contrast, the analytical physics solution still remains exact across al regimes. This highlights a key limiation of ML: it learns patterns rather than physical principals.

</details>
