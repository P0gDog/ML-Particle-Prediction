import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


h = 1.0
m = 1.0

# Generate dataset
def energy(n, L):
    return (n**2 * h**2) / (8 * m * L**2)

N = 1000
n_vals = np.random.randint(1, 10, size=N)
L_vals = np.random.uniform(0.5, 2.0, size=N)

E_vals = energy(n_vals, L_vals)

X = np.column_stack((n_vals, L_vals))
y = E_vals.reshape(-1, 1)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Neural network
model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training
for epoch in range(2000):
    pred = model(X)
    loss = loss_fn(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.5f}")

# Test prediction
with torch.no_grad():
    preds = model(X).numpy()

plt.scatter(y.numpy(), preds, s=5)
plt.xlabel("True Energy")
plt.ylabel("Predicted Energy")
plt.title("ML vs Quantum Energy Levels")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

L = 1.0
x = np.linspace(0, L, 1000)

for n in [1, 2, 3, 5]:
    psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    plt.plot(x, psi, label=f"n={n}")

plt.xlabel("Position x")
plt.ylabel("ψ(x)")
plt.title("Wavefunctions in a 1D Box")
plt.legend()
plt.show()

for n in [1, 2, 3, 5]:
    psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    plt.plot(x, psi**2, label=f"n={n}")

plt.xlabel("Position x")
plt.ylabel("|ψ(x)|²")
plt.title("Probability Density")
plt.legend()
plt.show()

n_vals = np.arange(1, 8)
E = n_vals**2

plt.hlines(E, xmin=0, xmax=1)
plt.yticks(E, [f"n={n}" for n in n_vals])
plt.xlabel("Box (arbitrary)")
plt.ylabel("Energy")
plt.title("Discrete Energy Levels")
plt.show()

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
line, = ax.plot(x, np.zeros_like(x))
ax.set_ylim(-2, 2)

def update(n):
    psi = np.sqrt(2/L) * np.sin(n * np.pi * x / L)
    line.set_ydata(psi)
    ax.set_title(f"Wavefunction n={n}")
    return line,

ani = FuncAnimation(fig, update, frames=range(1, 10), interval=700)
plt.show()

L_vals = np.linspace(0.2, 2, 100)
n = 3
E = n**2 / L_vals**2

plt.plot(L_vals, E)
plt.xlabel("Box Length L")
plt.ylabel("Energy")
plt.title("Energy vs Box Size")
plt.show()

errors = np.abs(preds.flatten() - y.numpy().flatten())

plt.scatter(n_vals, errors[:len(n_vals)])
plt.xlabel("Energy Level n")
plt.ylabel("Prediction Error")
plt.title("ML Error")
plt.show()
