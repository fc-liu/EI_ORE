import numpy as np
import matplotlib.pyplot as plt


X_norm = np.random.rand(100, 2)
y = np.random.randint(0, 5, 100)
plt.figure(figsize=(20, 20))
plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y,
            cmap=plt.cm.get_cmap("jet", 10), marker='o')
plt.xticks([])
plt.yticks([])
plt.savefig('figures/test.pdf',bbox_inches='tight')
