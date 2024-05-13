import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



x = np.linspace(0, 500, 101)
y = 3.819922*np.log(89.08492+x)-17.26882
plt.grid()
plt.plot(x, y, color="black")
y = [0, 1, 2, 3, 4, 5, 6, 7]
x = [0, 35, 75, 115, 150, 250, 350, 500]
plt.scatter(x, y, color="red",zorder=10)

plt.xlabel("PM2.5 Daily Average concentration", fontsize=18)
plt.ylabel("Air Quality Grade", fontsize=18)
plt.savefig("curve_fit.pdf", bbox_inches="tight")
plt.show()