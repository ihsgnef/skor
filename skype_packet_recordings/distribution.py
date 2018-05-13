import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import os

all_dists = []
files = ['org.txt', 'alpha2.txt','alpha4.txt','alpha6.txt','alpha8.txt', 'v40.txt']
for f in files:
    with open(f) as og:
        packet_sizes = []
        for line in og:
            try:
                packet_sizes.append(int(line.split("Len=")[1]))
            except:
                pass

        all_dists.append(packet_sizes)

print(len(all_dists))
for i,dist in enumerate(all_dists):
    plt.subplot(len(files), 1, i)
    plt.hist(dist, bins=50)


plt.show()
