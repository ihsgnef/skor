import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import os

all_dists = []
files = ['../baseline_wifi.txt', 'v30alpha.2.txt','v30alpha.4.txt','v30alpha.6.txt', 'v30alpha.8.txt', 'v30alpha1.txt', 'simplealpha.6.txt']
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
    plt.title(files[i])


plt.show()
