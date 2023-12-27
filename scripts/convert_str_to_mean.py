

import numpy as np
s = " 0.560 & 0.818 & 0.728 & 0.371"
arr = s.split("&")
score = 0.
for a in arr:
    score += float(a)
print(np.around(score / len(arr), 3))