

import numpy as np
s = "0.565 & 0.838 & 0.748 & 0.375"
arr = s.split("&")
score = 0.
for a in arr:
    score += float(a)
print(np.around(score / len(arr), 3))