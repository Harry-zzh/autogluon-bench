

import numpy as np
s = "0.937 & 0.402 & 0.503 & 0.439 & 0.045 & 0.479 & 0.612 & 0.592 & 0.861 & 0.539 & 0.875 & 0.915 & 0.928 & 0.760 & 0.582 & 0.934 & 0.284 & 0.874 & 0.565 & 0.681 & 0.082 & 0.805"
arr = s.split("&")
score = 0.
for a in arr:
    score += float(a)
print(np.around(score / len(arr), 3))