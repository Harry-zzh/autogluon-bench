

import numpy as np

s = "0.956 & 0.417 & 0.752 & 0.586 & 0.879 & 0.926 & 0.617 & 0.932 & 0.399 & 0.865 & 0.624 & 0.804"
arr = s.split("&")
score = 0.
for a in arr:
    score += float(a)
print(np.around(score / len(arr), 3))