import seaborn as sns
import matplotlib.pyplot as plt

import os
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
sns.set_style("darkgrid")
tips = sns.load_dataset("tips")
sns.regplot(x="total_bill", y="tip", data=tips,line_kws={"color": "orange","linewidth":5})
plt.tight_layout()
plt.show()