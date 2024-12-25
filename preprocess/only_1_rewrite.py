import json

with open('datasets/MQuAKE-CF-3k.json', 'r') as f:
    dataset = json.load(f)


new_dataset = []

for x in dataset:
    if len(x['requested_rewrite'])==1:
        new_dataset.append(x)

with open('datasets/MQuAKE-1R.json', 'w') as f:
    json.dump(new_dataset, f, indent=4)