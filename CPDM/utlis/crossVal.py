import random

def generateSubsets():
    elements = list(range(58))
    random.shuffle(elements)
    subsets = [[] for _ in range(15)]
    
    index = 0
    for element in elements:
        if index < 13 * 4:
            subsets[index // 4].append(element)
        else:
            subsets[13 + (index - 13 * 4) // 3].append(element)
        index += 1
    
    for i in range(13):
        while len(subsets[i]) < 4:
            subsets[i].append(subsets[-1].pop())
    
    return subsets

# Test
subsets = generateSubsets()
for i, subset in enumerate(subsets):
    print(f"Subset {i+1}: {subset}")
