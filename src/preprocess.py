"""
Logical categories (comparative, spatial, temporal, quantifier, numerical),
implicature, and taxonomic stands out as relatively harder capabilities across
systems
"""

def process_tsv(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                label, premise, hypothesis = parts
                data.append((label, premise, hypothesis))
            else:
                print(f"Something wrong with following line: {line.strip()}")
    return data


