import json

with open('notebooks/forestwatch_papua_full_pipeline.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open('dump_out.txt', 'w', encoding='utf-8') as out:
    out.write("=== BAGIAN 9 ===\n")
    for c in nb['cells'][29:34]:
        if c['cell_type'] == 'code':
            out.write(''.join(c['source']))
            out.write('\n---\n')

    out.write("\n=== CLASS WEIGHT ===\n")
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            source = ''.join(c['source'])
            if 'class_weight' in source.lower():
                out.write(f"Cell Index {i}:\n")
                out.write(source)
                out.write('\n---\n')
