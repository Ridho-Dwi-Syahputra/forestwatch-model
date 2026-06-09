import json

with open('notebooks/forestwatch_papua_full_pipeline.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

with open('patch_dir_dump.txt', 'w', encoding='utf-8') as out:
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            source = ''.join(c['source'])
            if 'PATCH_DIR' in source:
                out.write(source)
                out.write('\n---\n')
