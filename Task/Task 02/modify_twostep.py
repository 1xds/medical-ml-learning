import json
import os

nb_path = r"D:\Research\medical-ml-learning\Task\Task 02\task.ipynb"
task_dir = r"D:\Research\medical-ml-learning\Task\Task 02"

nb = json.load(open(nb_path, encoding='utf-8'))
cells = nb['cells']

def read_code_cell(path):
    """Read a .py file and convert to notebook code cell."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Remove trailing empty lines
    while lines and lines[-1].strip() == '':
        lines.pop()
    # Ensure each line ends with \n
    for i in range(len(lines)):
        if not lines[i].endswith('\n'):
            lines[i] += '\n'
    return {
        "cell_type": "code",
        "metadata": {},
        "source": lines,
        "outputs": [],
        "execution_count": None
    }

def make_md_cell(text):
    """Convert text to notebook markdown cell."""
    lines = text.split('\n')
    source = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source.append(line + '\n')
        else:
            if line:
                source.append(line + '\n')
    # Remove trailing empty entries
    while source and source[-1].strip() == '':
        source.pop()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

def join_and_split(cell):
    """Join cell source lines and re-split for modification."""
    text = ''.join(cell['source'])
    lines = text.split('\n')
    result = []
    for line in lines:
        result.append(line + '\n')
    # Remove trailing empty line (just '\n')
    while result and result[-1] == '\n':
        result.pop()
    return result

# === 1. Modify Cell 0: add CORR_THRESHOLD ===
cell0_text = ''.join(cells[0]['source'])
cell0_text = cell0_text.replace(
    'K_BEST = 20\n',
    'K_BEST = 20\nCORR_THRESHOLD = 0.9\n'
)
cells[0]['source'] = join_and_split(cells[0])
# Re-apply modification on the joined-split format
cells[0]['source'] = [line.replace('K_BEST = 20', 'K_BEST = 20\nCORR_THRESHOLD = 0.9') if 'K_BEST = 20' in line else line for line in cells[0]['source']]
# Actually, simpler approach: just directly modify the joined text
cell0_text = ''.join(cells[0]['source'])
cell0_text = cell0_text.replace('K_BEST = 20', 'K_BEST = 20\nCORR_THRESHOLD = 0.9')
# Now split back
lines = cell0_text.split('\n')
cells[0]['source'] = [line + '\n' for line in lines]
while cells[0]['source'] and cells[0]['source'][-1] == '\n':
    cells[0]['source'].pop()
cells[0]['outputs'] = []
cells[0]['execution_count'] = None
print('[Modify] Cell 0: added CORR_THRESHOLD = 0.9')

# === 2. Modify Cell 1: add feature selection section ===
cell1_text = ''.join(cells[1]['source'])
cell1_text = cell1_text.replace(
    '## Evaluation',
    '## Feature Selection\n\nTwo-step approach: correlation de-redundancy (|r| > 0.9) -> SelectKBest (ANOVA F-test, k=20)\n\n## Evaluation'
)
lines = cell1_text.split('\n')
cells[1]['source'] = [line + '\n' for line in lines]
while cells[1]['source'] and cells[1]['source'][-1] == '\n':
    cells[1]['source'].pop()
print('[Modify] Cell 1: added Feature Selection section')

# === 3. Modify Cell 15: update EDA note ===
cell15_text = ''.join(cells[15]['source'])
cell15_text = cell15_text.replace(
    'This is **EDA only** — visualization for understanding, not for deleting features before train/test split.',
    'This is **EDA only** — visualization for understanding data structure.\n\nThe actual correlation-based de-redundancy (Step 2 of feature selection) is computed on **training data only** after train/test split, to avoid data leakage.'
)
lines = cell15_text.split('\n')
cells[15]['source'] = [line + '\n' for line in lines]
while cells[15]['source'] and cells[15]['source'][-1] == '\n':
    cells[15]['source'].pop()
print('[Modify] Cell 15: updated EDA note to link to two-step approach')

# === 4. Insert new markdown cell at index 17 ===
md17_text = """## Two-Step Feature Selection Strategy

### Why two steps?

**SelectKBest limitation**: It scores each feature individually (ANOVA F-test), **without considering feature redundancy**. If two features have |r| > 0.9 but both score high, SelectKBest keeps both — wasting one slot on nearly identical information.

Example: Area and Perimeter both have high F-scores but |r| = 0.99. SelectKBest(k=2) keeps both, while Entropy (medium F-score, independent) gets dropped. The model ends up with redundant size information instead of complementary shape + intensity information.

**Standard radiomics practice**: Two-step approach:

```text
Step 1: Unsupervised de-redundancy — remove |r| > CORR_THRESHOLD pairs (does not depend on labels)
Step 2: Supervised selection — SelectKBest (ANOVA F-test) on remaining features
```

**Data leakage prevention**: Both steps are computed on **training data only** after train/test split. The test set is only transformed, never used for selection decisions.

This ensures:
- No redundant features waste selection slots
- Features from complementary families (e.g., shape2D <-> firstorder, r ~ 0.15) are prioritized
- The 7x7 family-level correlation analysis (EDA above) validates whether the final selection is balanced"""
cells.insert(17, make_md_cell(md17_text))
print('[Insert] Cell 17: Two-Step Feature Selection Strategy markdown')

# === 5. Replace Cell 18 (was Cell 17): two-step feature selection ===
cells[18] = read_code_cell(os.path.join(task_dir, 'cell18_code.py'))
print('[Rewrite] Cell 18: two-step feature selection code')

# === 6. Replace Cell 19 (was Cell 18 markdown): distribution explanation ===
md19_text = """## Two-Step Feature Selection Distribution by Family

How many features from each family survive each step?

- **After Step 2** (correlation filtering |r| > 0.9): shows how many redundant features were removed per family
- **After Step 3** (SelectKBest k=20): shows the final selection balance across families

A well-balanced selection should have features from **multiple families**, reflecting the 7x7 correlation insight: families with low inter-family correlation (e.g., shape2D <-> firstorder ~ 0.15) provide complementary information and should both be represented."""
cells[19] = make_md_cell(md19_text)
print('[Rewrite] Cell 19: distribution markdown')

# === 7. Replace Cell 20 (was Cell 19 code): distribution chart ===
cells[20] = read_code_cell(os.path.join(task_dir, 'cell20_code.py'))
print('[Rewrite] Cell 20: distribution chart code')

# === 8. Clear outputs for all code cells from Cell 18 onwards ===
for i in range(18, len(cells)):
    if cells[i].get('cell_type') == 'code':
        cells[i]['outputs'] = []
        cells[i]['execution_count'] = None
print(f'[Clear] Outputs cleared for cells 18..{len(cells)-1}')

# === Save to workspace (sandbox allows writes here) ===
ws_path = r"C:\Users\Lenovo\WorkBuddy\2026-07-22-15-07-07\task_modified.ipynb"
with open(ws_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'\nTotal cells: {len(cells)}')
print(f'Saved to workspace: {ws_path}')
