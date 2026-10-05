import pandas as pd

df = pd.read_csv('data/processed/all_qa_pairs.csv', dtype=str, keep_default_na=False)

target_ids = ['synpic50385', 'synpic50387', 'synpic27815', 'synpic27817', 'synpic27819']
sub = df[df['image_id'].isin(target_ids)]

print("ALL QA PAIRS FOR TARGET IMAGES:")
print("=" * 80)
for idx, r in sub.iterrows():
    print(f"{r['image_id']} | {r['category']:12} | {r['source']:10} | Q: {r['question']}")
    print(f"    --> Answer: {r['answer']}")

print("\n" + "=" * 80)
print("MATCHING ABNORMALITY ANSWERS IN ENTIRE DATASET:")
print("=" * 80)
abn_50387 = "penetrating injury (gunshot wound) to the head and temporal bone."
m1 = df[df['answer'] == abn_50387]
print(f"Matches for '{abn_50387}': {len(m1)}")
for idx, r in m1.iterrows():
    print(f"  {r['image_id']} | {r['category']} | {r['source']} | Q: {r['question']}")

abn_27819 = "gallbladder polyp"
m2 = df[df['answer'] == abn_27819]
print(f"\nMatches for '{abn_27819}': {len(m2)}")
for idx, r in m2.iterrows():
    print(f"  {r['image_id']} | {r['category']} | {r['source']} | Q: {r['question']}")
