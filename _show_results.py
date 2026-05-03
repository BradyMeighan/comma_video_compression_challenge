import csv
rows = [r for r in csv.DictReader(open('grid_search_results.csv')) if r['status']=='ok']
rows.sort(key=lambda r: float(r['total_score']))
for r in rows[:25]:
    print(f"{r['total_score']:>8s} codec={r['codec']} scale={r['scale']} crf={r['crf']} p={r['preset']} gop={r['gop']} seg={r['segnet_score']} pose={r['posenet_score']} rate={r['rate_score']} sz={r['archive_size_bytes']}")
print(f"\nTotal results: {len(rows)}")
