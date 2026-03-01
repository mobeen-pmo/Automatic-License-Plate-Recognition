"""
Generate research paper visualizations from ALPR pipeline output data.
Produces publication-quality graphs and tables from test.csv and test_interpolated.csv.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─── Configuration ───────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ─── Load Data ───────────────────────────────────────────────────
print("[1/8] Loading CSV data...")
raw = pd.read_csv('test.csv')
interp = pd.read_csv('test_interpolated.csv')

print(f"  Raw: {len(raw)} rows, {raw['car_id'].nunique()} unique vehicles")
print(f"  Interpolated: {len(interp)} rows, {interp['car_id'].nunique()} unique vehicles")

# ─── Figure 1: OCR Confidence Distribution ──────────────────────
print("[2/8] Generating OCR Confidence Distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
valid_scores = raw['license_number_score'].dropna()
valid_scores = valid_scores[valid_scores > 0]

counts, bins, patches = ax.hist(valid_scores, bins=20, color='#2196F3', edgecolor='white',
                                 alpha=0.85, linewidth=0.8)
# Color gradient
cm = plt.cm.RdYlGn
for i, (count, patch) in enumerate(zip(counts, patches)):
    patch.set_facecolor(cm(bins[i] / max(bins)))

ax.axvline(valid_scores.mean(), color='#E53935', linestyle='--', linewidth=2,
           label=f'Mean = {valid_scores.mean():.3f}')
ax.axvline(valid_scores.median(), color='#FF9800', linestyle='-.', linewidth=2,
           label=f'Median = {valid_scores.median():.3f}')

ax.set_xlabel('OCR Confidence Score')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of EasyOCR Confidence Scores Across All Detections')
ax.legend(framealpha=0.9)
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_ocr_confidence_distribution.png'))
plt.close()
print("  [OK] Saved fig1_ocr_confidence_distribution.png")

# ─── Figure 2: Detection Count per Vehicle ──────────────────────
print("[3/8] Generating Detection Count per Vehicle...")
fig, ax = plt.subplots(figsize=(10, 5))

raw_counts = raw.groupby('car_id').size().reset_index(name='raw_count')
interp_counts = interp.groupby('car_id').size().reset_index(name='interp_count')
merged = pd.merge(raw_counts, interp_counts, on='car_id', how='outer').fillna(0)
merged = merged.sort_values('interp_count', ascending=False).head(12)

x = np.arange(len(merged))
width = 0.35

bars1 = ax.bar(x - width/2, merged['raw_count'], width, label='Raw Detections',
               color='#1976D2', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, merged['interp_count'], width, label='After Interpolation',
               color='#43A047', edgecolor='white', linewidth=0.5)

ax.set_xlabel('Vehicle ID (car_id)')
ax.set_ylabel('Number of Frame Entries')
ax.set_title('Raw vs. Interpolated Detection Counts per Tracked Vehicle')
ax.set_xticks(x)
ax.set_xticklabels([f'ID {int(cid)}' for cid in merged['car_id']], rotation=45, ha='right')
ax.legend()

for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2., h + 2, f'{int(h)}',
                ha='center', va='bottom', fontsize=8)
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2., h + 2, f'{int(h)}',
                ha='center', va='bottom', fontsize=8)

plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_raw_vs_interpolated_counts.png'))
plt.close()
print("  [OK] Saved fig2_raw_vs_interpolated_counts.png")

# ─── Figure 3: Plate Detection Confidence vs OCR Confidence ─────
print("[4/8] Generating Detection vs OCR Confidence Scatter...")
fig, ax = plt.subplots(figsize=(8, 6))

valid = raw.dropna(subset=['license_plate_bbox_score', 'license_number_score'])
valid = valid[(valid['license_plate_bbox_score'] > 0) & (valid['license_number_score'] > 0)]

scatter = ax.scatter(valid['license_plate_bbox_score'], valid['license_number_score'],
                     c=valid['car_id'], cmap='tab10', alpha=0.5, s=20, edgecolors='none')
ax.set_xlabel('License Plate Detection Confidence (BBox Score)')
ax.set_ylabel('OCR Recognition Confidence Score')
ax.set_title('Plate Detection Confidence vs. OCR Recognition Confidence')

# Add regression line
z = np.polyfit(valid['license_plate_bbox_score'], valid['license_number_score'], 1)
p = np.poly1d(z)
x_line = np.linspace(valid['license_plate_bbox_score'].min(),
                     valid['license_plate_bbox_score'].max(), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7,
        label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
ax.legend()

plt.colorbar(scatter, ax=ax, label='Vehicle ID')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_detection_vs_ocr_confidence.png'))
plt.close()
print("  [OK] Saved fig3_detection_vs_ocr_confidence.png")

# ─── Figure 4: Vehicle Tracking Timeline ────────────────────────
print("[5/8] Generating Vehicle Tracking Timeline...")
fig, ax = plt.subplots(figsize=(12, 5))

top_cars = raw.groupby('car_id')['frame_nmr'].agg(['min', 'max', 'count'])
top_cars = top_cars.sort_values('min').head(12)

colors = plt.cm.Set3(np.linspace(0, 1, len(top_cars)))
for i, (car_id, row) in enumerate(top_cars.iterrows()):
    ax.barh(i, row['max'] - row['min'], left=row['min'], height=0.6,
            color=colors[i], edgecolor='gray', linewidth=0.5)
    ax.text(row['max'] + 5, i, f'{int(row["count"])} det.', va='center', fontsize=9)

ax.set_yticks(range(len(top_cars)))
ax.set_yticklabels([f'Car {int(cid)}' for cid in top_cars.index])
ax.set_xlabel('Frame Number')
ax.set_title('Vehicle Tracking Timeline - Frame Span per Tracked Vehicle')
ax.invert_yaxis()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_vehicle_tracking_timeline.png'))
plt.close()
print("  [OK] Saved fig4_vehicle_tracking_timeline.png")

# ─── Figure 5: OCR Confidence Over Time for Top Vehicles ────────
print("[6/8] Generating OCR Confidence Over Time...")
fig, ax = plt.subplots(figsize=(10, 5))

top_5_cars = raw.groupby('car_id').size().nlargest(5).index
colors_line = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
for i, car_id in enumerate(top_5_cars):
    car_data = raw[raw['car_id'] == car_id].sort_values('frame_nmr')
    valid_data = car_data[car_data['license_number_score'] > 0]
    if len(valid_data) > 0:
        ax.plot(valid_data['frame_nmr'], valid_data['license_number_score'],
                alpha=0.7, linewidth=1.2, color=colors_line[i], label=f'Car {int(car_id)}')

ax.set_xlabel('Frame Number')
ax.set_ylabel('OCR Confidence Score')
ax.set_title('OCR Confidence Score Variation Over Time (Top 5 Vehicles)')
ax.legend(loc='upper right')
ax.set_ylim(0, 1.05)
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_ocr_confidence_over_time.png'))
plt.close()
print("  [OK] Saved fig5_ocr_confidence_over_time.png")

# ─── Figure 6: Interpolation Coverage Pie Chart ─────────────────
print("[7/8] Generating Interpolation Coverage Pie Chart...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
raw_only = len(raw)
imputed = len(interp) - len(raw)
axes[0].pie([raw_only, imputed], labels=['Original\nDetections', 'Interpolated\nFrames'],
            autopct='%1.1f%%', colors=['#1976D2', '#66BB6A'],
            explode=(0.05, 0.05), startangle=90,
            textprops={'fontsize': 12})
axes[0].set_title(f'Data Composition\n(Total: {len(interp)} entries)')

# Summary statistics table
stats_data = [
    ['Total Raw Entries', f'{len(raw):,}'],
    ['Total Interpolated', f'{len(interp):,}'],
    ['Imputed Entries', f'{imputed:,}'],
    ['Data Increase', f'+{(imputed/len(raw))*100:.1f}%'],
    ['Unique Vehicles', f'{raw["car_id"].nunique()}'],
    ['Frame Range', f'0-{raw["frame_nmr"].max()}'],
    ['Mean OCR Score', f'{valid_scores.mean():.3f}'],
    ['Max OCR Score', f'{valid_scores.max():.3f}'],
    ['Plate Detect Range', f'{valid["license_plate_bbox_score"].min():.2f}-{valid["license_plate_bbox_score"].max():.2f}'],
]

axes[1].axis('off')
table = axes[1].table(cellText=stats_data, colLabels=['Metric', 'Value'],
                       loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.5)
for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_facecolor('#1976D2')
        cell.set_text_props(color='white', fontweight='bold')
    elif key[0] % 2 == 0:
        cell.set_facecolor('#E3F2FD')
axes[1].set_title('Summary Statistics', pad=20)

plt.suptitle('Temporal Interpolation Impact Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_interpolation_analysis.png'))
plt.close()
print("  [OK] Saved fig6_interpolation_analysis.png")

# ─── Figure 7: Per-Vehicle Best OCR Score ────────────────────────
print("[8/8] Generating Per-Vehicle Best OCR Score...")
fig, ax = plt.subplots(figsize=(10, 5))

best_scores = raw.groupby('car_id').agg(
    best_score=('license_number_score', 'max'),
    mean_score=('license_number_score', 'mean'),
    detections=('frame_nmr', 'count')
).reset_index()
best_scores = best_scores[best_scores['best_score'] > 0].sort_values('best_score', ascending=True)

y = np.arange(len(best_scores))
bars = ax.barh(y, best_scores['best_score'], color='#43A047', edgecolor='white',
               alpha=0.85, height=0.6, label='Best Score')
ax.barh(y, best_scores['mean_score'], color='#1976D2', edgecolor='white',
        alpha=0.6, height=0.4, label='Mean Score')

ax.set_yticks(y)
ax.set_yticklabels([f'Car {int(cid)}' for cid in best_scores['car_id']])
ax.set_xlabel('OCR Confidence Score')
ax.set_title('Best vs. Mean OCR Confidence Score per Vehicle')
ax.legend(loc='lower right')
ax.set_xlim(0, 1.05)
ax.axvline(0.5, color='red', linestyle=':', alpha=0.5, label='Threshold 0.5')

for i, (best, mean) in enumerate(zip(best_scores['best_score'], best_scores['mean_score'])):
    ax.text(best + 0.02, i, f'{best:.3f}', va='center', fontsize=9)

plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_per_vehicle_ocr_scores.png'))
plt.close()
print("  [OK] Saved fig7_per_vehicle_ocr_scores.png")

# ─── Figure 8: License Plate Bounding Box Area vs OCR Confidence ───
print("[9/11] Generating Plate BBox Area vs OCR Confidence...")
fig, ax = plt.subplots(figsize=(8, 6))

valid = valid.copy() # Avoid SettingWithCopyWarning
def compute_area(bbox_str):
    try:
        clean_str = str(bbox_str).replace('[', '').replace(']', '').strip()
        parts = [float(x) for x in clean_str.split()]
        if len(parts) == 4:
            return (parts[2] - parts[0]) * (parts[3] - parts[1])
    except:
        pass
    return np.nan

valid['plate_area'] = valid['license_plate_bbox'].apply(compute_area)
valid_area = valid.dropna(subset=['plate_area', 'license_number_score'])

if len(valid_area) > 0:
    scatter = ax.scatter(valid_area['plate_area'], valid_area['license_number_score'],
                         alpha=0.5, c='#9C27B0', edgecolors='none')
    ax.set_xlabel('License Plate Bounding Box Area (pixels)')
    ax.set_ylabel('OCR Recognition Confidence Score')
    ax.set_title('Plate Size (Resolution) vs. OCR Confidence')

    if len(valid_area) > 1:
        z = np.polyfit(valid_area['plate_area'], valid_area['license_number_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_area['plate_area'].min(), valid_area['plate_area'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend: y={z[0]:.2e}x+{z[1]:.2f}')
        ax.legend()

plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_plate_area_vs_ocr.png'))
plt.close()
print("  [OK] Saved fig8_plate_area_vs_ocr.png")

# ─── Figure 9: OCR Confidence Boxplots by Top Vehicles ────────────
print("[10/11] Generating OCR Confidence Boxplots...")
fig, ax = plt.subplots(figsize=(10, 6))

top_n_cars = raw.groupby('car_id').size().nlargest(8).index
box_data = [raw[(raw['car_id'] == cid) & (raw['license_number_score'] > 0)]['license_number_score'].dropna() for cid in top_n_cars]

valid_box_data = [d for d in box_data if len(d) > 0]
valid_top_cars = [top_n_cars[i] for i in range(len(box_data)) if len(box_data[i]) > 0]

if len(valid_box_data) > 0:
    ax.boxplot(valid_box_data, patch_artist=True,
               boxprops=dict(facecolor='#BBDEFB', color='#1976D2'),
               medianprops=dict(color='#E53935', linewidth=2),
               flierprops=dict(marker='o', markerfacecolor='#9E9E9E', markersize=4, alpha=0.5))
    
    ax.set_xticklabels([f'Car {int(cid)}' for cid in valid_top_cars])
    ax.set_ylabel('OCR Confidence Score')
    ax.set_title('Distribution of OCR Confidence Scores Across Top Vehicles')
    ax.set_ylim(0, 1.05)

plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_ocr_boxplots_by_vehicle.png'))
plt.close()
print("  [OK] Saved fig9_ocr_boxplots_by_vehicle.png")

# ─── Figure 10: Detection Density Over Time ─────────────────────
print("[11/11] Generating Detection Density Over Time...")
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(raw['frame_nmr'].dropna(), bins=50, color='#009688', edgecolor='white', alpha=0.85)
ax.set_xlabel('Frame Number')
ax.set_ylabel('Number of Detections')
ax.set_title('Vehicle Detection Frequency Across Video Frames')

plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_detection_density_over_time.png'))
plt.close()
print("  [OK] Saved fig10_detection_density_over_time.png")
print("\n" + "="*60)
print("ALL FIGURES GENERATED SUCCESSFULLY!")
print("="*60)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Files created:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  - {f} ({size_kb:.0f} KB)")

# ─── Print detailed stats for the prompt ─────────────────────────
print("\n" + "="*60)
print("DETAILED STATISTICS FOR RESEARCH PAPER")
print("="*60)
print(f"\nRaw Entries: {len(raw)}")
print(f"Interpolated Entries: {len(interp)}")
print(f"Imputed: {len(interp) - len(raw)} ({(len(interp)-len(raw))/len(raw)*100:.1f}%)")
print(f"Unique Vehicles: {raw['car_id'].nunique()}")
print(f"Frame Range: {raw['frame_nmr'].min()}-{raw['frame_nmr'].max()}")
print(f"\nOCR Confidence Stats:")
print(f"  Min: {valid_scores.min():.4f}")
print(f"  Max: {valid_scores.max():.4f}")
print(f"  Mean: {valid_scores.mean():.4f}")
print(f"  Median: {valid_scores.median():.4f}")
print(f"  Std Dev: {valid_scores.std():.4f}")
print(f"\nPlate Detection BBox Score:")
print(f"  Min: {valid['license_plate_bbox_score'].min():.4f}")
print(f"  Max: {valid['license_plate_bbox_score'].max():.4f}")
print(f"  Mean: {valid['license_plate_bbox_score'].mean():.4f}")

print(f"\nPer-Vehicle Stats:")
for car_id in sorted(raw['car_id'].unique()):
    car = raw[raw['car_id'] == car_id]
    car_valid = car[car['license_number_score'] > 0]
    plates = car['license_number'].dropna().unique()
    plates = [p for p in plates if str(p) != '0']
    best = car_valid['license_number_score'].max() if len(car_valid) > 0 else 0
    print(f"  Car {int(car_id):>4d}: {len(car):>4d} detections, "
          f"frames {int(car['frame_nmr'].min()):>4d}-{int(car['frame_nmr'].max()):>4d}, "
          f"best OCR: {best:.3f}, plates: {plates[:3]}")
