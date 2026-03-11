import csv
from collections import defaultdict

totals = defaultdict(lambda: {
    "count": 0, 
    "bare_errors": 0, "full_errors": 0,
    "bare_brand_match_true": 0, "bare_brand_match_total": 0,
    "full_brand_match_true": 0, "full_brand_match_total": 0,
    "bare_promo_detected_true": 0, "bare_promo_detected_total": 0,
    "full_promo_detected_true": 0, "full_promo_detected_total": 0,
    "bare_cta_detected_true": 0, "bare_cta_detected_total": 0,
    "full_cta_detected_true": 0, "full_cta_detected_total": 0,
    "bare_topic_match_true": 0, "bare_topic_match_total": 0,
    "full_topic_match_true": 0, "full_topic_match_total": 0,
    "bare_effectiveness": 0.0, "full_effectiveness": 0.0
})

with open('benchmark_results/benchmark/benchmark_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row['method']
        st = totals[method]
        st["count"] += 1
        
        if row.get('bare_error'): st["bare_errors"] += 1
        if row.get('full_error'): st["full_errors"] += 1
        
        # Track accuracy / detection
        for mode in ['bare', 'full']:
            # Brand Match
            bm = row.get(f'{mode}_brand_match')
            if bm and bm.lower() != 'null' and bm != '':
                st[f"{mode}_brand_match_total"] += 1
                if bm.lower() == 'true': st[f"{mode}_brand_match_true"] += 1
            
            # Promo Detected
            pd = row.get(f'{mode}_promo_detected')
            if pd and pd.lower() != 'null' and pd != '':
                st[f"{mode}_promo_detected_total"] += 1
                if pd.lower() == 'true': st[f"{mode}_promo_detected_true"] += 1
            
            # CTA Detected
            cd = row.get(f'{mode}_cta_detected')
            if cd and cd.lower() != 'null' and cd != '':
                st[f"{mode}_cta_detected_total"] += 1
                if cd.lower() == 'true': st[f"{mode}_cta_detected_true"] += 1
                
            # Topic Match
            tm = row.get(f'{mode}_topic_match')
            if tm and tm.lower() != 'null' and tm != '':
                st[f"{mode}_topic_match_total"] += 1
                if tm.lower() == 'true': st[f"{mode}_topic_match_true"] += 1
                
            # Effectiveness
            eff = row.get(f'{mode}_effectiveness')
            if eff and eff.lower() != 'null' and eff != '':
                st[f"{mode}_effectiveness"] += float(eff)

print(f"{'Method':<20} | {'Err(B/F)':<10} | {'BrandMatch(B/F)':<18} | {'PromoDet(B/F)':<18} | {'CTADet(B/F)':<18} | {'Topic(B/F)':<12} | {'Eff(B/F)':<12}")
print("-" * 125)
for method, st in totals.items():
    n = st["count"]
    if n == 0: continue
    
    err_b = f"{st['bare_errors']}/{n}"
    err_f = f"{st['full_errors']}/{n}"
    
    def pct(t, tot): return f"{(t/tot*100):.0f}%" if tot > 0 else "N/A"
    
    bm_b = pct(st['bare_brand_match_true'], st['bare_brand_match_total'])
    bm_f = pct(st['full_brand_match_true'], st['full_brand_match_total'])
    
    pd_b = pct(st['bare_promo_detected_true'], st['bare_promo_detected_total'])
    pd_f = pct(st['full_promo_detected_true'], st['full_promo_detected_total'])
    
    cd_b = pct(st['bare_cta_detected_true'], st['bare_cta_detected_total'])
    cd_f = pct(st['full_cta_detected_true'], st['full_cta_detected_total'])
    
    tm_b = pct(st['bare_topic_match_true'], st['bare_topic_match_total'])
    tm_f = pct(st['full_topic_match_true'], st['full_topic_match_total'])
    
    valid_eff_b = n - st['bare_errors']
    valid_eff_f = n - st['full_errors']
    eff_b = f"{(st['bare_effectiveness'] / valid_eff_b):.2f}" if valid_eff_b > 0 else "N/A"
    eff_f = f"{(st['full_effectiveness'] / valid_eff_f):.2f}" if valid_eff_f > 0 else "N/A"
    
    print(f"{method:<20} | {err_b:>4}/{err_f:<5} | {bm_b:>7} / {bm_f:<7} | {pd_b:>7} / {pd_f:<7} | {cd_b:>7} / {cd_f:<7} | {tm_b:>4}/{tm_f:<5} | {eff_b:>4}/{eff_f:<5}")
