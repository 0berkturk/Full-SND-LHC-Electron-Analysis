#!/usr/bin/env python3
import argparse
import csv
import sys

def main():
    parser = argparse.ArgumentParser(description="Hazır CSV tablosunu okuyup hedef Luminosity'ye göre ölçeklendirir ve ekrana basar.")
    parser.add_argument("--input", "-i", default="plots/summary_table.csv", help="Okunacak CSV dosyasının yolu (örn: tablo.csv)")
    parser.add_argument("--target-lumi", "-t", type=float, required=True, help="Tabloyu görmek istediğiniz hedef Lumi (fb^-1 cinsinden)")
    args = parser.parse_args()

    # Orijinal verinin 100 fb^-1 olduğu sabit olarak kabul ediliyor
    base_lumi = 100.0
    scale_factor = args.target_lumi / base_lumi

    print(f"\n[LUMINOSITY BİLGİSİ]")
    print(f"  Orijinal Veri (Simüle Edilen) : {base_lumi:.1f} fb^-1")
    print(f"  Hedeflenen Lumi               : {args.target_lumi:.1f} fb^-1")
    print(f"  Ağırlık Çarpanı               : {scale_factor:.6f}\n")

    print(f"{'='*85}")
    print(f"  Lumi @ {args.target_lumi:.2f} fb^-1")
    print(f"{'='*85}")
    print(f"  {'':6} | {'CC (e)':>9} {'CC (μ)':>9} {'CC (τ)':>9} | {'NC (e)':>9} {'NC (μ)':>9} {'NC (τ)':>9} | {'Total--':>10}")
    print(f"  {'-'*85}")

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader) # İlk satırı (Başlıkları) atla
            
            for row in reader:
                if not row: # Boş satırları atla
                    continue
                
                st = row[0]
                # Sayıları çarpanla (scale_factor) çarp
                vals = [float(x) * scale_factor for x in row[1:]]
                
                if st.upper() == "TOTAL":
                    print(f"  {'-'*85}")
                    
                print(f"  {st:<6} | {vals[0]:>9.2f} {vals[1]:>9.2f} {vals[2]:>9.2f} | {vals[3]:>9.2f} {vals[4]:>9.2f} {vals[5]:>9.2f} | {vals[6]:>10.2f}")
                
    except FileNotFoundError:
        print(f"[HATA] '{args.input}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
        sys.exit(1)

    print(f"{'='*85}\n")

if __name__ == "__main__":
    main()