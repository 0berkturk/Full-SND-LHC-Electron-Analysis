#!/usr/bin/env python3
"""
GENIE gst.root Neutrino Interaction Analyzer & Plotter
======================================================
SND@LHC MC Production Analizi ve Grafik Üretimi
"""

import argparse
import glob
import sys
import numpy as np
import os
import csv

try:
    import uproot
except ImportError:
    print("uproot bulunamadı. Kurmak için: pip install uproot awkward")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib bulunamadı. Kurmak için: pip install matplotlib")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────────────────────────────────────
PDG_NAMES = {
    12:  "νe",
    -12: "ν̄e",
    14:  "νμ",
    -14: "ν̄μ",
    16:  "ντ",
    -16: "ν̄τ",
}

class PlotDataAccumulator:
    def __init__(self):
        self.data = {
            "Ev": [], "Q2": [], "y": [], "x": [],
            "wght": [], "neu": [], "cc": [], "nc": [],
            "qel": [], "res": [], "dis": [], "coh": [], "mec": []
        }

    def append(self, branches, masks):
        n_events = len(branches["Ev"])
        self.data["Ev"].append(branches["Ev"])
        if "Q2" in branches: self.data["Q2"].append(branches["Q2"])
        if "y" in branches: self.data["y"].append(branches["y"])
        if "x" in branches: self.data["x"].append(branches["x"])
        
        self.data["wght"].append(branches["wght"])
        self.data["neu"].append(branches["neu"])
        self.data["cc"].append(branches["cc"])
        self.data["nc"].append(branches["nc"])
        self.data["qel"].append(branches["qel"])
        self.data["res"].append(branches["res"])
        self.data["dis"].append(branches["dis"])
        self.data["coh"].append(branches["coh"])
        
        if "mec" in branches:
            self.data["mec"].append(branches["mec"])
        else:
            self.data["mec"].append(np.zeros(n_events, dtype=bool))

    def concatenate(self):
        concat_data = {}
        for k, v in self.data.items():
            if len(v) > 0:
                concat_data[k] = np.concatenate(v)
        return concat_data

def load_branches(tree):
    needed = ["neu", "cc", "nc", "qel", "res", "dis", "coh", "mec", "dfr",
              "Ev", "Q2", "W", "x", "y", "wght", "El", "nf"]
    available = set(tree.keys())
    branches = {}
    missing = []
    for b in needed:
        if b in available:
            branches[b] = tree[b].array(library="np")
        else:
            missing.append(b)
    if missing:
        print(f"  [UYARI] Şu branch'ler dosyada yok: {missing}")
    return branches

def make_masks(b):
    masks = {}
    masks["CC"]   = b["cc"].astype(bool)
    masks["NC"]   = b["nc"].astype(bool)
    masks["QEL"]  = b["qel"].astype(bool)
    masks["RES"]  = b["res"].astype(bool)
    masks["DIS"]  = b["dis"].astype(bool)
    masks["COH"]  = b["coh"].astype(bool)
    if "mec" in b:
        masks["MEC"] = b["mec"].astype(bool)
    else:
        masks["MEC"] = np.zeros(len(b["qel"]), dtype=bool)

    for pdg, name in PDG_NAMES.items():
        masks[name] = (b["neu"] == pdg)

    masks["νe_all"]  = masks["νe"]  | masks.get("ν̄e", np.zeros_like(b["qel"]))
    masks["νμ_all"]  = masks["νμ"]  | masks.get("ν̄μ", np.zeros_like(b["qel"]))
    masks["ντ_all"]  = masks["ντ"]  | masks.get("ν̄τ", np.zeros_like(b["qel"]))
    masks["ALL"]     = np.ones(len(b["qel"]), dtype=bool)
    return masks

def weighted_count(mask, weights):
    return float(np.sum(weights[mask]))

# ─────────────────────────────────────────────────────────────────────────────
# Tablo Çıktısı ve CSV Kaydetme Fonksiyonu
# ─────────────────────────────────────────────────────────────────────────────
def save_and_print_summary_table(acc_data, target_lumi, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = make_masks(acc_data)
    w = acc_data["wght"]
    scatter_types = ["QEL", "RES", "DIS", "COH", "MEC"]

    print(f"\n\n{'='*85}")
    print(f"  TÜM DOSYALARIN TOPLAM ÖZET TABLOSU (@ {target_lumi:.1f} fb^-1)")
    print(f"{'='*85}")
    print(f"  SCATTERING × INTERACTION TİPİ × FLAVOR (weighted)")
    print(f"{'─'*85}")
    print(f"  {'':6} | {'CC (e)':>9} {'CC (μ)':>9} {'CC (τ)':>9} | {'NC (e)':>9} {'NC (μ)':>9} {'NC (τ)':>9} | {'TOPLAM':>10}")
    print(f"  {'-'*85}")
    
    sums = {'cc_e': 0, 'cc_mu': 0, 'cc_tau': 0, 'nc_e': 0, 'nc_mu': 0, 'nc_tau': 0, 'tot': 0}
    
    csv_rows = []
    
    for st in scatter_types:
        m_st = m[st]
        
        # CC Flavor Breakdown
        cc_e   = weighted_count(m_st & m["CC"] & m["νe_all"], w)
        cc_mu  = weighted_count(m_st & m["CC"] & m["νμ_all"], w)
        cc_tau = weighted_count(m_st & m["CC"] & m["ντ_all"], w)
        
        # NC Flavor Breakdown
        nc_e   = weighted_count(m_st & m["NC"] & m["νe_all"], w)
        nc_mu  = weighted_count(m_st & m["NC"] & m["νμ_all"], w)
        nc_tau = weighted_count(m_st & m["NC"] & m["ντ_all"], w)
        
        tot = weighted_count(m_st, w)
        
        sums['cc_e'] += cc_e; sums['cc_mu'] += cc_mu; sums['cc_tau'] += cc_tau
        sums['nc_e'] += nc_e; sums['nc_mu'] += nc_mu; sums['nc_tau'] += nc_tau
        sums['tot'] += tot
        
        print(f"  {st:<6} | {cc_e:>9.2f} {cc_mu:>9.2f} {cc_tau:>9.2f} | {nc_e:>9.2f} {nc_mu:>9.2f} {nc_tau:>9.2f} | {tot:>10.2f}")
        
        # CSV için satırı kaydet
        csv_rows.append([st, round(cc_e, 2), round(cc_mu, 2), round(cc_tau, 2), 
                         round(nc_e, 2), round(nc_mu, 2), round(nc_tau, 2), round(tot, 2)])

    print(f"  {'-'*85}")
    print(f"  {'TOTAL':<6} | {sums['cc_e']:>9.2f} {sums['cc_mu']:>9.2f} {sums['cc_tau']:>9.2f} | {sums['nc_e']:>9.2f} {sums['nc_mu']:>9.2f} {sums['nc_tau']:>9.2f} | {sums['tot']:>10.2f}")
    print(f"{'='*85}\n")

    # CSV DOSYASINA YAZMA İŞLEMİ
    csv_rows.append(["TOTAL", round(sums['cc_e'], 2), round(sums['cc_mu'], 2), round(sums['cc_tau'], 2), 
                     round(sums['nc_e'], 2), round(sums['nc_mu'], 2), round(sums['nc_tau'], 2), round(sums['tot'], 2)])
                     
    csv_path = os.path.join(out_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scattering", "CC_e", "CC_mu", "CC_tau", "NC_e", "NC_mu", "NC_tau", "TOTAL"])
        writer.writerows(csv_rows)
        
    print(f"  ✓ Tablo başarıyla CSV olarak kaydedildi: {csv_path}")


def draw_all_plots(acc_data, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  ✓ Grafikler {out_dir}/ klasörüne kaydediliyor...")

    m = make_masks(acc_data)
    Ev = acc_data["Ev"]
    w = acc_data["wght"]

    valid_Ev = Ev[Ev > 0]
    min_Ev = np.min(valid_Ev) if len(valid_Ev) > 0 else 1.0
    max_Ev = np.max(Ev) if len(Ev) > 0 else 10000.0
    bins_E = np.logspace(np.log10(min_Ev), np.log10(max_Ev), 40)

    flavors = [
        ("νe_all", "Electron Neutrinos ($\\nu_e + \\bar{\\nu}_e$)"),
        ("νμ_all", "Muon Neutrinos ($\\nu_\\mu + \\bar{\\nu}_\\mu$)"),
        ("ντ_all", "Tau Neutrinos ($\\nu_\\tau + \\bar{\\nu}_\\tau$)")
    ]
    scatters = ["QEL", "RES", "DIS", "COH", "MEC"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for flav_key, flav_title in flavors:
        plt.figure(figsize=(10, 6))
        plot_data, plot_weights, plot_labels = [], [], []
        for st in scatters:
            mask_current = m[flav_key] & m[st]
            if np.any(mask_current):
                plot_data.append(Ev[mask_current])
                plot_weights.append(w[mask_current])
                plot_labels.append(st)

        if plot_data:
            plt.hist(plot_data, bins=bins_E, weights=plot_weights, stacked=False,
                     histtype='step', linewidth=2.5,
                     label=plot_labels, color=colors[:len(plot_labels)], alpha=0.9)
        
        plt.xscale("log")
        plt.yscale("log") 
        plt.xlabel("True Neutrino Energy $E_{\\nu}$ [GeV]", fontsize=12)
        plt.ylabel("Expected Events", fontsize=12)
        plt.title(f"Scattering Channels for {flav_title}", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/flavor_{flav_key}_scattering.png", dpi=300)
        plt.close()

    plt.figure(figsize=(10, 6))
    m_cce   = m["CC"] & m["νe_all"]
    m_ccmu  = m["CC"] & m["νμ_all"]
    m_cctau = m["CC"] & m["ντ_all"]
    m_nc    = m["NC"]
    int_labels = ["CC $e$", "CC $\\mu$", "CC $\\tau$", "NC (All)"]
    int_masks  = [m_cce, m_ccmu, m_cctau, m_nc]
    
    for mask, label in zip(int_masks, int_labels):
        if np.any(mask):
            plt.hist(Ev[mask], bins=bins_E, weights=w[mask], histtype='step', 
                     linewidth=2.5, label=label, stacked=False)

    plt.xscale("log")
    plt.yscale("log") 
    plt.xlabel("True Neutrino Energy $E_{\\nu}$ [GeV]", fontsize=12)
    plt.ylabel("Expected Events", fontsize=12)
    plt.title("Interaction Channels (Neutrino + Antineutrino)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/interaction_channels_cc_nc.png", dpi=300)
    plt.close()

    if "Q2" in acc_data:
        plt.figure(figsize=(10, 6))
        valid_Q2 = acc_data["Q2"][acc_data["Q2"] > 0]
        min_Q2 = np.min(valid_Q2) if len(valid_Q2) > 0 else 1e-2
        max_Q2 = np.max(acc_data["Q2"]) if len(acc_data["Q2"]) > 0 else 1e3
        bins_Q2 = np.logspace(np.log10(min_Q2), np.log10(max_Q2), 40)
        
        plt.hist([acc_data["Q2"][m["CC"]], acc_data["Q2"][m["NC"]]], 
                 bins=bins_Q2, weights=[w[m["CC"]], w[m["NC"]]], 
                 stacked=False, histtype='step', linewidth=2.5,
                 label=["CC", "NC"], color=["#3498db", "#e74c3c"], alpha=0.9)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Momentum Transfer $Q^2$ [GeV$^2$]", fontsize=12)
        plt.ylabel("Expected Events", fontsize=12)
        plt.title("$Q^2$ Distribution", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, which="both", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/kinematics_Q2_distribution.png", dpi=300)
        plt.close()

    if "y" in acc_data:
        plt.figure(figsize=(10, 6))
        bins_y = np.linspace(0, 1, 40)
        plt.hist(acc_data["y"][m["DIS"]], bins=bins_y, weights=w[m["DIS"]], histtype='step', linewidth=2.5, label="DIS")
        plt.hist(acc_data["y"][m["QEL"]], bins=bins_y, weights=w[m["QEL"]], histtype='step', linewidth=2.5, label="QEL")
        plt.xlabel("Inelasticity $y$", fontsize=12)
        plt.ylabel("Expected Events", fontsize=12)
        plt.yscale("log")
        plt.title("Inelasticity ($y$) Distribution", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/kinematics_y_distribution.png", dpi=300)
        plt.close()

def analyze_file(path, accumulator):
    try:
        f = uproot.open(path)
    except Exception as e:
        print(f"  [HATA] Dosya açılamadı: {path} | Hata: {e}")
        return False

    if "gst" not in f:
        print(f"  [HATA] 'gst' tree bulunamadı: {path}")
        return False

    tree = f["gst"]
    n_entries = tree.num_entries
    
    print(f"  [OKUNDU] {os.path.basename(path)} ({n_entries:,} entries)")

    b = load_branches(tree)
    masks = make_masks(b)
    accumulator.append(b, masks)
    f.close()
    
    return True

def main():
    parser = argparse.ArgumentParser(description="GENIE gst.root analyzer & plotter")
    parser.add_argument("--input", "-i", required=True, help="gst.root dosya yolu veya glob pattern")
    parser.add_argument("--max-files", type=int, default=None, help="İşlenecek max dosya sayısı")
    parser.add_argument("--out-dir", type=str, default="plots", help="Grafiklerin ve tablonun kaydedileceği klasör")
    
    # YENİ EKLENEN LUMINOSITY PARAMETRELERİ
    parser.add_argument("--target-lumi", type=float, default=None, 
                        help="Hedeflenen luminosity (fb^-1 cinsinden). Örn: 40 ab^-1 için 40000. Belirtilmezse okunan dosya kadar kalır.")
    parser.add_argument("--lumi-per-file", type=float, default=100, 
                        help="Her bir ROOT dosyasının temsil ettiği simüle edilmiş luminosity (fb^-1). Varsayılan: 100.0")
    
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input))
    if not paths:
        paths = [args.input]
    if args.max_files:
        paths = paths[:args.max_files]

    print(f"\nBulunan toplam dosya sayısı: {len(paths)}\n")
    
    acc = PlotDataAccumulator()
    processed_count = 0
    
    for path in paths:
        if analyze_file(path, acc):
            processed_count += 1

    if processed_count > 0:
        final_data = acc.concatenate()
        
        # --- LUMINOSITY ÖLÇEKLENDİRME İŞLEMİ ---
        simulated_lumi = processed_count * args.lumi_per_file
        
        if args.target_lumi is not None:
            target_lumi = args.target_lumi
            scale_factor = target_lumi / simulated_lumi
        else:
            target_lumi = simulated_lumi
            scale_factor = 1.0

        print(f"\n[LUMINOSITY KONTROLÜ]")
        print(f"  Okunan Dosya Sayısı : {processed_count}")
        print(f"  Simüle Edilen Lumi  : {simulated_lumi:.1f} fb^-1")
        print(f"  Hedeflenen Lumi     : {target_lumi:.1f} fb^-1")
        print(f"  Ağırlık Çarpanı     : {scale_factor:.4f}")

        # Tüm ağırlıkları ("wght") hedef luminosity'e göre ölçeklendir
        final_data["wght"] = final_data["wght"] * scale_factor
        
        # 1. TOPLU TABLOYU YAZDIR VE CSV OLARAK KAYDET (Güncellenmiş ağırlıklarla)
        save_and_print_summary_table(final_data, target_lumi, args.out_dir)
        
        # 2. GRAFİKLERİ ÇİZ (Güncellenmiş ağırlıklarla)
        draw_all_plots(final_data, args.out_dir)

    print(f"\nTamamlandı. Toplam {processed_count} dosya işlendi.")

if __name__ == "__main__":
    main()