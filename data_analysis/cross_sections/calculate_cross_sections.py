"""
=============================================================
GENIE Nötrino Tesir Kesit Analizi  –  SND@LHC / Tungsten
=============================================================
Birim sistemi:
  GENIE xsec  →  GeV^-2  (natural units)
  Dönüşüm     →  × ħc² = 3.8938e-28 cm²
  Grafiklerde →  cm² / nucleon  veya  cm² / GeV / nucleon

N(etkileşim) = Φ × σ_nucleus × N_target
  Φ           : nötrino akısı [cm^-2]
  σ_nucleus   : bu dosyadan okunur, toplam hedef çekirdeği için [cm²]
  N_target    : hedef çekirdek sayısı = (M_detector / m_atom)
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# KULLANICI PARAMETRELER  ←  BURADAN DEĞİŞTİR
# ─────────────────────────────────────────────────────────────
XML_FILE = "genie_splines_GENIE_v32_SNDG18_02a_00_000.xml"

# Hedef çekirdek PDG kodu → 100ZZZAAA0
TARGET = {
    "pdg"    : "1000741840",   # ← SND@LHC Tungsten (W-184)
    "Z"      : 74,
    "A"      : 184,
    "symbol" : "W-184",
}
"""TARGET = {
    "pdg"    : "1000060120",   # ← Karbon-12 (C-12) PDG Kodu
    "Z"      : 6,              # Proton sayısı
    "A"      : 12,             # Toplam nükleon (Proton + Nötron) sayısı
    "symbol" : "C-12",
}"""

# Nötrino flavoru PDG kodu
# 14 / -14 : nu_mu / nubar_mu
# 12 / -12 : nu_e  / nubar_e
# 16 / -16 : nu_tau/ nubar_tau
NU_PDG = 14     # ← Değiştirebilirsin

# Enerji aralığı [GeV]
E_MIN, E_MAX = 0.1, 1000.0
N_POINTS     = 1000
# ─────────────────────────────────────────────────────────────

import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re, argparse

# Komut satırından parametre değiştirebilmek için
parser = argparse.ArgumentParser()
parser.add_argument('--tgt', default=None, help="Target PDG Code")
parser.add_argument('--nu',  default=None, type=int, help="Neutrino PDG Code")
parser.add_argument('--xml', default=None, help="GENIE XML File")
args = parser.parse_args()
if args.tgt: TARGET["pdg"] = args.tgt
if args.nu:  NU_PDG = args.nu
if args.xml: XML_FILE = args.xml

# ─── Sabitler ────────────────────────────────────────────────
HBARC2   = 3.8938e-28   # cm² / GeV^-2
E_GRID   = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_POINTS)
SKIP_KW  = ('NuEEL', 'IMD', 'IMDAnh', 'DFR') # Analiz dışı ufak etkileşimler
MODES    = ['QES', 'MEC', 'RES', 'DIS', 'COH']

NU_LABELS = {12:'ν_e', -12:'ν̄_e', 14:'ν_μ', -14:'ν̄_μ', 16:'ν_τ', -16:'ν̄_τ'}
NU_TEX    = {12:r'$\nu_e$', -12:r'$\bar{\nu}_e$',
             14:r'$\nu_\mu$', -14:r'$\bar{\nu}_\mu$',
             16:r'$\nu_\tau$', -16:r'$\bar{\nu}_\tau$'}

plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':11,
    'axes.labelsize':13, 'axes.titlesize':13,
    'legend.fontsize':10, 'lines.linewidth':2.0,
    'figure.dpi':150,
})

MODE_STYLE = {
    'QES': dict(color='#1f77b4', lw=2.2, ls='-',  label='CCQE'),
    'MEC': dict(color='#ff7f0e', lw=1.8, ls='--', label='MEC (2p2h)'),
    'RES': dict(color='#2ca02c', lw=1.8, ls='-',  label='RES'),
    'DIS': dict(color='#d62728', lw=2.2, ls='-',  label='DIS'),
    'COH': dict(color='#9467bd', lw=1.4, ls=':',  label='COH'),
}

# ─── Veri Okuma Fonksiyonları ────────────────────────────────
def spline_to_grid(sp_elem):
    """Bir GENIE spline'ını okur ve ortak E_GRID'e log-log interpole eder."""
    pairs = []
    for k in sp_elem.findall('knot'):
        try:
            E  = float(k.find('E').text)
            xs = float(k.find('xsec').text)
            if E > 0 and xs > 0:
                pairs.append((E, xs))
        except (TypeError, ValueError):
            continue
    if len(pairs) < 4:
        return None
        
    pairs.sort()
    Es  = np.array([p[0] for p in pairs])
    XSs = np.array([p[1] for p in pairs])
    
    f = interp1d(np.log10(Es), np.log10(XSs), kind='linear', bounds_error=False, fill_value=np.nan)
    log_xs = f(np.log10(E_GRID))
    return np.where(np.isfinite(log_xs), 10**log_xs, 0.0)

def load_data(xml_file, tgt_pdg, nu_pdg):
    """Belirtilen hedef ve nötrino için tüm spline'ları okur ve toplar."""
    try:
        tree = ET.parse(xml_file)
    except FileNotFoundError:
        print(f"HATA: {xml_file} dosyası bulunamadı!")
        exit(1)
        
    root = tree.getroot()
    acc = defaultdict(lambda: np.zeros_like(E_GRID))
    n_read = 0

    for sp in root.findall('.//spline'):
        name = sp.get('name', '')
        if f'tgt:{tgt_pdg}' not in name: continue
        if f'nu:{nu_pdg};' not in name: continue
        if any(kw in name for kw in SKIP_KW): continue

        pr_m = re.search(r'proc:Weak\[(\w+)\],(\w+)', name)
        if not pr_m: continue
        
        itype = pr_m.group(1)   # CC veya NC
        mode  = pr_m.group(2)   # QES, DIS, RES vb.

        if itype not in ('CC', 'NC'): continue
        if mode not in MODES: continue

        arr = spline_to_grid(sp)
        if arr is None: continue

        # GeV^-2 -> cm^2 dönüşümü ve toplama işlemi (charm vs. hepsi buraya eklenir)
        acc[(itype, mode)] += arr * HBARC2
        n_read += 1

    print(f"  Okunan ve birleştirilen spline sayısı: {n_read}")
    return dict(acc)

# ─── Veriyi Yükleme ve Tablo ─────────────────────────────────
print(f"\nYükleniyor: {TARGET['symbol']} Hedefi, {NU_LABELS.get(NU_PDG,'?')} Nötrinosu")
D = load_data(XML_FILE, TARGET['pdg'], NU_PDG)
A = TARGET['A']
nu_tex = NU_TEX.get(NU_PDG, f'ν({NU_PDG})')
tgt_str = TARGET['symbol']

print(f"\n{'='*65}")
print(f"  σ/E  [{nu_tex} CC,  {tgt_str},  per nucleon,  ×10⁻³⁸ cm²/GeV/nucleon]")
print(f"{'─'*65}")
print(f"  {'E [GeV]':>8}  {'QES':>8}  {'MEC':>8}  {'RES':>8}  {'DIS':>8}  {'Total':>8}")
print(f"{'─'*65}")
for E_ch in [1.0, 5.0, 10.0, 50.0, 100.0, 300.0, 1000.0]:
    idx = np.argmin(np.abs(E_GRID - E_ch))
    row = []
    tot = 0.0
    for mode in MODES[:4]:
        v = D.get(('CC', mode), np.zeros_like(E_GRID))[idx]
        row.append(v / E_ch / A * 1e38)
        tot += v
    print(f"  {E_ch:>8.1f}  " + "  ".join(f"{r:>8.4f}" for r in row) + f"  {tot/E_ch/A*1e38:>8.4f}")
print(f"{'='*65}\n")


# ════════════════════════════════════════════════════════════════
# PLOT 1 – σ/E vs E   (PDG Karşılaştırmalı Ana Deneysel Grafik)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle(
    rf'$\sigma/E_\nu$ — {nu_tex} CC, Target: {tgt_str}'
    '\nGENIE SNDG18_02a vs PDG World Average',
    fontsize=14
)

cc_total = np.zeros_like(E_GRID)
for mode in MODES:
    xs = D.get(('CC', mode), None)
    if xs is None: continue
    xs_nn = np.nan_to_num(xs)
    ratio = np.where(E_GRID > 0, xs_nn / E_GRID / A * 1e38, np.nan)
    valid = np.isfinite(ratio) & (ratio > 0)
    if valid.sum() < 5: continue
    ax.plot(E_GRID[valid], ratio[valid], **MODE_STYLE[mode])
    cc_total += xs_nn

# Total CC
ratio_tot = np.where(E_GRID > 0, cc_total / E_GRID / A * 1e38, np.nan)
v = np.isfinite(ratio_tot) & (ratio_tot > 0)
ax.plot(E_GRID[v], ratio_tot[v], 'k-', lw=3.0, label=f'GENIE Total CC ({tgt_str})', zorder=10)

# PDG WORLD AVERAGE (Isoscalar)
pdg_val = 0.677 if NU_PDG > 0 else 0.334
pdg_err = 0.014 if NU_PDG > 0 else 0.008
pdg_E = np.array([0.1, 350.0])
pdg_Y = np.array([pdg_val, pdg_val])

ax.plot(pdg_E, pdg_Y, color='darkred', lw=2.5, ls='-.', label=f'PDG Isoscalar Average ({pdg_val:.3f})', zorder=5)
ax.fill_between(pdg_E, pdg_Y - pdg_err, pdg_Y + pdg_err, color='darkred', alpha=0.2, zorder=4)

if TARGET['A'] == 184:
    ax.text(30, pdg_val + (0.08 if NU_PDG > 0 else -0.08), 
            'Note: W-184 has neutron excess.\nExpected to deviate from Isoscalar PDG.', 
            fontsize=10, color='darkred', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Deney bantları

ax.set_xscale('log')
ax.set_xlim(E_MIN, 1000)
ax.set_ylim(0, 1.2 if NU_PDG > 0 else 0.6)
ax.set_xlabel(r'$E_\nu$ [GeV]', fontsize=13)
ax.set_ylabel(r'$\sigma/E_\nu\;[\times10^{-38}$ cm$^2$/GeV/nucleon$]$', fontsize=12)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower right' if NU_PDG > 0 else 'upper right', framealpha=0.95, ncol=2, fontsize=10)
ax.grid(True, which='major', ls='-', alpha=0.5)
ax.grid(True, which='minor', ls='--', alpha=0.2)

out1 = f'xsec_sigma_over_E_PDG_compare_{tgt_str.replace("-","")}_nu{NU_PDG}.png'
fig.tight_layout()
fig.savefig(out1, dpi=300, bbox_inches='tight')
print(f"✓  {out1}")
plt.close(fig)



# ════════════════════════════════════════════════════════════════
# PLOT 3 – σ (log-log) CC + NC tam panel
# ════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 2, figsize=(15, 11))
fig3.suptitle(
    rf'{nu_tex} / $\bar{{{nu_tex[1:-1]}}}$ Cross Sections — CC & NC'
    f'\n{tgt_str}, GENIE, per nucleon',
    fontsize=14
)

for (ax, nu_s, itype, title) in [
    (axes3[0,0],  NU_PDG, 'CC', f'{NU_TEX.get(NU_PDG,"")} CC'),
    (axes3[0,1], -NU_PDG, 'CC', f'{NU_TEX.get(-NU_PDG,"")} CC'),
    (axes3[1,0],  NU_PDG, 'NC', f'{NU_TEX.get(NU_PDG,"")} NC'),
    (axes3[1,1], -NU_PDG, 'NC', f'{NU_TEX.get(-NU_PDG,"")} NC'),
]:
    Dtmp = load_data(XML_FILE, TARGET['pdg'], nu_s)
    total = np.zeros_like(E_GRID)
    for mode in MODES:
        xs = np.nan_to_num(Dtmp.get((itype, mode), np.zeros_like(E_GRID)))
        vals = xs / A * 1e38
        valid = vals > 0
        if valid.sum() < 5: continue
        ax.plot(E_GRID[valid], vals[valid], **MODE_STYLE[mode])
        total += xs
        
    tv = total > 0
    if tv.sum() > 5:
        ax.plot(E_GRID[tv], total[tv]/A*1e38, 'k--', lw=2.8, label='Total', zorder=10)
        
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(E_MIN, 1000)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(r'$E_\nu$ [GeV]')
    ax.set_ylabel(r'$\sigma\;[\times10^{-38}$ cm$^2$/nucleon$]$')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.grid(True, which='both', ls='--', alpha=0.3)

fig3.tight_layout()
out3 = f'xsec_CC_NC_{tgt_str.replace("-","")}_nu{abs(NU_PDG)}.png'
fig3.savefig(out3, dpi=300, bbox_inches='tight')
print(f"✓  {out3}")
plt.close(fig3)

print("\nTÜM GRAFİKLER BAŞARIYLA ÜRETİLDİ!")