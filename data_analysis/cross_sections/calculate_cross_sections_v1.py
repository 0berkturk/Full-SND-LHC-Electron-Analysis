"""
=============================================================
GENIE Nötrino Tesir Kesit Analizi  –  SND@LHC / Tungsten
=============================================================
Birim sistemi:
  GENIE xsec  →  GeV^-2  (natural units)
  Dönüşüm     →  × ħc² = 3.8938e-28 cm²
  Grafiklerde →  cm² / nucleon  veya  cm² / GeV / nucleon

N(etkileşim) = Φ × σ_nucleus × N_target
  Φ           : nötrino akısı [cm^-2]  (luminosity * xsec_pp / acceptance...)
  σ_nucleus   : bu dosyadan okunur, toplam hedef çekirdeği için [cm²]
  N_target    : hedef çekirdek sayısı = (M_detector / m_atom)

Kullanım:
  python genie_snd_xsec.py                  # varsayılan: W184, nu_mu
  python genie_snd_xsec.py --tgt 1000060120 # Carbon-12
  python genie_snd_xsec.py --nu 16          # nu_tau
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# KULLANICI PARAMETRELER  ←  BURADAN DEĞİŞTİR
# ─────────────────────────────────────────────────────────────
XML_FILE = "genie_splines_GENIE_v32_SNDG18_02a_00_000.xml"

# Hedef çekirdek PDG kodu → 100ZZZAAA0
# W-184  : 1000741840   (SND@LHC tungsten)
# C-12   : 1000060120
# Fe-56  : 1000260560
# Pb-208 : yok bu dosyada
TARGET = {
    "pdg"    : "1000741840",   # ← değiştir
    "Z"      : 74,
    "A"      : 184,
    "symbol" : "W-184",
}

# Nötrino flavoru PDG kodu
# 12 / -12 : nu_e  / nubar_e
# 14 / -14 : nu_mu / nubar_mu
# 16 / -16 : nu_tau/ nubar_tau
NU_PDG = 14     # ← değiştir

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
import re, os, sys, argparse

# Komut satırı override
parser = argparse.ArgumentParser()
parser.add_argument('--tgt', default=None)
parser.add_argument('--nu',  default=None, type=int)
parser.add_argument('--xml', default=None)
args = parser.parse_args()
if args.tgt: TARGET["pdg"] = args.tgt
if args.nu:  NU_PDG = args.nu
if args.xml: XML_FILE = args.xml

# ─── Sabitler ────────────────────────────────────────────────
HBARC2   = 3.8938e-28   # cm² / GeV^-2  (GENIE iç birimi → cm²)
E_GRID   = np.logspace(np.log10(E_MIN), np.log10(E_MAX), N_POINTS)
SKIP_KW  = ('NuEEL', 'IMD', 'IMDAnh', 'DFR', 'charm:')
MODES    = ['QES', 'MEC', 'RES', 'DIS', 'COH']

# Flavor isimleri
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

# ─── Veri Okuma ──────────────────────────────────────────────
def spline_to_grid(sp_elem):
    """Bir GENIE spline elementini E_GRID'e interpolasyonla çevirir.
    Çıktı birimi: GeV^-2  (ham, henüz cm²'ye çevrilmemiş)
    """
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
    # Log-log interpolasyon (parton xsec için kararlı)
    f = interp1d(np.log10(Es), np.log10(XSs),
                 kind='linear', bounds_error=False, fill_value=np.nan)
    log_xs = f(np.log10(E_GRID))
    return np.where(np.isfinite(log_xs), 10**log_xs, 0.0)


def load_data(xml_file, tgt_pdg, nu_pdg):
    """
    Belirtilen hedef + nötrino flavoru için tüm spline'ları oku.
    Döndürür: dict[(itype, mode)] = σ_nucleus(E) [cm²]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    acc = defaultdict(lambda: np.zeros_like(E_GRID))
    n_read = 0

    for sp in root.findall('.//spline'):
        name = sp.get('name', '')
        if f'tgt:{tgt_pdg}' not in name:
            continue
        if f'nu:{nu_pdg};' not in name:
            continue
        if any(kw in name for kw in SKIP_KW):
            continue

        pr_m = re.search(r'proc:Weak\[(\w+)\],(\w+)', name)
        if not pr_m:
            continue
        itype = pr_m.group(1)   # CC / NC
        mode  = pr_m.group(2)   # QES / MEC / RES / DIS / COH

        if itype not in ('CC', 'NC'):
            continue
        if mode not in MODES:
            continue

        arr = spline_to_grid(sp)
        if arr is None:
            continue

        # GeV^-2 → cm²
        acc[(itype, mode)] += arr * HBARC2
        n_read += 1

    print(f"  Okunan spline sayısı: {n_read}")
    return dict(acc)


def get_xs(data, itypes, modes):
    """Seçilen itype ve mode kombinasyonlarının toplam σ(E) [cm²]"""
    out = np.zeros_like(E_GRID)
    for it in itypes:
        for m in modes:
            if (it, m) in data:
                out += data[(it, m)]
    out[out <= 0] = np.nan
    return out


# ─── Yükle ───────────────────────────────────────────────────
print(f"\nYükleniyor: {TARGET['symbol']}, {NU_LABELS.get(NU_PDG,'?')}")
D = load_data(XML_FILE, TARGET['pdg'], NU_PDG)
A = TARGET['A']
nu_tex = NU_TEX.get(NU_PDG, f'ν({NU_PDG})')
tgt_str = TARGET['symbol']

# ─── Özet Tablo ──────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  σ/E  [{nu_tex} CC,  {tgt_str},  per nucleon,  ×10⁻³⁸ cm²/GeV/nucleon]")
print(f"{'─'*65}")
print(f"  {'E [GeV]':>8}  {'QES':>8}  {'MEC':>8}  {'RES':>8}  {'DIS':>8}  {'Total':>8}")
print(f"{'─'*65}")
for E_ch in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0]:
    idx = np.argmin(np.abs(E_GRID - E_ch))
    row = []
    tot = 0.0
    for mode in MODES[:4]:  # QES MEC RES DIS
        v = D.get(('CC', mode), np.zeros_like(E_GRID))[idx]
        row.append(v / E_ch / A * 1e38)
        tot += v
    print(f"  {E_ch:>8.1f}  " + "  ".join(f"{r:>8.4f}" for r in row) +
          f"  {tot/E_ch/A*1e38:>8.4f}")
print(f"{'─'*65}")
print(f"  PDG referans (DIS-baskın, yüksek E): ~0.67 ×10⁻³⁸ cm²/GeV/nucleon")
print(f"{'='*65}\n")


# ════════════════════════════════════════════════════════════════
# PLOT 1 – σ/E vs E   (ana deneysel grafik)
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(
    rf'$\sigma/E_\nu$ — {nu_tex} CC, Hedef: {tgt_str}'
    '\nGENIE SNDG18_02a, per nucleon',
    fontsize=13
)

cc_total = np.zeros_like(E_GRID)
for mode in MODES:
    xs = D.get(('CC', mode), None)
    if xs is None:
        continue
    xs_nn = np.nan_to_num(xs)
    ratio = np.where(E_GRID > 0, xs_nn / E_GRID / A * 1e38, np.nan)
    valid = np.isfinite(ratio) & (ratio > 0)
    if valid.sum() < 5:
        continue
    ax.plot(E_GRID[valid], ratio[valid], **MODE_STYLE[mode])
    cc_total += xs_nn

# Total CC
ratio_tot = np.where(E_GRID > 0, cc_total / E_GRID / A * 1e38, np.nan)
v = np.isfinite(ratio_tot) & (ratio_tot > 0)
ax.plot(E_GRID[v], ratio_tot[v], 'k--', lw=2.8, label='Total CC', zorder=10)

# Deney bantları
ax.axvspan(0.2,  2.0,  alpha=0.09, color='gold',      label='T2K / NOvA (~0.6 GeV)')
ax.axvspan(1.0,  10.0, alpha=0.08, color='limegreen',  label='DUNE (1-5 GeV)')
ax.axvspan(10.0, 300., alpha=0.07, color='skyblue',    label='SND@LHC (10-300 GeV)')

ax.set_xscale('log')
ax.set_xlim(E_MIN, 500)
ax.set_ylim(0, 1.6)
ax.set_xlabel(r'$E_\nu$ [GeV]', fontsize=13)
ax.set_ylabel(r'$\sigma/E_\nu\;[\times10^{-38}$ cm$^2$/GeV/nucleon$]$', fontsize=12)
ax.legend(loc='upper right', framealpha=0.92, ncol=2)
ax.grid(True, which='both', ls='--', alpha=0.3)

out1 = f'xsec_sigma_over_E_{tgt_str.replace("-","")}_nu{NU_PDG}.png'
fig.tight_layout()
fig.savefig(out1, dpi=200, bbox_inches='tight')
print(f"✓  {out1}")
plt.close(fig)


# ════════════════════════════════════════════════════════════════
# PLOT 2 – QES vs DIS yakın plan + geçiş noktası
# ════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.suptitle(
    rf'CCQE vs DIS — $\sigma/E_\nu$, {nu_tex} + $\bar{{\nu}}$, {tgt_str}'
    '\nGENIE SNDG18_02a, per nucleon',
    fontsize=13
)

for nu_s, ls in [(NU_PDG, '-'), (-NU_PDG, '--')]:
    Dtmp = load_data(XML_FILE, TARGET['pdg'], nu_s)
    nu_lbl = NU_TEX.get(nu_s, str(nu_s))
    for mode, color in [('QES','#1f77b4'), ('DIS','#d62728')]:
        xs = np.nan_to_num(Dtmp.get(('CC', mode), np.zeros_like(E_GRID)))
        ratio = np.where(E_GRID > 0, xs / E_GRID / A * 1e38, np.nan)
        valid = np.isfinite(ratio) & (ratio > 0)
        if valid.sum() < 5:
            continue
        ax2.plot(E_GRID[valid], ratio[valid], color=color, ls=ls, lw=2.0,
                 label=f'{MODE_STYLE[mode]["label"]} {nu_lbl}')

# QES = DIS geçiş noktası (nu sadece)
qes_arr = np.nan_to_num(D.get(('CC','QES'), np.zeros_like(E_GRID)))
dis_arr = np.nan_to_num(D.get(('CC','DIS'), np.zeros_like(E_GRID)))
diff    = qes_arr - dis_arr
cross   = np.where(np.diff(np.sign(diff)))[0]
if len(cross):
    i = cross[0]
    E_c = E_GRID[i]
    y_c = qes_arr[i] / E_c / A * 1e38
    ax2.axvline(E_c, color='gray', ls=':', lw=1.5)
    ax2.annotate(f'QES=DIS\n≈{E_c:.1f} GeV',
                 xy=(E_c, y_c), xytext=(E_c*1.5, y_c + 0.1),
                 fontsize=9, color='gray',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1))

ax2.axvspan(10.0, 300., alpha=0.08, color='skyblue', label='SND@LHC')
ax2.set_xscale('log')
ax2.set_xlim(E_MIN, 500)
ax2.set_ylim(0, 1.4)
ax2.set_xlabel(r'$E_\nu$ [GeV]', fontsize=13)
ax2.set_ylabel(r'$\sigma/E_\nu\;[\times10^{-38}$ cm$^2$/GeV/nucleon$]$', fontsize=12)
ax2.legend(loc='upper right', framealpha=0.92, ncol=2, fontsize=9)
ax2.grid(True, which='both', ls='--', alpha=0.3)

out2 = f'xsec_QES_vs_DIS_{tgt_str.replace("-","")}_nu{abs(NU_PDG)}.png'
fig2.tight_layout()
fig2.savefig(out2, dpi=200, bbox_inches='tight')
print(f"✓  {out2}")
plt.close(fig2)


# ════════════════════════════════════════════════════════════════
# PLOT 3 – σ (log-log) CC + NC tam panel
# ════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 2, figsize=(15, 11))
fig3.suptitle(
    rf'{nu_tex} / $\bar{{{nu_tex[1:-1]}}}$ Tesir Kesitleri — CC & NC'
    f'\n{tgt_str}, GENIE SNDG18_02a, per nucleon',
    fontsize=13
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
        if valid.sum() < 5:
            continue
        ax.plot(E_GRID[valid], vals[valid], **MODE_STYLE[mode])
        total += xs
    tv = total > 0
    if tv.sum() > 5:
        ax.plot(E_GRID[tv], total[tv]/A*1e38, 'k--', lw=2.8, label='Total', zorder=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(E_MIN, 1000)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r'$E_\nu$ [GeV]')
    ax.set_ylabel(r'$\sigma\;[\times10^{-38}$ cm$^2$/nucleon$]$')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.grid(True, which='both', ls='--', alpha=0.3)

fig3.tight_layout()
out3 = f'xsec_CC_NC_{tgt_str.replace("-","")}_nu{abs(NU_PDG)}.png'
fig3.savefig(out3, dpi=200, bbox_inches='tight')
print(f"✓  {out3}")
plt.close(fig3)

print(f"\n{'='*50}")
print("  N(etkileşim) = Φ × σ_nucleus × N_target")
print(f"  σ_nucleus(E) = σ_per_nucleon(E) × A = σ_per_nucleon × {A}")
print(f"  Bu grafiklerdeki σ: per nucleon [×10⁻³⁸ cm²]")
print(f"  cm² almak için: okunan değer × 10⁻³⁸")
print("  N_target = (dedektör kütlesi [g]) / ({:.1f} × 1.6605e-24 g)".format(A))
print(f"{'='*50}\n")