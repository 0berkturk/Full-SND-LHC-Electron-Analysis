import os
import glob
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
# ==========================================
# 1. Plotting Utilities 
# ==========================================
cmap = plt.get_cmap('plasma')
cmap.set_under('white')

def plot_multiple_hist(qdc_energy_list,N, x_label, title, label_str, outdir,alpha_list,name="EMPTY"):
    for i in range(len(qdc_energy_list)):
        qdc_energy=qdc_energy_list[i]
        plt.hist(qdc_energy,bins=N,label=label_str[i],alpha=alpha_list[i],histtype='step' )    
        # Log scale
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel('Number of Events')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    # Save
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/{name}.png", dpi=300)
    plt.close()

def plot_2d_hist(x_data, y_data, bins_x=50, bins_y=50, out_name="2d_hist", xlabel="X", ylabel="Y", title="2D", outdir="plots"):
    if x_data is None or y_data is None or len(x_data) == 0:
        return
    plt.figure(figsize=(8,6))
    hist = plt.hist2d(x_data.numpy(), y_data.numpy(), bins=[bins_x, bins_y], cmap=cmap, alpha=0.8, vmin=1)  
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label='Events')
    plt.grid(True, linestyle='--', alpha=0.5)
    if not os.path.exists(outdir): os.makedirs(outdir)
    plt.savefig(os.path.join(outdir, f"{out_name}.png"), dpi=300)
    plt.clf()
    plt.close()

def plot_1dhist_overlay(data_list, label_list, title, xlabel, ylabel, scale, x_min, x_max, outdir="plots", out_name="1d_hist"):
    plt.figure(figsize=(8,6))
    for data, label in zip(data_list, label_list):
        if data is not None and len(data) > 0:
            plt.hist(data.numpy(), bins=50, range=(x_min, x_max), histtype='step', linewidth=1.5, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    if scale == 'log': plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    if not os.path.exists(outdir): os.makedirs(outdir)
    plt.savefig(os.path.join(outdir, f"{out_name}.png"), dpi=300)
    plt.clf()
    plt.close()

# ==========================================
# 2. Cut & Dosya Yükleme Fonksiyonları
# ==========================================
# ==========================================
# 2. Cut & Dosya Yükleme Fonksiyonları
# ==========================================
def generate_cut_names(dict_cuts):
    cut_names = []
    for tb_recal in dict_cuts.get("TB_RECALIBRATION_S2Y", [False]):
        for t_win in dict_cuts.get("t_window_data", [(0.5,2.3)]):
            for qdc in dict_cuts.get("qdc_threshold_value_scifi_data", [0]):
                cut_names.append(f"S2Ycal{tb_recal}_qdcD{qdc}_twinD{t_win[0]}_{t_win[1]}")
    return cut_names

def get_unique_id(data_dict):
    """run_id ve event_number kullanarak event bazlı benzersiz bir kimlik (ID) tensörü üretir."""
    if "run_id" in data_dict and "event_number" in data_dict:
        # Taşmaları (overflow) önlemek için int64 (long) tipine çeviriyoruz
        r_id = torch.as_tensor(data_dict["run_id"], dtype=torch.long)
        e_num = torch.as_tensor(data_dict["event_number"], dtype=torch.long)
        return (r_id * 1000000000) + e_num
    else:
        raise KeyError("Sözlükte 'run_id' veya 'event_number' bulunamadı!")




def load_all_data(file_id, dl_processed_dir, extracted_props_dir, prefixes):
    """4. Dosyayı ve 3 DL model dosyasını aynı anda yükler ve event-by-event doğrular."""
    basename = f"batched_data_{year}_{file_id}"
    
    # 1. 4. DOSYAYI YÜKLE (Ana Özellikler ve Cut Kaynağı)
    prop_path = os.path.join(extracted_props_dir, f"{basename}_extracted.pt")
    if not os.path.exists(prop_path):
        print(f"    [Hata] 4. Dosya bulunamadı: {prop_path}")
        return None
    extracted_props = torch.load(prop_path, map_location='cpu', weights_only=False)
    """for k in extracted_props:
        print(k)"""

    # Base dosyanın run_id ve event_number bilgilerini kontrol et ve tensöre çevir
    if "run_id" not in extracted_props or "event_number" not in extracted_props:
        print(f"    [Hata] 4. Dosyada run_id veya event_number eksik! Atlanıyor: {basename}")
        return None
        
    base_run_id = torch.as_tensor(extracted_props["run_id"])
    base_event_num = torch.as_tensor(extracted_props["event_number"])
    base_len = len(base_run_id)

    # Ekstra özelliklerin hesaplanması
    if "us_notime_hits_per_layer" in extracted_props:
        extracted_props["us_active_layers_notime_hits0"] = torch.sum(extracted_props["us_notime_hits_per_layer"] > 0, dim=1)
    if "us_notime_qdc_per_layer" in extracted_props:
        extracted_props["us_active_layers_notime_qdc0"] = torch.sum(extracted_props["us_notime_qdc_per_layer"] > 0, dim=1)
    if "scifi_notime_hits_per_layer" in extracted_props:
        extracted_props["scifi_active_layers_notime_hitsN"] = torch.sum(extracted_props["scifi_notime_hits_per_layer"] > layerscifihitnumber, dim=1)
        extracted_props["scifi_active_layers_nocuts"] = torch.sum(extracted_props["scifi_notime_hits_per_layer"] > 0 , dim=1)
    

    extracted_props["radian_neutrino"] = tan_2_radian(extracted_props["angle_incoming_neutrino_x"], extracted_props["angle_incoming_neutrino_y"])
    extracted_props["radian_unweighted"] = tan_2_radian(extracted_props["angle_unweighted_x"], extracted_props["angle_unweighted_y"])
    extracted_props["radian_avg_step"] = tan_2_radian(extracted_props["angle_avg_step_y"], extracted_props["angle_avg_step_x"])
    extracted_props["radian_avg_lin"] = tan_2_radian(extracted_props["angle_avg_linear_x"], extracted_props["angle_avg_linear_y"])

    extracted_props["radian_step_1"] = tan_2_radian(extracted_props["angle_step_1_x"], extracted_props["angle_step_2_y"])

    loaded_dicts = {"props_4th_file": extracted_props}
    
    # 2. DL MODELLERİNİ YÜKLE VE BİREBİR (EVENT-BY-EVENT) KONTROL ET
    for key, prefix in prefixes.items():
        search_pattern = os.path.join(dl_processed_dir, f"{basename}_{prefix}*.pt")
        found_files = glob.glob(search_pattern)
        if not found_files:
            print(f"    [Hata] DL dosyası bulunamadı: {search_pattern}")
            return None
        
        data = torch.load(found_files[0], map_location='cpu', weights_only=False)
        
        # DL dosyasında run_id ve event_number var mı?
        if "run_id" not in data or "event_number" not in data:
            print(f"    [Uyumsuzluk] {key} model dosyasında run_id veya event_number bulunamadı! Dosya atlanıyor.")
            return None
            
        dl_run_id = torch.as_tensor(data["run_id"])
        dl_event_num = torch.as_tensor(data["event_number"])
        
        # 1. Aşama Kontrol: Event sayıları (Tensör boyutları) aynı mı?
        if len(dl_run_id) != base_len:
            print(f"    [Uyumsuzluk - Boyut] 4. Dosya ({base_len} event) ile {key} ({len(dl_run_id)} event) boyutu eşleşmiyor! Atlanıyor.")
            return None
            
        # 2. Aşama Kontrol: Bütün run_id ve event_number'lar aynı sırada birebir eşleşiyor mu?
        if not torch.equal(base_run_id, dl_run_id) or not torch.equal(base_event_num, dl_event_num):
            print(f"    [Uyumsuzluk - İçerik] 4. Dosya ile {key} modelinin run_id/event_number dizilimleri birebir eşleşmiyor! Atlanıyor.")
            return None

        # Kontrollerden geçerse, zaten %100 birebir aynı dosyalardır. Kesişim (mask) işlemi yapmaya gerek yok.
        loaded_dicts[key] = data
        
    print(f"    [Başarılı] {basename} için tüm modeller {base_len} event üzerinden %100 eşleşti.")
    return loaded_dicts


def create_combined_event_cut_mask(props_4th_file, m_100gev_data, hit_cut_name, dl_100gev, dl_400gev, dl_cls, event_cut_dict, num_events):
    """Hem 4. dosyadan hem de DL modellerinin çıktılarından/özelliklerinden maske üretir."""
    mask = torch.ones(num_events, dtype=torch.bool)
    
    for mode in ["larger_than", "smaller_than"]:
        for key, threshold in event_cut_dict.get(mode, {}).items():
            
            # 1. Özel İsimlendirilmiş DL Model Çıktıları mı? (Score / Energy)
            if key == "DL_100GeV_Energy":
                tensor_to_cut = dl_100gev
                #print("DL_100GeV_Energy cut applied ")
            elif key == "DL_400GeV_Energy":
                tensor_to_cut = dl_400gev
                #print("DL_400GeV_Energy cut applied ")
            elif key == "DL_CLS_Score":
                tensor_to_cut = dl_cls
                #print("DL_CLS_Score cut applied ")
                
            # 2. DL dosyasında saklanan, sonuna {hit_cut_name} alan özellikler mi? (Örn: "Total Hits")
            elif f"{key}{hit_cut_name}" in m_100gev_data:
                tensor_to_cut = m_100gev_data[f"{key}{hit_cut_name}"]
                #print(f"{key}{hit_cut_name} cut is applied on m100gevdata")
            
            elif f"{key}" in m_100gev_data:
                tensor_to_cut = m_100gev_data[f"{key}"]
                #print(f"{key} cut is applied on m100gevdata with predetermined hit cut")
                
            # 3. 4. Dosya (Ham) özellikleri mi? (Örn: "scifi_notime_total_qdc")
            elif key in props_4th_file:
                tensor_to_cut = props_4th_file[key]
                #print(key,"cut applied on prop file")
            else:
                print(f"    [Uyarı] '{key}' özelliği hiçbir dosyada bulunamadı, cut atlandı.")
                continue
                
            # Seçilen tensöre göre maskeyi uygula
            if mode == "larger_than":
                mask &= (tensor_to_cut > threshold)
            else:
                mask &= (tensor_to_cut < threshold)
                
    return mask

def tan_2_radian(slope_x, slope_y):
    tan_theta = (slope_x**2 + slope_y**2)**0.5
    theta_radian = torch.atan(tan_theta) # np.arctan yerine
    return theta_radian

# ==========================================
# 3. Main Analysis Workflow
# ==========================================
def analyze_dl_physics(data_ids, dl_processed_dir, extracted_props_dir, hit_cuts_dict, event_cut_configs, prefixes, base_outdir="analysis_plots"):
    alreadsaved=set()
    hit_cut_names = generate_cut_names(hit_cuts_dict)
    
    # 1. TÜM DOSYALARI YÜKLE VE BİRLEŞTİR
    aggregated_data = {
        "props_4th_file": defaultdict(list),
        "m_100gev": defaultdict(list), 
        "m_400gev": defaultdict(list), 
        "m_cls": defaultdict(list)
    }
    
    print("\n--- Dosyalar Yükleniyor ve Birleştiriliyor ---")
    valid_files_count = 0
    for file_id in data_ids:
        print(f"  -> Loading File ID: {file_id}")
        model_data = load_all_data(file_id, dl_processed_dir, extracted_props_dir, prefixes)
        if not model_data:
            continue
        
        valid_files_count += 1
        for dict_key in aggregated_data.keys():
            for key, tensor in model_data[dict_key].items():
                if isinstance(tensor, torch.Tensor):
                    aggregated_data[dict_key][key].append(tensor)

    if valid_files_count == 0:
        print("Geçerli hiçbir dosya bulunamadı. Çıkılıyor.")
        return

    # Listeleri tensörlere dönüştür
    for dict_key in aggregated_data:
        for key in aggregated_data[dict_key]:
            if len(aggregated_data[dict_key][key]) > 0:
                aggregated_data[dict_key][key] = torch.cat(aggregated_data[dict_key][key], dim=0)
    
    print(f"\n--- Toplam {valid_files_count} dosyanın 4. dosyaları ve DL dosyaları birleştirildi ---")

    # 2. CUT UYGULAMA VE ÇİZİM AŞAMASI
    for event_cut_name, event_cut_dict in event_cut_configs.items():
        print(f"\n-> Applying Event Cut Scenario: {event_cut_name}")
        
        for hit_cut_name in hit_cut_names:
            if hit_cut_name not in aggregated_data["m_100gev"]:
                continue
                
            cut_outdir = os.path.join(base_outdir, event_cut_name, hit_cut_name)
            
            # --- HAM DL MODEL ÇIKTILARI ---
            dl_100gev_raw = aggregated_data["m_100gev"][hit_cut_name]
            dl_400gev_raw = aggregated_data["m_400gev"][hit_cut_name]
            dl_cls_raw    = aggregated_data["m_cls"][hit_cut_name]
            num_events = dl_100gev_raw.shape[0]

            # --- HAM DL PROPERTİLERİ (Önceden hesaplanıp m_100gev içine kaydedilmiş olanlar) ---
            total_hits_raw = aggregated_data["m_100gev"].get(f"Total Hits{hit_cut_name}")
            total_qdc_raw  = aggregated_data["m_100gev"].get(f"Total QDC{hit_cut_name}")
            frac_qdc_raw   = aggregated_data["m_100gev"].get(f"Log of Fraction Abs QDC{hit_cut_name}")
            frac_hit_raw   = aggregated_data["m_100gev"].get(f"Log of Fraction Abs Hits{hit_cut_name}")
            run_id = aggregated_data["props_4th_file"].get(f"run_id")
            event_number = aggregated_data["props_4th_file"].get(f"event_number")

            angle_incoming_neutrino_x = aggregated_data["props_4th_file"].get("angle_incoming_neutrino_x")
            angle_incoming_neutrino_y = aggregated_data["props_4th_file"].get(f"angle_incoming_neutrino_y")
            angle_unweighted_x = aggregated_data["props_4th_file"].get(f"angle_unweighted_x")
            angle_unweighted_y = aggregated_data["props_4th_file"].get(f"angle_unweighted_y")

            angle_avg_step_x = aggregated_data["props_4th_file"].get(f"angle_avg_step_x")
            angle_avg_step_y = aggregated_data["props_4th_file"].get(f"angle_avg_step_y")
            angle_avg_linear_x = aggregated_data["props_4th_file"].get(f"angle_avg_linear_x")
            angle_avg_linear_y = aggregated_data["props_4th_file"].get(f"angle_avg_linear_y")

            radian_neutrino = aggregated_data["props_4th_file"].get("radian_neutrino")
            radian_unweighted = aggregated_data["props_4th_file"].get("radian_unweighted")
            radian_avg_step = aggregated_data["props_4th_file"].get("radian_avg_step")
            radian_avg_lin = aggregated_data["props_4th_file"].get("radian_avg_lin")
            #print(run_id)

            mask = create_combined_event_cut_mask(
                props_4th_file=aggregated_data["props_4th_file"], 
                m_100gev_data=aggregated_data["m_100gev"], 
                hit_cut_name=hit_cut_name, 
                dl_100gev=dl_100gev_raw, 
                dl_400gev=dl_400gev_raw, 
                dl_cls=dl_cls_raw, 
                event_cut_dict=event_cut_dict, 
                num_events=num_events
            )
            

            if mask.sum() == 0:
                print(f"    [Atlandı] {hit_cut_name} için bu event cut'tan geçen veri yok.")
                continue

            dl_100gev  = dl_100gev_raw[mask]
            dl_400gev  = dl_400gev_raw[mask]
            dl_cls     = dl_cls_raw[mask]
            run_id = run_id[mask]
            event_number=event_number[mask]

            angle_incoming_neutrino_x=angle_incoming_neutrino_x[mask]
            angle_incoming_neutrino_y=angle_incoming_neutrino_y[mask]
            angle_unweighted_x=angle_unweighted_x[mask]
            angle_unweighted_y=angle_unweighted_y[mask]

            angle_avg_linear_y=angle_avg_linear_y[mask]
            angle_avg_linear_x=angle_avg_linear_x[mask]
            angle_avg_step_y=angle_avg_step_y[mask]
            angle_avg_step_x=angle_avg_step_x[mask]

            radian_neutrino=radian_neutrino[mask]
            radian_unweighted=radian_unweighted[mask]
            radian_avg_step=radian_avg_step[mask]
            radian_avg_lin=radian_avg_lin[mask]
            time_no_thr_first_diff_raw = aggregated_data["props_4th_file"].get("time_no_thr_first_diff_last_first")
            time_no_thr_mean_diff_raw = aggregated_data["props_4th_file"].get("time_no_thr_mean_diff_last_first")
            time_thr_0_first_diff_raw = aggregated_data["props_4th_file"].get("time_thr_0_mean_diff_last_first")
            # --- ZAMAN VERİLERİNE MASKE UYGULAMA ---
            time_no_thr_first_diff = time_no_thr_first_diff_raw[mask] if time_no_thr_first_diff_raw is not None else None
            time_no_thr_mean_diff = time_no_thr_mean_diff_raw[mask] if time_no_thr_mean_diff_raw is not None else None
            time_thr_0_first_diff = time_thr_0_first_diff_raw[mask] if time_thr_0_first_diff_raw is not None else None

            print("NUMBER OF EVENT", len(run_id),len(event_number),len(dl_100gev))
            #print("runid",run_id)
            #print("eventnumber",event_number)
        
            total_hits = total_hits_raw[mask] if total_hits_raw is not None else None
            total_qdc  = total_qdc_raw[mask]  if total_qdc_raw is not None else None
            frac_qdc   = frac_qdc_raw[mask]   if frac_qdc_raw is not None else None
            frac_hit   = frac_hit_raw[mask]   if frac_hit_raw is not None else None
            #print("frachit",frac_hit,"\n")

            save_dir_path = f"../{base_outdir}/{event_cut_name}" 
            
            qdc_2_gev=0.053
            output_file = f"events_list_{base_outdir}.txt"
            stop=0
            with open(output_file, "a") as file:
                for i in range(len(run_id)):
                    if stop >= 100:
                        break
                    r_id = run_id[i].item()
                    e_num = event_number[i].item()
                    if (r_id,e_num) not in alreadsaved:
                        alreadsaved.add((r_id,e_num))
                        stop+=1
                        ithdl_100 = dl_100gev[i].item()
                        ithdl_400= dl_400gev[i].item()
                        ithqdc_en = total_qdc[i].item()*qdc_2_gev
                        ithdl_cls = dl_cls[i].item()
                        ithfrac_qdc = frac_qdc[i].item()
                        ithfrac_hit = frac_hit[i].item()  

                        ith_neutrino_x = angle_incoming_neutrino_x[i].item()
                        ith_neutrino_y = angle_incoming_neutrino_y[i].item()
                        ith_unweighted_x = angle_unweighted_x[i].item()
                        ith_unweighted_y = angle_unweighted_y[i].item()
                        ith_avg_linear_x = angle_avg_linear_x[i].item()
                        ith_avg_linear_y = angle_avg_linear_y[i].item()
                        ith_avg_step_x = angle_avg_step_x[i].item()
                        ith_avg_step_y = angle_avg_step_y[i].item()

                        ith_time_first_diff = time_no_thr_first_diff[i].item() if time_no_thr_first_diff is not None else -999.0
                        ith_time_mean_diff = time_no_thr_mean_diff[i].item() if time_no_thr_mean_diff is not None else -999.0
                        ith_time_thr0_diff = time_thr_0_first_diff[i].item() if time_thr_0_first_diff is not None else -999.0

                        #radian_neutrino=radian_neutrino[i].item()
                        #radian_unweighted=radian_unweighted[i].item()
                        #radian_avg_step=radian_avg_step[i].item()
                        #radian_avg_lin=radian_avg_lin[i].item()

                        file.write(f"{save_dir_path} {r_id} {e_num} {ithdl_100:.1f} {ithdl_400:.1f} {ithqdc_en:.1f} {ithdl_cls:.2f} {ithfrac_qdc:.2f} {ithfrac_hit:.2f} {ith_neutrino_x} {ith_neutrino_y} {ith_unweighted_x} {ith_unweighted_y} {ith_avg_linear_x} {ith_avg_linear_y} {ith_avg_step_x} {ith_avg_step_y} {ith_time_first_diff} {ith_time_mean_diff} {ith_time_thr0_diff}\n")
                #print(f"Successfully saved {len(run_id_filtered)} events to {output_file}")

            #1d hists
            alpha_list=[0.5,0.5,0.5]
            
            label_str=["DL(trained up to 100GeV)","DL(trained up to 400GeV)",f"QDC*{qdc_2_gev}"]
            plot_multiple_hist([dl_100gev, dl_400gev,total_qdc*qdc_2_gev],20, "Reconstructed Energy[GeV]", "Reconstructed Energy Histograms", label_str, cut_outdir,alpha_list,name="recon_energy")
            plot_multiple_hist([dl_cls],20, "DL e-hadron Classifier", "DL e-hadron Classifier Histograms", [""], cut_outdir,[1],name="dl_cls")


            #2d hists
            plot_2d_hist(dl_400gev, dl_100gev, bins_x=50, bins_y=50, 
                         out_name="DL_400GeV_vs_100GeV", xlabel="DL Energy (400GeV)", ylabel="DL Energy (100GeV)", 
                         title=f"400GeV Model vs 100GeV Model", outdir=cut_outdir)

            plot_2d_hist(dl_400gev, dl_cls, bins_x=50, bins_y=50, 
                         out_name="DL_400GeV_vs_CLS", xlabel="DL Energy (400GeV)", ylabel="Classification Score", 
                         title=f"400GeV Energy vs CLS Score", outdir=cut_outdir)
            
            plot_2d_hist(dl_100gev, dl_cls, bins_x=50, bins_y=50, 
                         out_name="DL_100GeV_vs_CLS", xlabel="DL Energy (100GeV)", ylabel="Classification Score", 
                         title=f"100GeV Energy vs CLS Score", outdir=cut_outdir)
            
            plot_2d_hist(total_hits, dl_100gev, bins_x=50, bins_y=50, 
                         out_name="Hits_vs_DL_100GeV", xlabel="Total SciFi Hits", ylabel="DL Energy (100GeV)", 
                         title=f"Total Hits vs 100GeV Model", outdir=cut_outdir)
                         
            plot_2d_hist(total_hits, dl_400gev, bins_x=50, bins_y=50, 
                         out_name="Hits_vs_DL_400GeV", xlabel="Total SciFi Hits", ylabel="DL Energy (400GeV Log)", 
                         title=f"Total Hits vs 400GeV Model", outdir=cut_outdir)
            
            plot_2d_hist(total_qdc, dl_100gev, bins_x=50, bins_y=50, 
                         out_name="QDC_vs_DL_100GeV", xlabel="Total QDC", ylabel="DL Energy (100GeV)", 
                         title=f"Total QDC vs 100GeV Model", outdir=cut_outdir)
            
            plot_2d_hist(total_qdc, dl_400gev, bins_x=50, bins_y=50, 
                         out_name="QDC_vs_DL_400GeV", xlabel="Total QDC", ylabel="DL Energy (400GeV)", 
                         title=f"Total QDC vs 400GeV Model", outdir=cut_outdir)
            
            plot_2d_hist(frac_qdc, dl_cls, bins_x=50, bins_y=50, 
                         out_name="FracQDC_vs_CLS", xlabel="Log(Fraction Abs QDC)", ylabel="Classification Score", 
                         title=f"Shower Shape vs Classification", outdir=cut_outdir)
            
            plot_2d_hist(frac_hit, dl_cls, bins_x=50, bins_y=50, 
                         out_name="Frachit_vs_CLS", xlabel="Log(Fraction Hit)", ylabel="Classification Score", 
                         title=f"Shower Shape vs Classification", outdir=cut_outdir)
                        # 1. Frac QDC vs DL 400GeV
            plot_2d_hist(frac_qdc, dl_400gev, bins_x=50, bins_y=50, 
                        out_name="FracQDC_vs_DL400GeV", xlabel="Log(Fraction Abs QDC)", ylabel="DL Energy (400GeV)", 
                        title="Fraction QDC vs DL Energy (400GeV)", outdir=cut_outdir)

            # 2. Frac QDC vs DL 100GeV
            plot_2d_hist(frac_qdc, dl_100gev, bins_x=50, bins_y=50, 
                        out_name="FracQDC_vs_DL100GeV", xlabel="Log(Fraction Abs QDC)", ylabel="DL Energy (100GeV)", 
                        title="Fraction QDC vs DL Energy (100GeV)", outdir=cut_outdir)

            # 3. Frac Hit vs DL 400GeV
            plot_2d_hist(frac_hit, dl_400gev, bins_x=50, bins_y=50, 
                        out_name="FracHit_vs_DL400GeV", xlabel="Log(Fraction Hit)", ylabel="DL Energy (400GeV)", 
                        title="Fraction Hit vs DL Energy (400GeV)", outdir=cut_outdir)

            # 4. Frac Hit vs DL 100GeV
            plot_2d_hist(frac_hit, dl_100gev, bins_x=50, bins_y=50, 
                        out_name="FracHit_vs_DL100GeV", xlabel="Log(Fraction Hit)", ylabel="DL Energy (100GeV)", 
                        title="Fraction Hit vs DL Energy (100GeV)", outdir=cut_outdir)

            
                        # 1. Total Hits vs Total QDC
            plot_2d_hist(total_qdc, total_hits, bins_x=50, bins_y=50, 
                        out_name="TotalQDC_vs_TotalHits", xlabel="Total QDC", ylabel="Total Hits", 
                        title="Total Hits vs Total QDC", outdir=cut_outdir)

            # 2. Total Hits vs Frac QDC
            plot_2d_hist(frac_qdc, total_hits, bins_x=50, bins_y=50, 
                        out_name="FracQDC_vs_TotalHits", xlabel="Log(Fraction Abs QDC)", ylabel="Total Hits", 
                        title="Total Hits vs Fraction QDC", outdir=cut_outdir)

            # 3. Total Hits vs Frac Hit
            plot_2d_hist(frac_hit, total_hits, bins_x=50, bins_y=50, 
                        out_name="FracHit_vs_TotalHits", xlabel="Log(Fraction Hit)", ylabel="Total Hits", 
                        title="Total Hits vs Fraction Hit", outdir=cut_outdir)

            # 4. Total QDC vs Frac QDC
            plot_2d_hist(frac_qdc, total_qdc, bins_x=50, bins_y=50, 
                        out_name="FracQDC_vs_TotalQDC", xlabel="Log(Fraction Abs QDC)", ylabel="Total QDC", 
                        title="Total QDC vs Fraction QDC", outdir=cut_outdir)

            # 5. Total QDC vs Frac Hit
            plot_2d_hist(frac_hit, total_qdc, bins_x=50, bins_y=50, 
                        out_name="FracHit_vs_TotalQDC", xlabel="Log(Fraction Hit)", ylabel="Total QDC", 
                        title="Total QDC vs Fraction Hit", outdir=cut_outdir)

            # 6. Frac QDC vs Frac Hit
            plot_2d_hist(frac_qdc, frac_hit, bins_x=50, bins_y=50, 
                        out_name="FracQDC_vs_FracHit", xlabel="Log(Fraction Abs QDC)", ylabel="Log(Fraction Hit)", 
                        title="Fraction Hit vs Fraction QDC", outdir=cut_outdir)
            


def generate_electron_cuts(bases, energy_label, max_layers=None, us_leak="all"):
    if max_layers is None:
        max_layers = [2, 3, 4, 5] # Varsayılan olarak 3, 4 ve 5 katmanlı testleri yap
        
    # Sözlükleri tuple'dan çıkarıyoruz
    base_smaller, base_larger = bases

    configs = {}

    base_smaller = {**base_smaller,
    "radian_unweighted":max_radian_unw,
    "radian_avg_step":max_radian_avg_step,
    "radian_avg_lin":max_radian_avg_lin,
    "radian_step_1":max_radian_step_1 }

    base_larger = {**base_larger,
        #"time_no_thr_mean_diff_last_first": min_tof_diff,
        "time_thr_0_mean_diff_last_first": min_tof_diff,
        "time_thr_1_mean_diff_last_first": min_tof_diff,
        "time_thr_2_mean_diff_last_first": min_tof_diff,
        "time_no_thr_first_diff_last_first": min_tof_diff
    }

    for max_layer in max_layers:
        # 1. US ve DS Kaçış Yok
        if us_leak=="NOUSLEAK":
            name_no_us = f"Cut_method_{energy_label}_no_us_ds_scifi_layer_{max_layer}"
            configs[name_no_us] = {
                "larger_than": {
                    **base_larger,
                    "scifi_active_layers_notime_hitsN": max_layer -0.1,
                },
                "smaller_than": {
                    **base_smaller,
                    #"scifi_active_layers_notime_hits0": max_layer + 0.1, 
                    "scifi_active_layers_notime_hitsN": max_layer + 0.1, 
                    "scifi_active_layers_nocuts":max_layer + 1.1, # 1 layer sarkma olabilir ama dahası olamaz.
                    "us_notime_total_hits": 0.9, 
                    "us_active_layers_notime_qdc0": 0.9, 
                    "us_active_layers_notime_hits0": 0.9, 
                    "dsh_notime_total_hits": 0.9, 
                    "dsv_notime_total_hits": 0.9
                }
            }
        elif us_leak=="1USleak":
            # 2. Düşük US Kaçışına İzin Var (Low 1 US)
            name_low_us = f"Cut_method_{energy_label}_low1us_ds_scifi2_{max_layer}"
            configs[name_low_us] = {
                "larger_than": base_larger.copy(),
                "smaller_than": {
                    **base_smaller,
                    #"scifi_active_layers_notime_hits0": max_layer + 0.1, 
                    "scifi_active_layers_notime_hitsN": max_layer + 0.1, 
                    "us_notime_total_qdc": 2000,
                    "us_notime_total_hits": 100, 
                    "us_active_layers_notime_qdc0": 1.1, 
                    "us_active_layers_notime_hits0": 1.1, 
                    "dsh_notime_total_hits": 0.9, 
                    "dsv_notime_total_hits": 0.9
                }
            }
            
        elif us_leak=="ALLLEAK":
            # 3. Sadece SciFi Limiti (Sınırsız Hadronic/US/DS)
            name_base = f"Cut_method_{energy_label}_scifi2_{max_layer}"
            configs[name_base] = {
                "larger_than": base_larger.copy(),
                "smaller_than": {
                    **base_smaller,
                    #"scifi_active_layers_notime_hits0": max_layer + 0.1, 
                    "scifi_active_layers_notime_hitsN": max_layer + 0.1, 
                }
            }

    return configs


if __name__ == "__main__":
    layerscifihitnumber=20
    qdc_frac_largerthan=2.3#2.3
    hit_frac_largerthan=1.5#1.5
    year=2025
    
    min_tof_diff=0

    max_radian_unw=0.1
    max_radian_avg_lin=0.15
    max_radian_avg_step=0.15
    max_radian_step_1=0.15
    #

    dl_processed_data_dir = f"/eos/experiment/sndlhc/users/beturk/Data/PT/Data_{year}_DL_processed"
    extracted_props_dir   = f"/eos/experiment/sndlhc/users/beturk/Data/PT/Data_{year}_extracted_props"
    event_cut_configs={}
    model_prefixes = {
        "m_100gev": "energy_recon_MC_electrons_ResNets_SciFi_2layers_R256_100gev_-0.5_s2_q13_ft0_layer5_",
        "m_400gev": "energy_recon_log_input_more_data_MC_electrons_ResNets_SciFi_R128_400gev_0_s2_q13_ft0_layer2_",
        "m_cls":    "CLS_resnet_v6_TBHadrons23_MCElectrons_2layer_samexy_cut_ideal_"
    }
    
    hit_cuts_dict = {
        "TB_RECALIBRATION_S2Y": [False],
        "t_window_data": [(0.5,2.3)],
        "qdc_threshold_value_scifi_data": [0]
    }
    
    """event_cut_configs = {    
        "No_Event_Cuts": {
            "larger_than": {},
            "smaller_than": {}
        }
    }"""

    cal_suffix = "S2YcalFalse_qdcD0_twinD0.5_2.3"
    qdc_key = f"Total QDC{cal_suffix}"
    hits_key = f"Total Hits{cal_suffix}"
    qdc_frac_key = f"Log of Fraction Abs QDC{cal_suffix}"
    hit_frac_key = f"Log of Fraction Abs Hits{cal_suffix}"
    us_leak="NOUSLEAK"

    outdir_plots = f"analysis_{year}_plots_qdcfrac{qdc_frac_largerthan}_hitfrac{hit_frac_largerthan}_layers_1layerscifileakage_fid_vol_cuts_scifihitcut{layerscifihitnumber}_radian_{max_radian_unw}_{max_radian_avg_lin}_{max_radian_avg_step}_{max_radian_step_1}_tof_{min_tof_diff}"


    base_larger = {
        qdc_key: 150,
        hits_key: 40,
        #"scifi_active_layers_nocuts": 1.1, 
        #"scifi_active_layers_notime_hitsN": 1.1,
        qdc_frac_key:qdc_frac_largerthan,
        hit_frac_key:hit_frac_largerthan
    }

    base_smaller = {
        qdc_key: 500,
        hits_key: 200}
    
    event_cut_configs.update(
        generate_electron_cuts(
            bases=(base_smaller, base_larger), # Virgül eksikliği giderildi
            energy_label="lowerthan_50gev" , us_leak=us_leak
        )
    )


    base_larger = {
        qdc_key: 150,
        hits_key: 40,
        #"scifi_active_layers_nocuts": 1.1, 
        #"scifi_active_layers_notime_hitsN": 1.1,
        qdc_frac_key:qdc_frac_largerthan,
        hit_frac_key:hit_frac_largerthan
    }

    base_smaller = {}
    
    event_cut_configs.update(
        generate_electron_cuts(
            bases=(base_smaller, base_larger), # Virgül eksikliği giderildi
            energy_label="no_low_energy" , us_leak=us_leak
        )
    )



    base_larger = {
        qdc_key: 3000,
        hits_key: 900,
        #"scifi_active_layers_notime_hits0": 1.1, 
        #"scifi_active_layers_notime_hitsN": 1.1,
        qdc_frac_key:qdc_frac_largerthan,
        hit_frac_key:hit_frac_largerthan
    }
    base_smaller = {}
    
    event_cut_configs.update(
        generate_electron_cuts(
            bases=(base_smaller, base_larger), # Virgül eksikliği giderildi
            energy_label="higherthan_50gev" , us_leak=us_leak
        )
    )

    # --- 2. KESME: 50-150 GeV ELECTRONS ---
    base_larger = {
        qdc_key: 500,
        hits_key: 200,
        #"scifi_active_layers_notime_hits0": 1.1, 
        #"scifi_active_layers_notime_hitsN": 1.1,
        qdc_frac_key:qdc_frac_largerthan,
        hit_frac_key:hit_frac_largerthan
    }
    base_smaller = {
        qdc_key: 3000, 
        hits_key: 900
    }
    
    event_cut_configs.update(
        generate_electron_cuts(
            bases=(base_smaller, base_larger), # Parametre ataması eklendi
            energy_label="50-150gev_electrons", us_leak=us_leak
        )
    )

    """for max_layer in [3, 4, 5]:  #USUAL method, nothing sophisticated.
        config_name = f"Cut_method_no_us_ds_scifi2_{max_layer}"
        event_cut_configs[config_name] = {
            "larger_than": {
                #"scifi_active_layers_notime_hits0": 1.1, 
                #"scifi_active_layers_notime_hitsN": 1.1,
                qdc_frac_key:qdc_frac_largerthan,
                hit_frac_key:hit_frac_largerthan
            },
            "smaller_than": {
                #"scifi_active_layers_notime_hits0": max_layer + 0.1, # 3.1, 4.1, 5.1 olur
                #"scifi_active_layers_notime_hitsN": max_layer + 0.1, 
                "us_notime_total_hits": 0.9, 
                "us_active_layers_notime_qdc0": 0.9, 
                "us_active_layers_notime_hits0": 0.9, 
                "dsh_notime_total_hits": 0.9, 
                "dsv_notime_total_hits": 0.9
            }
        }"""
        
    files_to_process = np.arange(0,10,1)
    print(files_to_process)
    
    analyze_dl_physics(files_to_process, dl_processed_data_dir, extracted_props_dir, hit_cuts_dict, event_cut_configs, model_prefixes, outdir_plots)



#for cuts apply also logfrac cuts.
#for cuts apply also logfrac cuts and cls output >-20
#plot1d hists of cls.
#after cls cut

# do like this.
# for only 2 layers(3,4, and 5 scifi layers) # take single layer.
#   for no us ds leakage(1 us leakage, unlimited leakage)
#       for low scifi energy(<500 qdc'ye kadar.) ve > 500-3000 qdc. ve  > 3000qdc
#           
#print run_id and event_numbers for all and save them into txt files 