#!/bin/bash

INPUT_FILE=$1


PYTHON_SCRIPT="/afs/cern.ch/work/b/beturk/private/snd/data_analysis/plotting_2devents/save_2devent.py"

MAX_JOBS=8

while read -r save_dir run_id event_number dl_100 dl_400 qdc_en dl_cls frac_qdc frac_hit \
              neutrino_x neutrino_y unweighted_x unweighted_y lin_x lin_y step_x step_y time1 time2 time3; do
    
    (
        echo "Processing Run: $run_id, Event: $event_number"
        
        cd "$save_dir" || exit 1
        

        ##unweighted_theta, average_step_theta,  average_lin_theta 
        python "$PYTHON_SCRIPT" -r "$run_id" -en "$event_number" --rootbatch --collision_axis --shower_dir --extension pdf \
            --which_angle "unweighted_theta" --dl_100 "$dl_100" --dl_400 "$dl_400" --qdc_en "$qdc_en" --dl_cls "$dl_cls" \
            --frac_qdc "$frac_qdc" --frac_hit "$frac_hit" \
            --neutrino_theta_x "$neutrino_x" --neutrino_theta_y "$neutrino_y" \
            --unweighted_theta_x "$unweighted_x" --unweighted_theta_y "$unweighted_y" \
            --average_lin_theta_x "$lin_x" --average_lin_theta_y "$lin_y" \
            --average_step_theta_x "$step_x" --average_step_theta_y "$step_y"
    
        rm -f *_0000.png
        
    ) &  # '&' işareti bu bloğu arka planda (paralel) çalışmaya zorlar
    
    if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
        wait -n
    fi

done < "$INPUT_FILE"

wait

cd /afs/cern.ch/work/b/beturk/private/snd/data_analysis/plotting_2devents || exit
echo "All events processed."