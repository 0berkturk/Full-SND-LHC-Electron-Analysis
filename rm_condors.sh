# Delete files that have all three: .log, .out, .err
for f in *.log; do
    base="${f%.log}"   # strip .log
    if [[ -f "${base}.out" || -f "${base}.err" ]]; then
        echo "Deleting ${base}.log, ${base}.out, ${base}.err"
        rm "${base}.log" "${base}.out" "${base}.err"
    fi
done
