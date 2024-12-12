OUTFILE="results.csv"
if [ ! -f "$OUTFILE" ]; then
    echo "method,score_func,N_AVG,coverage,covGap,mean_size,std_size" > "$OUTFILE"
fi

#METHODS="standard classwise clustered"
METHODS="standard"
#N_AVG_VALUES="10 20 30 40 50 75 100 150"
N_AVG_VALUES="10"
#SCORE_FUNCS="softmax APS RAPS"
SCORE_FUNCS="softmax"

for method in $METHODS; do
    for N in $N_AVG_VALUES; do
        if [ "$method" = "standard" ]; then
            SCRIPT="Experiments/run_standard_conformal.py"
        elif [ "$method" = "classwise" ]; then
            SCRIPT="Experiments/run_classwise_conformal.py"
        elif [ "$method" = "clustered" ]; then
            SCRIPT="Experiments/run_clustered_conformal.py"
        else
            echo "Unknown method: $method"
            exit 1
        fi
        
        echo "Running $method method with scoring functions: $SCORE_FUNCS and N_AVG=$N ..."
        
        # Run once per scoring function, passing them individually
        for sf in $SCORE_FUNCS; do
            python3 "$SCRIPT" --N_AVG "$N" --score_func "$sf" >> "$OUTFILE"
        done
    done
done

echo "All experiments completed. Results are in $OUTFILE."