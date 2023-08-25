## UNCOMMENT THE FOLLOWING LINES TO RUN THE EXPERIMENTS
# ## German Dataset
# for i in 1 2 3 4 5 6 7 8 9 10
# do
#     for b in 6 7 9 -1  #### -1 for no bias
#     do
#         echo "Run no. $i"
#         python run_main.py --file config_German.jsonc --loss PL-rank-3 --postprocess_algorithms none,GDL23,GAK19 --run_no $i --bias $b
#         python run_main.py --file config_German.jsonc --loss Group-Fair-PL --postprocess_algorithms none,GDL23,GAK19 --run_no $i --bias $b
#     done  
# done

# for b in 6 7 8 9 -1
# do
#     python plot_results.py --dataset German --postprocess_algorithms none,GDL23,GAK19 --bias $b --k 20
# done


# ## pcaHMDA Dataset
# for i in 1 2 3 4 5 6 7 8 9 10
# do
#     for b in 8 85 9 -1  #### -1 for no bias
#     do
#         echo "Run no. $i"
#         python run_main.py --file config_pcaHMDA.jsonc --loss PL-rank-3 --postprocess_algorithms none,GDL23,GAK19 --run_no $i --bias $b
#         python run_main.py --file config_pcaHMDA.jsonc --loss Group-Fair-PL --postprocess_algorithms none,GDL23,GAK19 --run_no $i --bias $b
#     done  
# done

# for b in 8 85 9 -1
# do
#     python plot_results.py --dataset pcaHMDA --postprocess_algorithms none,GDL23,GAK19 --bias $b --k 25
# done



# ## MOVLENS Dataset
# for i in 1 2 3 4 5 6 7 8 9 10
# do
#     for b in 7 8 9 -1  #### -1 for no bias
#     do
#         echo "Run no. $i"
#         python run_main.py --file config_MOVLENS.jsonc --loss PL-rank-3 --postprocess_algorithms none,GDL23,GAK19 --run_no $i --bias $b
#         python run_main.py --file config_MOVLENS.jsonc --loss Group-Fair-PL --postprocess_algorithms none,GDL23,GAK19 --run_no $i --bias $b
#     done  
# done

# for b in 7 8 9 -1
# do
#     python plot_results.py --dataset MOVLENS --postprocess_algorithms none,GDL23,GAK19 --bias $b --k 10
# done