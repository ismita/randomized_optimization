#%%
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import mlrose_hiive

pd.options.display.max_columns = 1000
pd.options.display.max_info_columns = 1000
pd.options.display.max_info_rows = 10000000


#Fitness function
fitness = mlrose_hiive.FourPeaks(t_pct=0.2)
seed = 7
problem_size = 100
max_attp = 200
iter_list = [1000] # 50 * np.arange(20)
#Definning the problem
problem = mlrose_hiive.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=True, max_val=2)

# res_list = [0,1,2]
res_times = 9
rhc = mlrose_hiive.RHCRunner(problem=problem,
                       experiment_name="RHC",
                       output_directory="./four_peaks_problem",
                       seed=seed,
                       iteration_list=iter_list,
                       max_attempts=max_attp,
                       restart_list=[res_times])  # restart position goes from 0 to the number in this list
rhc_run_stats, rhc_run_curves = rhc.run()

# schedule = mlrose_hiive.ExpDecay()

temperature_list= [1, 10, 50, 100]
sa = mlrose_hiive.SARunner(problem=problem,
                     experiment_name="SA",
                       output_directory="./four_peaks_problem",
                     seed=seed,
                     iteration_list=iter_list,
                     max_attempts=max_attp,
                     temperature_list=temperature_list,
                     decay_list=[mlrose_hiive.GeomDecay])
sa_run_stats, sa_run_curves = sa.run()
sa_run_curves['Temperature'] = sa_run_curves.Temperature.astype('str')


mutation_list = [0.01, 0.05, 0.1, 0.2]
ga = mlrose_hiive.GARunner(problem=problem,
                     experiment_name="GA",
                       output_directory="./four_peaks_problem",
                     seed=seed,
                     iteration_list=iter_list,
                     max_attempts=max_attp,
                     population_sizes=[max(10, int(problem_size*0.2))],
                     mutation_rates=mutation_list)
ga_run_stats, ga_run_curves = ga.run()
ga_run_curves['Mutation Rate'] =ga_run_curves['Mutation Rate'].astype('str')

percent_list = [0.1, 0.2, 0.3]
mimic = mlrose_hiive.MIMICRunner(problem=problem,
                           experiment_name="MIMIC",
                       output_directory="./four_peaks_problem",
                           seed=seed,
                           iteration_list=iter_list,
                           population_sizes=[max(10, int(problem_size*0.2))],
                           max_attempts=max_attp,
                           keep_percent_list=percent_list,
                           use_fast_mimic=True)
mimic_run_stats, mimic_run_curves = mimic.run()
mimic_run_curves['Keep Percent'] = mimic_run_curves['Keep Percent'].astype('str')


fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# for rs in res_list:
selected = rhc_run_curves[rhc_run_curves['current_restart'] == res_times]
ax[0,0].plot(selected["Iteration"], selected["Fitness"], label=res_times + 1)
ax[0,0].legend(loc="lower right",title='Restarts')
ax[0,0].set_xlabel('Iterations')
ax[0,0].set_ylabel('Fitness')
ax[0,0].title.set_text('RHC')


for t in temperature_list:
    selected = sa_run_curves[sa_run_curves['Temperature']==str(t)] # t is int
    ax[0,1].plot(selected["Iteration"], selected["Fitness"],  label = t)
ax[0,1].legend(loc="lower right", title='Temperature')
# ax[0,1].legend(title='Temperature')
ax[0,1].set_xlabel('Iterations')
ax[0,1].set_ylabel('Fitness')
ax[0,1].title.set_text('SA')


for m in mutation_list:
    selected = ga_run_curves[ga_run_curves['Mutation Rate']==str(m)]
    ax[1,0].plot(selected["Iteration"], selected["Fitness"],  label = m)
ax[1,0].legend(loc="lower right", title='Mutation rates')
# ax[1,0].legend(title='Mutation rates')
ax[1,0].set_xlabel('Iterations')
ax[1,0].set_ylabel('Fitness')
ax[1,0].title.set_text('GA')


for p in percent_list:
    selected = mimic_run_curves[mimic_run_curves['Keep Percent']==str(p)]
    ax[1,1].plot(selected["Iteration"], selected["Fitness"],  label = p)
ax[1,1].legend(loc="lower right", title='Keep percent')
# ax[1,1].legend(title='Keep percent')
ax[1,1].set_xlabel('Iterations')
ax[1,1].set_ylabel('Fitness')
ax[1,1].title.set_text('MIMIC')
# ax[1,1].set_ylim([int(problem_size*0.2), int(problem_size*1.05)])

# plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
# fig.tight_layout(pad=1.0)
fig.suptitle('Fitness Curves - Four Peaks Problem')
plt.subplots_adjust(left=0.05,
                    bottom=0.05,
                    right=0.95,
                    top=0.95,
                    wspace=0.3,
                    hspace=0.3)
# plt.show()
plt.savefig('./four_peaks_problem/fitness_curves_each_fpp.png')


#%%
# combine the plots
# choose the ones with highest fitness
rhc_run_df = rhc_run_curves[rhc_run_curves['current_restart']==res_times][['Fitness','Iteration','Time']]
sa_run_df = sa_run_curves[sa_run_curves['Temperature']=='50'][['Fitness','Iteration','Time']]
ga_run_df = ga_run_curves[ga_run_curves['Mutation Rate']=='0.2'][['Fitness','Iteration','Time']]
mimic_df = mimic_run_curves[mimic_run_curves['Keep Percent']=='0.2'][['Fitness','Iteration','Time']]


rhc_run_df['Alg'] = 'RHC'
sa_run_df['Alg'] = 'SA'
ga_run_df['Alg'] = 'GA'
mimic_df['Alg'] = 'MIMIC'

summary = pd.concat([rhc_run_df, sa_run_df,ga_run_df,mimic_df])

iter_vs_fit = px.line(summary, x="Iteration", y="Fitness", color='Alg',title="Fitness Curves Comparison Four Peaks Problem",width=600, height=500)
# iter_vs_fit.show()
iter_vs_fit.update_layout(margin=dict(l=30, r=20, t=50, b=20))

iter_vs_fit.write_image("./four_peaks_problem/alg_comparison_fpp.png") 

alg_time = pd.DataFrame(list(zip(['RHC','SA','GA','MIMIC'], [rhc_run_df.iloc[max_attp].Time, sa_run_df.iloc[max_attp].Time,ga_run_df.iloc[max_attp].Time,mimic_df.iloc[max_attp].Time])),
               columns =['Alg', 'Time'])

time_fig = px.bar(alg_time, x='Alg', y='Time',
			title="Execution Time Comparison Four Peaks Problem",
			color = 'Alg',
			width=600, height=500)

time_fig.update_layout(margin=dict(l=30, r=20, t=50, b=20))

time_fig.write_image("./four_peaks_problem/time_comparison_fpp.png") 


# %%
