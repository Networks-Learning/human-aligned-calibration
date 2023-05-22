from human_ai_interactions_data import haiid
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.calibration import calibration_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None
np.set_printoptions(precision=3)
pd.options.display.float_format = "{:,.6f}".format
path = "plots/"

class Experiment:
    #class constructor, sets experiment's parameters for task
    def __init__(self, name, h_bins, b_bins):
        self.task_name = name
        self.h_bins = h_bins
        self.b_bins = b_bins
        self.h_bin_boundaries = []
        self.min_datapoints = 30
        self.task_data = None
        self.df_cell_prob = None
        self.df_cell_mass = None
        self.df_metrics = None

        self.set_task_data()
        #compute probability matrix for cells
        self.compute_cell_prob_matrix()
        #run experiment
        self.run_experiment()

    # retrieve data in from of (h,b,y,t) for the defined task, 
    # h =human risk estimate, b=model risk estimate, y= outcome, t=decision (response of human after seeing b)
    def set_task_data(self):

        # load all data
        df = haiid.load_dataset("./human_ai_interactions_data")
        # get specified task data
        task_df = haiid.load_task(df, self.task_name)
        if self.task_name =='census':
            self.task_data = self.preprocess_data(task_df, '>=50k')
        elif self.task_name =='sarcasm':
            self.task_data = self.preprocess_data(task_df, 'sarcasm')
        else:
            self.task_data  = self.preprocess_data(task_df)
        
        self.discretize_human_estimates()

        return (self.task_data)

    # map advice in [-1,1] to probabilities in [0,1] by assigning event Y=1 to label specified by label_1 
    # if label_1==None, assigns event Y=1 to a random label per task instance 
    def preprocess_data(self, data, label_1=None):

        #filter data for 'geographic region' and 'perceived accuracy'
        df = data.loc[(data['geographic_region']=='United States') & (data['perceived_accuracy']==80)]
        #select relevant data columns
        df = df[['task_instance_id','participant_id','correct_label', 'advice', 'response_1','response_2']] 

        #assign event Y=1
        if label_1 == None:
            #assigning event Y=1 to a random label per task instance
            #randomness needed since there are 4 labels accross tasks, but each task has two labels -> assign one of them to event Y=1 
            np.random.seed(320)
            task_ids = df['task_instance_id'].unique()
            df_label = pd.DataFrame(task_ids, columns=['task_instance_id'])
            df_label['y'] = np.random.choice(2, len(task_ids)).astype(int) 
            df = df.merge(df_label, how='left', on='task_instance_id' )
        else:
            #assigning event Y=1 to label specified by label_1 
            df['y'] = (df['correct_label']==label_1).astype(int)

        #compute mapping from [-1,1] to [0,1]
        df[['b','h','h+AI']] = (df.loc[:,[ 'advice','response_1', 'response_2']]+1)/ 2.0 
        df.loc[df['y']==0,['b','h','h+AI']] = 1 - df.loc[df['y']==0,['b','h','h+AI']]
 
        #print data instances used
        # print(self.task_name)
        # print('all regions: ', data['geographic_region'].unique())
        # print('region used: ', 'United States')
        # print('number of participants: ', data['participant_id'].unique().size)
        # print('number of participants used: ', df['participant_id'].unique().size)
        # print('all datapoints: ',data.shape[0])
        # print('used datapoints: ',df.shape[0])


        return(df[['task_instance_id','participant_id','h','b','y','h+AI']])


    #discretize human estimates into n_bins bins with aprroximately the same mass
    def discretize_human_estimates(self):

        #sort calibration data by human estimate h ascending
        df = self.task_data.copy()
        df= df.sort_values(by=['h'])
        
        # split calibration data into uniform sized bins
        #find bin boundaries
        if self.h_bins==2 :
            bin_bounds = [0.5, 1]
        else:
            df_split = np.array_split(df, self.h_bins)
            #find maximum value of h in each bin
            bin_bounds = []
            for df_h in df_split:
                max_h = round(df_h.loc[:,'h'].max(), 2)
                bin_bounds.append(max_h)

        # set value of h in calibration data to maximum value in each bin
        df.loc[ (df['h']< bin_bounds[0]) | np.isclose(df['h'],bin_bounds[0]), ['h_bin']] = bin_bounds[0]  
        for i in range(1, len(bin_bounds)):
            df.loc[((df['h']< bin_bounds[i]) | np.isclose(df['h'],bin_bounds[i])) & (df['h']> bin_bounds[i-1]),['h_bin']] = bin_bounds[i]   

        #set data and h bin boundaries
        self.task_data = df
        self.h_bin_boundaries = bin_bounds
    

    #compute probabilities and density mass of each bin (h,b)
    def compute_cell_prob_matrix(self):
        df = self.task_data
        lambda_param = 1/self.b_bins
        
        # partition model risk estimate space into B bins,
        # set new column in df with the max bin value (indicates in which bin each data point is))
        df['b_bin'] = (df['b']// lambda_param) +1
        df.loc[df['b_bin']==self.b_bins+1, ['b_bin']] = self.b_bins
        df['b_bin'] *= lambda_param
        df['b_bin'] = df['b_bin'].round(3) 

        #drop original model risk estimates
        df = df.drop(columns=['b','h+AI'])
        #compute probability Y=1 and density mass in each bin
        self.df_cell_prob = df.groupby(by=['b_bin','h_bin'])['y'].mean().unstack()
        self.df_cell_mass = df.groupby(by=['b_bin','h_bin'])['y'].count().unstack()

        

    #computes expected and average alignment error
    def check_alignment(self):
        #compute max and average alignment violations
        max_aligment = 0.0
        sum_aligment = 0.0
        num_summants = 0.0
        disaligned_cells = set({})

        for h in range(self.df_cell_prob.columns.shape[0]):
            for b in range(self.df_cell_prob.index.shape[0]):
                for h1 in range(h+1):
                    for b1 in range(b+1):
			num_summants +=1
                        # check misalignment of the pair of cells if enough datapoints in each cells
                        if (self.df_cell_mass.iat[b,h]>=self.min_datapoints) & (self.df_cell_mass.iat[b1,h1]>=self.min_datapoints):
                            alignment = max(0.0, self.df_cell_prob.iat[b1,h1] - self.df_cell_prob.iat[b,h] )
                            max_aligment = max(max_aligment, alignment)
                            
                            if alignment > 0.0:
                                sum_aligment += alignment
                                disaligned_cells |= {(self.df_cell_prob.index[b],self.df_cell_prob.columns[h]),(self.df_cell_prob.index[b1],self.df_cell_prob.columns[h1])}
                                
        if num_summants > 0: 
            avg_alignment = sum_aligment / num_summants
        else:
            avg_alignment = 0.0 
        
        self.disaligned_cells = disaligned_cells 
        mae = max_aligment 
        eae = avg_alignment

        return(eae, mae)

    #computes expected and maximum calibration error
    def check_calibration(self):
        prob_true, prob_pred = calibration_curve(self.task_data['y'], self.task_data['b'], n_bins=self.b_bins)
        abs_diff = pd.DataFrame(data={'b_mass': prob_true-prob_pred})
        abs_diff['b_mass'] = abs_diff['b_mass'].abs()

        #compute maximum calibration error (MCE)
        mce = abs_diff['b_mass'].max()

        #compute expected calibration error (ECE)
        df_density = self.df_cell_mass / self.df_cell_mass.sum().sum()
        b_bin_mass = df_density.sum(axis=1)

        ece = b_bin_mass * abs(prob_true-prob_pred)
        ece = ece.sum(axis=0)

        return(ece, mce)

    def run_experiment(self):

        #compute expected and maximum alignment error
        eae, mae = self.check_alignment() 
        
        #compute calibration measures ECE and MCE
        ece, mce = self.check_calibration()

        #compute ROC AUC
        roc_auc_h = roc_auc_score(self.task_data['y'], self.task_data['h']) 
        roc_auc_hAI = roc_auc_score(self.task_data['y'], self.task_data['h+AI']) 
        roc_auc_b = roc_auc_score(self.task_data['y'], self.task_data['b']) 

        dict_metrics = {'EAE': [eae], 'MAE': [mae], 'ECE': [ece],'MCE': [mce],'roc_auc_b':[roc_auc_b], 'roc_auc_h':[roc_auc_h],'roc_auc_h+AI':[roc_auc_hAI]}
        self.df_metrics = pd.DataFrame(dict_metrics, index=[self.task_name])

        all_hatched =[]
        for h in range(self.df_cell_prob.columns.shape[0]):
            for b in range(self.df_cell_prob.index.shape[0]):
                if (self.df_cell_prob.index[b],self.df_cell_prob.columns[h]) in self.disaligned_cells:
                    all_hatched.append('//'*3)
                else:
                    all_hatched.append('') 

        alignment_barplot(self.task_data, all_hatched, self.min_datapoints, self.task_name)
        confidence_change_barplot(self.task_data, all_hatched, self.task_name)
        plot_histogram_cells(self.task_data, self.task_name)
        plot_roc(self.task_data, self.task_name)

    def get_task_data(self):
        return (self.task_data)

    def get_metrics(self):
        return (self.df_metrics)

#plots ROC curve
def plot_roc(data, task_name):
    fig, ax = plt.subplots()
    fpr, tpr, th = roc_curve(data['y'], data['b'])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    
    fpr, tpr, th = roc_curve(data['y'], data['h'])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    
    fpr, tpr, th = roc_curve(data['y'], data['h+AI'])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
    ax.set_xlabel('False Positive Rate',fontsize=18)
    ax.set_ylabel('True Positive Rate',fontsize=18)
    ax.tick_params(labelsize=18)
    plt.legend([r'$\pi_B$',r'$\pi_H$',r'$\pi_{H_\mathrm{+AI}}$'],title ='Decision Policies',title_fontsize=18, prop = { "size": 18},loc='lower right')
    plt.tight_layout()
    plt.savefig(path+"/roc/roc_"+task_name+".pdf")
    plt.close()


#barplot of cell probabilities P(Y=1 | cell=(h,b))
def alignment_barplot(df, bar_hatches, min_cell_mass, name):

    #
    h_conf_bin_limits = df['h_bin'].unique()
    legend_array = [ '\'low\': [0.0, ' + str(h_conf_bin_limits[0])+']']
    legend_array += [ '\'mid\': ('+str(h_conf_bin_limits[0]) +', ' + str(h_conf_bin_limits[1])+']'] 
    legend_array += [ '\'high\': ('+str(h_conf_bin_limits[1]) +', ' + str(h_conf_bin_limits[2])+']'] 

    #plot figures
    custom_params = {"axes.edgecolor": "lightgray"} #"axes.spines.right": False, "axes.spines.top": False,
    sns.set(context="paper", style="ticks", color_codes=True, rc=custom_params)
    fig, ax = plt.subplots(figsize=(10,5))
    #plot all bins with errorbar
    bar = sns.barplot(x='b_bin', y='y', hue='h_bin',estimator=np.nanmean, errorbar=('ci', 90), errwidth=.12, capsize=.12, data=df, palette="colorblind", ax=ax)
    # set hatch to bars with disalignment (only considers disalignment on bins with mass >= min_bin_mass_prob)
    if len(bar_hatches)>0:
        for i, b in enumerate(bar.patches):
            b.set_hatch(bar_hatches[i])

    ax.set_xlabel( r"Model Confidence, $b$",fontsize=18)
    ax.set_ylabel( r"$P(Y=1 \mid (X,Y) \in \mathcal{S}_{h,\lambda(b)})$",fontsize=18)
    ax.tick_params(labelsize=18)
    ax.set_ylim(-0.02,1.05)
    #change legend
    h, _ = bar.get_legend_handles_labels()
    bar.legend(h, legend_array, title =r"Human Confidence, $h$",loc='upper left', prop = { "size": 18}, title_fontsize=18)

    plt.tight_layout()
    plt.savefig(path+"/barplot/alignment_"+name+".pdf")
    plt.close()

#barplot of cell expectation of confidence change E(h_+AI - h | cell=(h,b))
def confidence_change_barplot(df, bar_hatches, name):
    

    df['confidence_change'] = df['h+AI']-df['h']     
    #plot figures
    custom_params = {"axes.edgecolor": "lightgray"} #"axes.spines.right": False, "axes.spines.top": False,
    sns.set(context="paper", style="ticks", color_codes=True, rc=custom_params)
    fig, ax = plt.subplots(figsize=(10,5))
    #plot all bins with errorbar
    bar = sns.barplot(x='b_bin', y='confidence_change', hue='h_bin',estimator=np.nanmean, errorbar=('ci', 90), errwidth=.12, capsize=.12, data=df, palette="colorblind", ax=ax)
    
    # set hatch to bars with disalignment (only considers disalignment on bins with mass >= min_bin_mass_prob)
    if len(bar_hatches)>0:
        for i, b in enumerate(bar.patches):
            b.set_hatch(bar_hatches[i])

    h_conf_bin_limits = df['h_bin'].unique()
    legend_array = [ '\'low\': [0.0, ' + str(h_conf_bin_limits[0])+']']
    legend_array += [ '\'mid\': ('+str(h_conf_bin_limits[0]) +', ' + str(h_conf_bin_limits[1])+']'] 
    legend_array += [ '\'high\': ('+str(h_conf_bin_limits[1]) +', ' + str(h_conf_bin_limits[2])+']'] 

    #plot only large bins with errorbar
    ax.set_xlabel( r"Model Confidence, $b$",fontsize=18)
    ax.set_ylabel( r"$E[ h_{\mathrm{+AI}} - h \mid (X,Y) \in \mathcal{S}_{h,\lambda(b)}]$",fontsize=18)
    ax.tick_params(labelsize=18)
    #change legend
    h, _ = bar.get_legend_handles_labels()
    bar.legend(h, legend_array, title =r"Human Confidence, $h$",loc='lower right', prop = { "size": 18}, title_fontsize=18)
    plt.tight_layout()
    plt.savefig(path+"/barplot/confidence_change_"+name+".pdf")
    plt.close()


#histogram of cell mass
def plot_histogram_cells(df, name):

    #2d histogram
    custom_params = {"axes.edgecolor": "lightgray"} 
    sns.set(context="paper", style="ticks", color_codes=True, rc=custom_params)
    fig, ax = plt.subplots(figsize=(10,5))

    df_cell_mass = df[['b_bin','h_bin','y']]
    df_cell_mass = df_cell_mass.groupby(['h_bin','b_bin']).count().unstack()
    df_reverse = df_cell_mass.reindex(index=df_cell_mass.index[::-1])
    
    bar = sns.heatmap(df_reverse,cbar=True, cmap="crest")
    cbar = bar.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax.set_ylabel(r"Human Confidence $h$", fontsize=16)
    ax.set_xlabel(r"Model Confidence $b$", fontsize=16)
    ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(path+"/hist/hist2d_"+name+".pdf")
    plt.close()

    
def main():
    
    exp_art = Experiment('art', h_bins=3, b_bins=8)
    exp_sarcasm = Experiment('sarcasm', h_bins=3, b_bins=8)
    exp_cities = Experiment('cities', h_bins=3, b_bins=8)
    exp_census = Experiment('census', h_bins=3, b_bins=8)

    df_results = exp_art.get_metrics()
    df_results = pd.concat([df_results, exp_sarcasm.get_metrics()])
    df_results = pd.concat([df_results, exp_cities.get_metrics()])
    df_results = pd.concat([df_results, exp_census.get_metrics()])

    print(df_results)


if __name__ == "__main__":
    main()