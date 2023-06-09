import pandas as pd
#to create summary from already existing data
datasets = ['metabric', 'flchain', 'rgbsg', 'support']
path = '/Users/JUSC/Documents/xgbsurv/experiments/deep_learning/'
agg_metrics_cindex = []
agg_metrics_ibs = []
for name in datasets:
    name = str.upper(name)
    df_outer_scores = pd.read_csv(path+'eh_skorch_results/eh_metric_summary_'+name+'_adapted.csv')
    print(df_outer_scores)
    dataset_name = name
    # cindex
    df_agg_metrics_cindex = pd.DataFrame({'dataset':[dataset_name],
                                            'cindex_test_mean':df_outer_scores['cindex_test_'+dataset_name].mean(),
                                            'cindex_test_std':df_outer_scores['cindex_test_'+dataset_name].std() })
    # IBS
    df_agg_metrics_ibs = pd.DataFrame({'dataset':[dataset_name],
                                            'ibs_test_mean':df_outer_scores['ibs_test_'+dataset_name].mean(),
                                            'ibs_test_std':df_outer_scores['ibs_test_'+dataset_name].std() })
    
    agg_metrics_cindex.append(df_agg_metrics_cindex)
    agg_metrics_ibs.append(df_agg_metrics_ibs)

df_final_eh_1_cindex = pd.concat([df for df in agg_metrics_cindex]).round(4)
df_final_eh_1_cindex.to_csv(path+'metrics/final_dl_1_eh_cindex.csv', index=False)
#df_final_eh_1_cindex.to_csv('/Users/JUSC/Documents/644928e0fb7e147893e8ec15/05_thesis/tables/final_dl_1_eh_cindex.csv', index=False)  #
df_final_eh_1_cindex

df_final_eh_1_ibs = pd.concat([df for df in agg_metrics_ibs]).round(4)
df_final_eh_1_ibs.to_csv(path+'metrics/final_dl_1_eh_ibs.csv', index=False)
#df_final_eh_1_ibs.to_csv('/Users/JUSC/Documents/644928e0fb7e147893e8ec15/05_thesis/tables/final_dl_1_eh_ibs.csv', index=False) 
df_final_eh_1_ibs
    

# to create summary from already existing data
cancer_types = [
    'BLCA',
    'BRCA',
    'HNSC',
    'KIRC',
    'LGG',
    'LIHC',
    'LUAD',
    'LUSC',
    'OV',
    'STAD']


agg_metrics_cindex = []
agg_metrics_ibs = []
for name in cancer_types:
    df_outer_scores = pd.read_csv(path+'eh_skorch_results/eh_tcga_metric_summary_'+name+'_adapted.csv')
    dataset_name = name #+'_adapted'
    # cindex
    df_agg_metrics_cindex = pd.DataFrame({'dataset':[dataset_name],
                                            'cindex_test_mean':df_outer_scores['cindex_test_'+dataset_name].mean(),
                                            'cindex_test_std':df_outer_scores['cindex_test_'+dataset_name].std() })
    # IBS
    df_agg_metrics_ibs = pd.DataFrame({'dataset':[dataset_name],
                                            'ibs_test_mean':df_outer_scores['ibs_test_'+dataset_name].mean(),
                                            'ibs_test_std':df_outer_scores['ibs_test_'+dataset_name].std() })
    
    agg_metrics_cindex.append(df_agg_metrics_cindex)
    agg_metrics_ibs.append(df_agg_metrics_ibs)

df_final_eh_tcga_cindex = pd.concat([df for df in agg_metrics_cindex]).round(4)
df_final_eh_tcga_cindex.to_csv(path+'metrics/final_dl_tcga_eh_cindex.csv', index=False)
df_final_eh_tcga_ibs = pd.concat([df for df in agg_metrics_ibs]).round(4)
df_final_eh_tcga_ibs.to_csv(path+'metrics/final_dl_tcga_eh_ibs.csv', index=False)



# to create summary from already existing data xgbsurv
cancer_types = [
    'BLCA',
    'BRCA',
    'HNSC',
    'KIRC',
    'LGG',
    'LIHC',
    'LUAD',
    'LUSC',
    'OV',
    'STAD']

path = '/Users/JUSC/Documents/xgbsurv/experiments/boosting/'
agg_metrics_cindex = []
agg_metrics_ibs = []
for name in cancer_types:
    df_outer_scores = pd.read_csv(path+'eh_xgbsurv_results/eh_metric_summary_'+name+'_adapted.csv')
    dataset_name = name #+'_adapted'
    # cindex
    df_agg_metrics_cindex = pd.DataFrame({'dataset':[dataset_name],
                                            'cindex_test_mean':df_outer_scores['cindex_test_'+dataset_name].mean(),
                                            'cindex_test_std':df_outer_scores['cindex_test_'+dataset_name].std() })
    # IBS
    df_agg_metrics_ibs = pd.DataFrame({'dataset':[dataset_name],
                                            'ibs_test_mean':df_outer_scores['ibs_test_'+dataset_name].mean(),
                                            'ibs_test_std':df_outer_scores['ibs_test_'+dataset_name].std() })
    
    agg_metrics_cindex.append(df_agg_metrics_cindex)
    agg_metrics_ibs.append(df_agg_metrics_ibs)

df_final_eh_tcga_cindex = pd.concat([df for df in agg_metrics_cindex]).round(4)
df_final_eh_tcga_cindex.to_csv(path+'metrics/final_gbdt_tcga_eh_cindex.csv', index=False)
df_final_eh_tcga_ibs = pd.concat([df for df in agg_metrics_ibs]).round(4)
df_final_eh_tcga_ibs.to_csv(path+'metrics/final_gbdt_tcga_eh_ibs.csv', index=False)


# to create summary from already existing data xgbsurv
cancer_types = [
    'BLCA',
    'BRCA',
    'HNSC',
    'KIRC',
    'LGG',
    'LIHC',
    'LUAD',
    'LUSC',
    'OV',
    #'STAD'
    ]

path = '/Users/JUSC/Documents/xgbsurv/experiments/deep_learning/'
agg_metrics_cindex = []
agg_metrics_ibs = []
for name in cancer_types:
    df_outer_scores = pd.read_csv(path+'aft_skorch_results/aft_metric_summary_'+name+'_adapted.csv')
    dataset_name = name #+'_adapted'
    # cindex
    df_agg_metrics_cindex = pd.DataFrame({'dataset':[dataset_name],
                                            'cindex_test_mean':df_outer_scores['cindex_test_'+dataset_name].mean(),
                                            'cindex_test_std':df_outer_scores['cindex_test_'+dataset_name].std() })
    # IBS
    df_agg_metrics_ibs = pd.DataFrame({'dataset':[dataset_name],
                                            'ibs_test_mean':df_outer_scores['ibs_test_'+dataset_name].mean(),
                                            'ibs_test_std':df_outer_scores['ibs_test_'+dataset_name].std() })
    
    agg_metrics_cindex.append(df_agg_metrics_cindex)
    agg_metrics_ibs.append(df_agg_metrics_ibs)

df_final_eh_tcga_cindex = pd.concat([df for df in agg_metrics_cindex]).round(4)
df_final_eh_tcga_cindex.to_csv(path+'metrics/final_dl_tcga_aft_cindex.csv', index=False)
df_final_eh_tcga_ibs = pd.concat([df for df in agg_metrics_ibs]).round(4)
df_final_eh_tcga_ibs.to_csv(path+'metrics/final_dl_tcga_aft_ibs.csv', index=False)

#/Users/JUSC/Documents/xgbsurv/experiments/deep_learning/aft_skorch_results/aft_metric_summary_LIHC_adapted.csv