# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:48:25 2023

@author: user
"""


import numpy as np
import pandas as pd
import concurrent.futures
import datetime
import copy

def LS_pred(df):
    data = df
   # data = df[df['CLASS_ID'] == clas][['POST_DATE', 'SKU', 'STORE_NUM', 'DAYOFWEEK_NM', 'DEPT_STORE_GRADE','days_to_EOL', 'div_tot_stock', 'div_dpt_tot_stock','div_dpt_CC_tot_stock', 'div_dpt_class_tot_stock','div_dpt_class_style_tot_stock', 'INV_OH_UT_QN', 'SLS_UT_QN','div_dpt_class_style_color_tot_stock', 'STYLE_ID', 'size_id','COLOR_ID', 'PRES_EFF_DATE', 'PRES_DISC_DATE']]
    data['POST_DATE'] = pd.to_datetime(data['POST_DATE'],format="%Y/%m/%d")
    data['days_started'] = (data['POST_DATE'] - data['PRES_EFF_DATE']).dt.days
    data = data[data['days_started']>=0]
    data['floorset'] = ''
    data['floorset'][data['days_started']<=22] = 'Newness'
    data['floorset'][data['days_to_EOL']<=30] = 'EOL'
    data['floorset'][(data['days_started']>21) & (data['days_to_EOL']>30)] = 'Carryover'
    ls_data = copy.deepcopy(data)
    
    cols = ['SKU','STORE_NUM','DAYOFWEEK_NM','DEPT_STORE_GRADE','size_id','COLOR_ID','STYLE_ID']
    ls_data[cols] = ls_data[cols].astype(str)
    ls_data['SKU_STR'] = ls_data['SKU']+"-"+ls_data["STORE_NUM"]
    ls_data['sale_flag'] = np.where(ls_data['SLS_UT_QN']>0,1,0)
    ls_data['inv_flag'] = np.where(ls_data['INV_OH_UT_QN']>0,1,0)
    ls_data['SLS_UT_QN_Pred'] = ls_data['SLS_UT_QN'].astype(int)
    ls_data['SLS_UT_QN_Pred'] = ls_data['SLS_UT_QN_Pred'].astype(str)
    
    #Removing skus that do not have atleast 2 days of sale in the stores
    l1 = pd.DataFrame(ls_data.groupby(['SKU','STORE_NUM']).sale_flag.sum()).reset_index().rename(columns = {'sale_flag':'sale_ct'})
    l1['SKU_STR'] = l1['SKU']+"-"+l1["STORE_NUM"]
    l1_min2_sale = l1[l1['sale_ct']>1]
#    x = l1[l1['sale_ct']<=1]
    ls_data = ls_data[ls_data['SKU_STR'].isin(l1_min2_sale['SKU_STR'])]
    
    #Identifying the outliers - stores that sold a particular count of units only once in a class
    ls_data['SLS_UT_QN'].value_counts()
    c1 = pd.DataFrame(ls_data.groupby('STORE_NUM').SLS_UT_QN.value_counts()).rename(columns = {'SLS_UT_QN':'sale_ct'}).reset_index()
    c1_only1 = c1[c1['sale_ct']==1]
    c1_only1['str_sl'] = c1_only1['STORE_NUM'] + "-" + c1_only1['SLS_UT_QN'].astype(str)
    ls_data['str_sl'] = ls_data['STORE_NUM'] + "-" + ls_data['SLS_UT_QN'].astype(str)
    #pulling those rows occurence which happened only once in a store - will use it later
    ls_store_sl_only1 = ls_data[ls_data['str_sl'].isin(c1_only1['str_sl'])]
    ls_data = ls_data[~ls_data['str_sl'].isin(c1_only1['str_sl'])]
    
    #Finding the 1st sale day for a SKU-Store
    sale_days_data = ls_data[ls_data['SLS_UT_QN']>0]
    min_sale_day = pd.DataFrame(sale_days_data.groupby(['SKU','STORE_NUM']).POST_DATE.min()).reset_index().rename(columns = {'POST_DATE':'Sale_Start_Date'})
    ls_data = ls_data.merge(min_sale_day,how = 'left',left_on = ['SKU','STORE_NUM'],right_on = ['SKU','STORE_NUM'])
    
    #Splitting out of stock days and good days
    ooo_data = ls_data[(ls_data['INV_OH_UT_QN'] == 0) & (ls_data['SLS_UT_QN'] == 0)]
    ooo_data['STYLE_ID']=ooo_data['STYLE_ID'].astype(str)
    ls_data = ls_data[ls_data['INV_OH_UT_QN']>0]
    
    #Making the probability of selling before the sale has started in a store as 1, which means that though the sku is in presentation, the stock hasn't arrived to the store, so, sale cannot happen in that case, so SLS_QT_UN_Pred will be 0 on those ooo days (by default this is the value in the cell), thus marking the probability to 1
    ooo_good = ooo_data[(ooo_data['floorset'] == 'Newness') & (ooo_data['POST_DATE']<ooo_data['Sale_Start_Date'])]
    ooo_good['probability'] = 1
    ooo_data = ooo_data[~ooo_data.index.isin(ooo_good.index)]
    ooo_good = ooo_good[['POST_DATE', 'SKU', 'STORE_NUM', 'DAYOFWEEK_NM', 'STYLE_ID', 'size_id','COLOR_ID', 'floorset','INV_OH_UT_QN','SLS_UT_QN', 'SLS_UT_QN_Pred', 'probability','SKU_STR']]
    
    sscs_sale = ls_data[['POST_DATE','STYLE_ID','COLOR_ID','size_id','STORE_NUM','DAYOFWEEK_NM','floorset','sale_flag']].drop_duplicates()
    sscs_sale_ct = pd.DataFrame(sscs_sale.groupby(['STYLE_ID','COLOR_ID','size_id','DAYOFWEEK_NM','STORE_NUM','floorset']).sale_flag.sum()).reset_index().rename(columns  = {"sale_flag":"sscs_sale_ct"})
    sscs_avail_df = ls_data[['POST_DATE','STYLE_ID','COLOR_ID','size_id','STORE_NUM','DAYOFWEEK_NM','floorset','inv_flag']].drop_duplicates()
    sscs_avail_ct = pd.DataFrame(sscs_avail_df.groupby(['STYLE_ID','COLOR_ID','size_id','STORE_NUM','DAYOFWEEK_NM','floorset']).inv_flag.sum()).reset_index().rename(columns  = {"inv_flag":"sscs_avail_ct"})
    sscs_df = sscs_sale_ct.merge(sscs_avail_ct,how = "left",on = ['STYLE_ID','COLOR_ID','size_id','STORE_NUM','DAYOFWEEK_NM','floorset'])
    sscs_df['sscs_sale_prob'] = sscs_df['sscs_sale_ct']/sscs_df['sscs_avail_ct']
    
    cols = ['POST_DATE','SKU','STORE_NUM', 'DAYOFWEEK_NM','STYLE_ID', 'size_id','COLOR_ID','floorset','INV_OH_UT_QN','SLS_UT_QN','SKU_STR']
    #To get the post_date and number of units sold for each occurrence of such combinations of sale in the past, we need to merge it with the ls_data
    op = sscs_df.merge(ls_data[cols],how = 'left',on = ['STYLE_ID','COLOR_ID','size_id','DAYOFWEEK_NM','STORE_NUM','floorset'])
    #Aggregating to find the average number of units sold in the past
    op1 = pd.DataFrame(op.groupby(['STYLE_ID', 'COLOR_ID', 'size_id', 'DAYOFWEEK_NM', 'STORE_NUM','floorset']).agg({'SLS_UT_QN':'mean', 'sscs_sale_prob':'mean'})).reset_index().rename(columns  = {"SLS_UT_QN":"SLS_UT_QN_Pred","sscs_sale_prob":"probability"})
    
    #ooo_cols = ['POST_DATE','SKU','STORE_NUM', 'DAYOFWEEK_NM','STYLE_ID', 'size_id','COLOR_ID','floorset','SLS_UT_QN','INV_OH_UT_QN','SKU_STR']
    #Now, in the out of stock days, if there is any event that has occurred same like in the past, we will give the probability and average sale count to that instance - so, we need to merge ooo data and the aggregated op1 data
    ooo_op = ooo_data[cols].merge(op1,how = 'left',on = ['STORE_NUM', 'DAYOFWEEK_NM','STYLE_ID', 'size_id','COLOR_ID','floorset'])
    ooo_op['SLS_UT_QN_Pred']=round(ooo_op['SLS_UT_QN_Pred'])
    #Since we found the probability to sell and then joined the sls_units num, if prob to sell = 0 it means that we are confident that sale will not happen or sls_ut_qn = 0
    #So, i tried to show the probability for sls_ut_qn = 0 will be 1-probability to sell
    ooo_op['probability'] = np.where((ooo_op['SLS_UT_QN_Pred'] == 0) ,1-ooo_op['probability'],ooo_op['probability'])
    ooo_good1 = ooo_op.dropna()
    ooo_good = pd.concat([ooo_good,ooo_good1])
    #Below instances are those which have not occurred in the past
    ooo_unknown = ooo_op[ooo_op['SLS_UT_QN_Pred'].isna()]
    
    #To find probability of unknown instances, We are trying to check if there is a sale for the same product irrespective of its color
    size_sale_df = ls_data[['POST_DATE','STORE_NUM','STYLE_ID','size_id','DAYOFWEEK_NM','floorset','sale_flag']].drop_duplicates()
    size_sale_ct = pd.DataFrame(size_sale_df.groupby(['STORE_NUM','STYLE_ID','size_id','floorset','DAYOFWEEK_NM']).sale_flag.sum()).reset_index().rename(columns  = {"sale_flag":"size_sale_ct"})
    size_avail_df = ls_data[['POST_DATE','STYLE_ID','size_id','STORE_NUM','DAYOFWEEK_NM','floorset','inv_flag']].drop_duplicates()
    size_avail_ct = pd.DataFrame(size_avail_df.groupby(['STORE_NUM','STYLE_ID','size_id','floorset','DAYOFWEEK_NM']).inv_flag.sum()).reset_index().rename(columns  = {"inv_flag":"size_avail_ct"})
    size_df = size_sale_ct.merge(size_avail_ct,how = "left",on = ['STORE_NUM','STYLE_ID','size_id','floorset','DAYOFWEEK_NM'])
    size_df['size_sale_prob'] = size_df['size_sale_ct']/size_df['size_avail_ct']
    
    #To get the post_date and number of units sold for each occurrence of such combinations of sale in the past, we need to merge it with the ls_data
    cols = ['POST_DATE','SKU','STORE_NUM', 'DAYOFWEEK_NM','STYLE_ID', 'size_id','COLOR_ID','floorset','INV_OH_UT_QN','SLS_UT_QN','SKU_STR']
    op_size = size_df.merge(ls_data[cols],how = 'left',on = ['STYLE_ID','size_id','DAYOFWEEK_NM','STORE_NUM','floorset'])
    #Aggregating to find the average number of units sold in the past
    op2 = pd.DataFrame(op_size.groupby(['STYLE_ID', 'size_id', 'DAYOFWEEK_NM', 'STORE_NUM','floorset']).agg({'SLS_UT_QN':'mean', 'size_sale_prob':'mean'})).reset_index().rename(columns  = {"SLS_UT_QN":"SLS_UT_QN_Pred","size_sale_prob":"probability"})
    #Now, in the out of stock days, if there is any event that has occurred same like in the past, we will give the probability and average sale count to that instance - so, we need to merge ooo data and the aggregated op1 data
    ooo_u = ooo_unknown[cols].merge(op2,how = 'left',on = ['STORE_NUM', 'DAYOFWEEK_NM','STYLE_ID', 'size_id','floorset'])
    ooo_u['SLS_UT_QN_Pred']=round(ooo_u['SLS_UT_QN_Pred'])
    #Since we found the probability to sell and then joined the sls_units num, if prob to sell = 0 it means that we are confident that sale will not happen or sls_ut_qn = 0
    #So, i tried to show the probability for sls_ut_qn = 0 will be 1-probability to sell
    ooo_u['probability'] = np.where((ooo_u['SLS_UT_QN_Pred'] == 0) ,1-ooo_u['probability'],ooo_u['probability'])
    ooo_unknown_good = ooo_u.dropna()
    ooo_good = pd.concat([ooo_good,ooo_unknown_good])
    #Below are the instances which say that in the past, there is no such occurence in the past on that particular day and floorset
    ooo_unknown1 = ooo_u[ooo_u['SLS_UT_QN_Pred'].isna()]
    
    #To find probability of unknown1 instances, We are trying to check if there is a sale for the same style on the same day and floorset and find the average sale
    style_sale = ls_data[['POST_DATE','STORE_NUM','STYLE_ID','floorset','DAYOFWEEK_NM','sale_flag']].drop_duplicates()
    style_sale_ct = pd.DataFrame(style_sale.groupby(['STORE_NUM','STYLE_ID','floorset','DAYOFWEEK_NM']).sale_flag.sum()).reset_index().rename(columns  = {"sale_flag":"style_sale_ct"})
    style_avail_df = ls_data[['POST_DATE','STORE_NUM','STYLE_ID','floorset','DAYOFWEEK_NM','inv_flag']].drop_duplicates()
    style_avail_ct = pd.DataFrame(style_avail_df.groupby(['STORE_NUM','STYLE_ID','floorset','DAYOFWEEK_NM']).inv_flag.sum()).reset_index().rename(columns  = {"inv_flag":"style_avail_ct"})
    style_df = style_sale_ct.merge(style_avail_ct,how = "left",on = ['STORE_NUM','STYLE_ID','floorset','DAYOFWEEK_NM'])
    style_df['style_sale_prob'] = style_df['style_sale_ct']/style_df['style_avail_ct']
    #To get the post_date and number of units sold for each occurrence of such combinations of sale in the past, we need to merge it with the ls_data
    #cols = ['POST_DATE','SKU','STORE_NUM', 'DAYOFWEEK_NM','STYLE_ID', 'floorset','INV_OH_UT_QN','SLS_UT_QN','SKU_STR']
    op_style = style_df.merge(ls_data[cols],how = 'left',on = ['STORE_NUM','STYLE_ID','floorset','DAYOFWEEK_NM'])
    #Aggregating to find the average number of units sold in the past
    op3 = pd.DataFrame(op_style.groupby(['STORE_NUM','STYLE_ID', 'floorset', 'DAYOFWEEK_NM']).agg({'SLS_UT_QN':'mean', 'style_sale_prob':'mean'})).reset_index().rename(columns  = {"SLS_UT_QN":"SLS_UT_QN_Pred","style_sale_prob":"probability"})
    
    #Now, in the out of stock days, if there is any event that has occurred same like in the past, we will give the probability and average sale count to that instance - so, we need to merge ooo data and the aggregated op1 data
    ooo_unknown1 = ooo_unknown1.merge(op3,how = 'left',on = ['STORE_NUM','STYLE_ID','floorset','DAYOFWEEK_NM']).drop(columns = ['SLS_UT_QN_Pred_x', 'probability_x']).rename(columns  = {"SLS_UT_QN_Pred_y":"SLS_UT_QN_Pred","probability_y":"probability"})
    ooo_unknown1_good = ooo_unknown1.dropna()
    ooo_good = pd.concat([ooo_good,ooo_unknown1_good])
    #Below are the instances which say that in the past, there is no such occurence of style sale in the past on that particular day and floorset
    ooo_unknown2 = ooo_unknown1[ooo_unknown1['SLS_UT_QN_Pred'].isna()]
    
    #To find probability of unknown2 instances, We are trying to check if there is a sale for the same SKU on the same day in the previous floorset
    #Replacing Carryover - EOL;  Newness - Carryover in the bkp so that EOL and carryover in the unknown2 dataset match to the newly formed bkp dataset
    op_bkp = copy.deepcopy(op1)
    op_bkp = op_bkp[(op_bkp['floorset']=='Newness') | (op_bkp['floorset']=='Carryover')]
    op_bkp = op_bkp.replace({'floorset' : { 'Carryover' : 'EOL', 'Newness' :'Carryover' }})
    ooo_unknown2 = ooo_unknown2.merge(op_bkp,how = 'left',on = ['STORE_NUM', 'STYLE_ID','COLOR_ID', 'size_id','floorset','DAYOFWEEK_NM']).drop(columns = ['probability_x','SLS_UT_QN_Pred_x']).rename(columns = {'probability_y':'probability','SLS_UT_QN_Pred_y':'SLS_UT_QN_Pred'})
    ooo_unknown2['SLS_UT_QN_Pred']=round(ooo_unknown2['SLS_UT_QN_Pred'])
    ooo_unknown2['probability'] = np.where((ooo_unknown2['SLS_UT_QN_Pred'] == 0) ,1-ooo_unknown2['probability'],ooo_unknown2['probability'])
    ooo_unknown2_good = ooo_unknown2.dropna()
    ooo_good = pd.concat([ooo_good,ooo_unknown2_good])
    #Below are the instances which say that in the past, there is no such occurence in the past on any day in the floorset
    ooo_unknown3 = ooo_unknown2[ooo_unknown2['SLS_UT_QN_Pred'].isna()]
    
    #The unknown instances happened because these are the SKUs which have very low sales  and thus assuming that the probability of selling is also very low. Thus making the sale_ct = 0 and probability = 1
    ooo_unknown3['SLS_UT_QN_Pred'] = 0 
    ooo_unknown3['probability'] = 1 
    ooo_good = pd.concat([ooo_good,ooo_unknown3])
    ooo_good['SLS_UT_QN_Pred'] = ooo_good['SLS_UT_QN_Pred'].astype(int)
    
    #Identifying the days of ooo where prediction is more than the SLS_UT_QN in the past and replacing it with the max sale observed in the past
    ls_data_sale_max = pd.DataFrame(ls_data.groupby(['SKU','STORE_NUM']).SLS_UT_QN.max()).reset_index().rename(columns = {"SLS_UT_QN":"SLS_UT_QN_max"})
    ooo_good = ooo_good.merge(ls_data_sale_max,how='left',on = ['SKU','STORE_NUM'])
    ooo_good['SLS_UT_QN_Pred'] = np.where(ooo_good['SLS_UT_QN_Pred']>ooo_good['SLS_UT_QN_max'],ooo_good['SLS_UT_QN_max'],ooo_good['SLS_UT_QN_Pred'])
    ooo_good['SLS_UT_QN_Pred'] = ooo_good['SLS_UT_QN_Pred'].astype(str)
    ooo_good = ooo_good.drop(columns = ['SLS_UT_QN_max'])
    
    cols = ['POST_DATE', 'SKU', 'STORE_NUM', 'DAYOFWEEK_NM', 'STYLE_ID', 'size_id','COLOR_ID', 'floorset','INV_OH_UT_QN', 'SLS_UT_QN','SLS_UT_QN_Pred','SKU_STR']
    anp = pd.concat([ls_data[cols],ls_store_sl_only1[cols],ooo_good[cols]])
    anp['SLS_UT_QN_Pred'] = anp['SLS_UT_QN_Pred'].astype(float)
    anp['SLS_UT_QN_Pred'] = anp['SLS_UT_QN_Pred'].astype(int)
    anp['SLS_UT_QN_Pred'] = anp['SLS_UT_QN_Pred'].astype(str)
    
    ans = anp.merge(ooo_good,how = "left",left_on = ['POST_DATE', 'SKU', 'STORE_NUM'],right_on = ['POST_DATE', 'SKU', 'STORE_NUM']).drop(columns = ['DAYOFWEEK_NM_y','STYLE_ID_y', 'size_id_y', 'COLOR_ID_y', 'floorset_y', 'INV_OH_UT_QN_y','SLS_UT_QN_y', 'SLS_UT_QN_Pred_y','SKU_STR_y']).rename(columns = {'DAYOFWEEK_NM_x':'DAYOFWEEK_NM', 'STYLE_ID_x':'STYLE_ID', 'size_id_x':'size_id', 'COLOR_ID_x':'COLOR_ID', 'floorset_x':'floorset', 'INV_OH_UT_QN_x':'INV_OH_UT_QN','SLS_UT_QN_x':'SLS_UT_QN', 'SLS_UT_QN_Pred_x':'SLS_UT_QN_Pred', 'SKU_STR_x':'SKU_STR'})
    return ans


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    start_time = datetime.datetime.now()
    data = pd.read_csv('source file1') 
    sku_info = pd.read_csv('source file2')
    #    sku_info1 = copy.deepcopy(sku_info)
    dates = pd.read_csv('source_dates_file')
    dates[['PRES_EFF_DATE','PRES_DISC_DATE']] = dates[['PRES_EFF_DATE','PRES_DISC_DATE']].apply(pd.to_datetime)
    data['INV_OH_UT_QN'] = np.where(data['INV_OH_UT_QN']<data['SLS_UT_QN'],data['SLS_UT_QN'],data['INV_OH_UT_QN'])

    #Merging PRES_EFF and PRES_DISC dates to the data
    data = data.merge(sku_info,how = 'left',left_on=(['SKU']),right_on=(['sku'])).drop(columns = ['sku'])
    data = data.merge(dates,how = 'left',left_on = ['SKU','STORE_NUM'],right_on = ['sku','store_num']).drop(columns = ['sku','store_num'])   
    res = executor.submit(LS_pred, data)
end_time = datetime.datetime.now()
run_time = end_time-start_time

df = res.result()
