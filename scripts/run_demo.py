"""Run the complete synthetic public demonstration workflow."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_generation import save_demo_data
from src.machine_learning import evaluate_stress_classifier,evaluate_flower_regressor
from src.nutrient_analysis import treatment_summary,late_season_k_summary

def save_svg(fig,name):
    out=ROOT/'figures'/name; out.parent.mkdir(exist_ok=True); fig.savefig(out,format='svg',bbox_inches='tight'); plt.close(fig); return out

def main():
    data_path=ROOT/'data'/'sample_dragon_fruit_demo.csv'; save_demo_data(data_path,42); df=pd.read_csv(data_path)
    _,cmets,cm,importance=evaluate_stress_classifier(df); _,rmets,pred=evaluate_flower_regressor(df)
    rd=ROOT/'results'; rd.mkdir(exist_ok=True)
    rows=[{'model':cmets['model'],'target':cmets['target'],'validation':cmets['validation'],'accuracy':round(cmets['accuracy'],4),'precision':round(cmets['precision'],4),'recall':round(cmets['recall'],4),'f1':round(cmets['f1'],4),'mae':'','r2':''},{'model':rmets['model'],'target':rmets['target'],'validation':rmets['validation'],'accuracy':'','precision':'','recall':'','f1':'','mae':round(rmets['mae'],4),'r2':round(rmets['r2'],4)}]
    pd.DataFrame(rows).to_csv(rd/'model_metrics.csv',index=False); importance.head(25).to_csv(rd/'feature_importance.csv',index=False); treatment_summary(df).to_csv(rd/'treatment_summary.csv',index=False); late_season_k_summary(df).to_csv(rd/'late_season_k_summary.csv',index=False)
    po=df[['plant_id','dap','flower_count']].copy(); po['predicted_flower_count_cv']=np.round(pred,3); po.to_csv(rd/'flower_predictions_grouped_cv.csv',index=False)
    s=df.groupby(['dap','species'],as_index=False)['flower_count'].mean(); fig,ax=plt.subplots(figsize=(8,4.8)); [ax.plot(g['dap'],g['flower_count'],marker='o',label=k) for k,g in s.groupby('species')]; ax.set(title='Demo flowering trajectory by species',xlabel='Days after planting',ylabel='Mean flower count'); ax.legend(); save_svg(fig,'phenology_time_series.svg')
    fig,ax=plt.subplots(figsize=(7,5)); [ax.scatter(g['tissue_n_pct'],g['ndre'],alpha=.65,s=25,label='Stress flag' if k==1 else 'Lower-risk') for k,g in df.groupby('stress_flag')]; ax.set(title='Demo relationship: tissue N and NDRE',xlabel='Tissue N (%)',ylabel='NDRE'); ax.legend(); save_svg(fig,'nutrient_spectral_relationship.svg')
    fig,ax=plt.subplots(figsize=(5.5,4.8)); im=ax.imshow(cm); [ax.text(j,i,int(v),ha='center',va='center') for (i,j),v in np.ndenumerate(cm)]; ax.set(title='Stress classifier grouped-CV confusion matrix',xlabel='Prediction',ylabel='Observed'); fig.colorbar(im,ax=ax); save_svg(fig,'confusion_matrix.svg')
    top=importance.head(12).iloc[::-1]; fig,ax=plt.subplots(figsize=(8,5.5)); ax.barh(top['feature'],top['importance']); ax.set(title='Top demo features for stress classification',xlabel='Random Forest importance'); save_svg(fig,'feature_importance.svg')
    ts=df.groupby(['environment','treatment_t_acre'],as_index=False)['flower_count'].mean(); fig,ax=plt.subplots(figsize=(7.5,4.8)); [ax.plot(g['treatment_t_acre'],g['flower_count'],marker='o',label=k) for k,g in ts.groupby('environment')]; ax.set(title='Demo treatment response by production environment',xlabel='Treatment (t/acre)',ylabel='Mean flower count'); ax.legend(); save_svg(fig,'treatment_response.svg')
    fig,ax=plt.subplots(figsize=(6.7,4.5)); vals=[cmets['accuracy'],cmets['f1'],max(rmets['r2'],0)]; bars=ax.bar(['Accuracy','F1','R²'],vals); ax.set_ylim(0,1); ax.set(title='Grouped cross-validation performance',ylabel='Score'); [ax.text(b.get_x()+b.get_width()/2,v+.02,f'{v:.2f}',ha='center') for b,v in zip(bars,vals)]; save_svg(fig,'model_performance.svg')
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
