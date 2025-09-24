import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# ----------------- Data Generator -----------------
def generate_data(n=1000):
    comp = np.clip(np.random.normal(25,8,n),5,50)
    coh = np.clip(np.random.normal(2.5,0.8,n),0.5,6)
    grain = np.clip(np.random.lognormal(-1.5,0.8,n),0.05,2.0)
    dens = np.clip(np.random.normal(850,50,n),750,950)
    flow = np.clip(np.random.lognormal(3.5,0.8,n),10,200)
    perm = np.random.lognormal(2,1.5,n)
    draw = np.clip(np.random.normal(15,5,n),2,35)
    fric = np.clip(np.random.normal(30,5,n),20,45)
    cement = np.random.uniform(0.3,0.9,n)
    clay = np.random.beta(2,5,n)*0.3
    sat = np.random.beta(2,3,n)
    pres = np.clip(np.random.normal(250,50,n),100,400)
    comp_type=np.random.choice(["OpenHole","CasedHole","Gravel_Pack"],n)

    comp_fac=np.where(comp_type=="OpenHole",1.2,np.where(comp_type=="CasedHole",1.0,0.6))
    crit=(comp*coh)/(grain*dens)
    vel=(flow/(perm*draw))/crit
    rock=comp*coh*np.cos(np.radians(fric))
    form=cement*(1-clay)*(1-sat)
    sand=vel*(grain/comp)*(flow/rock)*comp_fac*(1/form)*(draw/pres)*100
    sand=np.clip(sand*np.random.normal(1,0.3,n),0,500)

    return pd.DataFrame({
        "Permeability":perm,"Cohesion":coh,"Grain_Size":grain,"Density":dens,
        "Flow_Rate":flow,"Drawdown":draw,"Friction_Angle":fric,"Cement_Quality":cement,
        "Clay_Content":clay,"Reservoir_Pressure":pres,"Completion":comp_type,
        "Sand_Production":sand
    })

# ----------------- Train & Evaluate -----------------
def train_models(df):
    X,y=df.drop("Sand_Production",axis=1),df["Sand_Production"]
    X["Completion"]=LabelEncoder().fit_transform(X["Completion"])
    X=StandardScaler().fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)

    models={
        "Linear":LinearRegression(),
        "RandomForest":RandomForestRegressor(n_estimators=100,random_state=42),
        "GradientBoost":GradientBoostingRegressor(n_estimators=100,random_state=42)
    }
    results={}
    for name,model in models.items():
        model.fit(Xtr,ytr); yp=model.predict(Xte)
        results[name]={
            "model":model,"yt":yte,"yp":yp,
            "train_r2":r2_score(ytr,model.predict(Xtr)),
            "test_r2":r2_score(yte,yp),
            "rmse":np.sqrt(mean_squared_error(yte,yp)),
            "mae":mean_absolute_error(yte,yp),
            "cv":np.sqrt(-cross_val_score(model,Xtr,ytr,cv=5,
                        scoring="neg_mean_squared_error").mean())
        }
    best=max(results,key=lambda k:results[k]["test_r2"])
    return results,best,Xtr,ytr,df.drop("Sand_Production",axis=1).columns

# ----------------- Feature Importance (always RF) -----------------
def get_feature_importance(results,features):
    rf_model = results["RandomForest"]["model"]
    fi=pd.DataFrame({
        "Feature":features,
        "Importance":rf_model.feature_importances_
    }).sort_values("Importance",ascending=False)
    return fi

# ----------------- Plots -----------------
def plot_results(results,best,feature_importance):
    names=list(results.keys())
    r2=[results[n]["test_r2"] for n in names]
    rmse=[results[n]["rmse"] for n in names]
    yt,yp=results[best]["yt"],results[best]["yp"]

    plt.figure(figsize=(20,15))
    # 1 R²
    plt.subplot(231); plt.bar(names,r2,color="skyblue"); plt.title("R²"); plt.xticks(rotation=45)
    # 2 RMSE
    plt.subplot(232); plt.bar(names,rmse,color="lightcoral"); plt.title("RMSE"); plt.xticks(rotation=45)
    # 3 Actual vs Predicted
    plt.subplot(233); plt.scatter(yt,yp,alpha=0.6); plt.plot([yt.min(),yt.max()],[yt.min(),yt.max()],"r--"); plt.title(f"Actual vs Pred ({best})")
    # 4 Feature Importance (always RF)
    plt.subplot(234); top=feature_importance.head(10); plt.barh(top["Feature"],top["Importance"],color="pink"); plt.title("Top Features (RF)")
    # 5 Residuals
    plt.subplot(235); plt.scatter(yp,yt-yp,alpha=0.6,color="purple"); plt.axhline(0,color="r",ls="--"); plt.title("Residuals")
    # 6 Distribution
    plt.subplot(236); plt.hist(yt,bins=30,alpha=0.6,label="Actual"); plt.hist(yp,bins=30,alpha=0.6,label="Pred"); plt.legend(); plt.title("Distribution")
    plt.tight_layout(); plt.show()

# ----------------- Main -----------------
def main():
    df=generate_data(1000)
    results,best,Xtr,ytr,features=train_models(df)
    fi=get_feature_importance(results,features)

    print("\nMODEL SUMMARY\n","="*60)
    print(f"{'Model':<15}{'TrainR²':<8}{'TestR²':<8}{'RMSE':<8}{'MAE':<8}{'CV':<8}")
    for n,r in results.items():
        print(f"{n:<15}{r['train_r2']:<8.3f}{r['test_r2']:<8.3f}{r['rmse']:<8.2f}{r['mae']:<8.2f}{r['cv']:<8.2f}")
    print(f"\nBest model: {best} (R²={results[best]['test_r2']:.3f})")
    print("\nTop Features (RandomForest):\n",fi.head(5))

    plot_results(results,best,fi)
    df.to_csv("sand_data.csv",index=False)
    fi.to_csv("feature_importance.csv",index=False)
    print("✅ Data & feature importance saved")

if __name__=="__main__":
    main()
