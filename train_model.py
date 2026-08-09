from pathlib import Path
import json, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data_processed/cleaned_startups.csv")
target="Valuation_USD_Millions"
X=df.drop(columns=["Company_ID",target]); y=df[target]
cat=X.select_dtypes("object").columns.tolist(); num=X.select_dtypes(exclude="object").columns.tolist()
pre=ColumnTransformer([
 ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
models={
 "Linear Regression":LinearRegression(),
 "Random Forest":RandomForestRegressor(n_estimators=100,max_depth=14,min_samples_leaf=2,n_jobs=-1,random_state=42)
}
results={}; fitted={}
for name,m in models.items():
    pipe=Pipeline([("preprocessor",pre),("model",m)])
    pipe.fit(Xtr,ytr); p=pipe.predict(Xte)
    results[name]={"MAE":float(mean_absolute_error(yte,p)),"RMSE":float(mean_squared_error(yte,p)**.5),"R2":float(r2_score(yte,p))}
    fitted[name]=pipe
best=max(results,key=lambda k:results[k]["R2"])
out=ROOT/"outputs"; (out/"model").mkdir(parents=True,exist_ok=True)
joblib.dump(fitted[best],out/"model/valuation_model.joblib")
(out/"model_metrics.json").write_text(json.dumps({"best_model":best,"metrics":results},indent=2))
print(json.dumps({"best_model":best,"metrics":results},indent=2))
