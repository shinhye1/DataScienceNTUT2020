import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("studentsdata2014.csv")
df.head(5)

df['DegreeF']=df['Degree Students-Officially Studying Degrees Foreign Students']
df['DegreeN']=df['Degree Students-Officially Studying Degrees Land Students']
df['NonDegreeF']=df['Non-degree Students-Foreign Exchange Students']+df['Non-degree Students-Foreign Short-Term Study and personal selection']+df['Non-degree students-college students with Chinese Language Center']
df['DegreeO']=df['Degree Students-Overseas Students (including Hong Kong and Macao)']+df['Overseas special classe']
df['NonDegreeN']=df['Non-degree students-Haiqing class']+df['Non-degree students-mainland students']
col=['Continent','DegreeF','DegreeN','DegreeO','NonDegreeF','NonDegreeN','Total']
df2=df[col]
df2=df2.groupby(['Continent'])['DegreeF','DegreeN','DegreeO','NonDegreeF','NonDegreeN','Total'].sum()
df2.head()

bin_labels = ['LowDegreeF','MedDegreeF','TopDegreeF']
df['DegreeF'] = pd.cut(df['DegreeF'],bins=3,labels=bin_labels)
bin_labels = ['LowDegreeN','MedDegreeN','TopDegreeN']
df['DegreeN'] = pd.cut(df['DegreeN'],bins=3,labels=bin_labels)
bin_labels = ['LowDegreeO','MedDegreeO','TopDegreeO']
df['DegreeO'] = pd.cut(df['DegreeO'],bins=3,labels=bin_labels)
bin_labels = ['LowNonDegreeF','MedNonDegreeF','TopNonDegreeF']
df['NonDegreeF'] = pd.cut(df['NonDegreeF'],bins=3,labels=bin_labels)
bin_labels = ['LowNonDegreeN','MedNonDegreeN','TopNonDegreeN']
df['NonDegreeN'] = pd.cut(df['NonDegreeN'],bins=3,labels=bin_labels)
col1=['DegreeF','DegreeN','DegreeO','NonDegreeF','NonDegreeN']
df1=df[col1]
df1.head(5)

from mlxtend.preprocessing import TransactionEncoder
te=TransactionEncoder()
te_new=te.fit(df1.values).transform(df1.values)
df_new=pd.DataFrame(te_new, columns=te.columns_)
df_new.head(5)

from mlxtend.frequent_patterns import fpgrowth, association_rules
ans=fpgrowth(df_new,min_support=0.97, use_colnames=True)
ans.head(5)

fp=association_rules(ans,metric='lift',min_threshold=1)
fp=fp.sort_values(['confidence','lift'],ascending=[False, False])
fp.head(15)

from mlxtend.frequent_patterns import apriori
ans1=apriori(df_new,min_support=0.97, use_colnames=True)
ans1.head(5)

ap=association_rules(ans1,metric='lift',min_threshold=1)
ap=ap.sort_values(['confidence','lift'],ascending=[False, False])
ap.head(15)