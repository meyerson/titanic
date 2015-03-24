import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe
test_df = pd.read_csv('test.csv', header=0)
verify_df =  pd.read_csv('titanic.csv', header=0)
train_names = train_df['Name'].values

test_names = test_df['Name'].values
ids = test_df['PassengerId'].values

print train_df.shape, test_df.shape, verify_df.shape

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

def prep_data(df,cols=None,neighbor=True):
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = 'N/A'

    df.Cabin = df.Cabin.fillna('N0')
    df['CabinN'] = df.Cabin.apply(lambda x: str(x)[1::])
    df.CabinN = df.CabinN.convert_objects(convert_numeric=True)
    
    df.Cabin = df.Cabin.apply(lambda x: str(x)[0])
    
    df['AgeNULL'] = (df.Age.isnull()).astype(int)
    df['Child'] = (df.Age < 2).astype(int)
    df['Family'] = ((df.SibSp*df.Parch) >0).astype(int)
    
    df['Stowaway']  = (df.AgeNULL*(df.CabinN==0)) == 1


    def ticket_parse(ticket_str):
        ticket = str(ticket_str).split()
        if len(ticket) == 1:
            return ['NA',ticket[0]]
        else:
            #return ticket
            return [ticket[0].replace(".", "").replace("A/","A"), ticket[-1]]
        
    df['Ticket_pre'] = df.Ticket.apply(lambda x: ticket_parse(x)[0])   
    df['Ticket'] = df.Ticket.apply(lambda x: ticket_parse(x)[1])
    
    df.Ticket = df.Ticket.convert_objects(convert_numeric=True).fillna(0)
    #roughly next (often same) door coupling - highly collinear
    
    droplist= ['Name', 'Sex','PassengerId']
    cat_to_dums = ['Embarked','Cabin','Family','Ticket_pre']
    #join on the original index by default!
    if neighbor:
        from scipy.spatial.distance import pdist, squareform,cdist
        #compute a distance between 
        rsuffix='_next'
       
       
        import copy

        
        def look_for_dead(g):

            new_cols = [x+'_'+y for x in ['near','far','mean'] for y in ['dead','live']]
            
            if len(g)==1:
                for c in new_cols:
                    g[c] = np.NaN
              

                return g
            
            a_live,a_die,a_test = g[g.Survived==1],g[g.Survived==0],g[g.Survived=='?']

            dist = cdist(g['CabinN'][:,None],a_die['CabinN'][:,None])
            m_dist = np.ma.masked_array(dist,np.isnan(dist),fill_value=np.NaN)
            
            m_min = np.array(m_dist.min(axis=1))
            m_max =  np.array(m_dist.max(axis=1))
            m_mean = np.array(m_dist.mean(axis=1))
            m_min[m_min== 1.00000000e+20] = np.NaN
            m_max[m_max== 1.00000000e+20] = np.NaN
            m_mean[m_mean== 0.] = np.NaN
            
            
            g['near_dead'] =m_min
            g['mean_dead'] = m_mean
            g['far_dead'] = m_max

            
          
            dist = cdist(g['CabinN'][:,None],a_live['CabinN'][:,None])
            m_dist = np.ma.masked_array(dist,np.isnan(dist))
            
            m_min = np.array(m_dist.min(axis=1))
            m_max =  np.array(m_dist.max(axis=1))
            m_mean = np.array(m_dist.mean(axis=1))
            m_min[m_min== 1.00000000e+20] = np.NaN
            m_mean[m_mean== 0.] = np.NaN
            m_max[m_max== 1.00000000e+20] = np.NaN
            
            g['mean_live'] =  m_dist.mean(axis=1)
            g['near_live'] = m_dist.min(axis=1)
            g['far_live'] = m_max

            
            return g

            
        df = df.groupby('Cabin').apply(look_for_dead)    
        
        
       

        df = df.join((df.sort('CabinN').groupby('Cabin').shift(1))[['Pclass','Family','Embarked','Survived']] ,rsuffix=rsuffix)
       
        cat_to_dums =cat_to_dums + map(lambda x: x+rsuffix,['Family','Embarked'])
        #droplist= droplist+ map(lambda x: x+rsuffix,droplist)
        
   
        
    
    df = pd.get_dummies(df.iloc[:],columns=cat_to_dums,dummy_na =False).fillna(0)

    median_age = df['Age'].dropna().median()
    
    if len(df.Age[ df.Age.isnull() ]) > 0:
        df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

    

    #df = df.drop(['Name', 'Sex', 'Ticket','PassengerId'], axis=1)
    #df = df.drop(['Name_1', 'Sex_1', 'Ticket_1','PassengerId_1'], axis=1)
    print 'drolist: ', droplist
    df = df.drop(droplist, axis=1)
    
    #df = df.drop(['Name', 'Sex', 'Ticket','PassengerId','Name_1', 'Sex_1', 'Ticket_1','PassengerId_1'], axis=1) 

    #create a bunch of composite vars

    
    
    

    
    if cols != None:
        print 'cols:',cols
        for col in cols:
            if col not in df.columns:
                df[col] = 0
                
        for new_col in df.columns:
            if new_col not in cols:
                
                df = df.drop(new_col,axis = 1)

    print df.columns
    return df
## create a join set
test_df['Survived'] = '?'
all_df = train_df.append(test_df,ignore_index=True)
all_df = prep_data(all_df)

train_df = all_df[all_df.Survived!='?']
test_df = all_df[all_df.Survived=='?']
print test_df.columns
test_df = test_df.drop('Survived',axis=1)
#exit()

#######################

#train_df = train_df.drop(['Name', 'Sex', 'Ticket','PassengerId'], axis=1)
#train_df = prep_data(train_df)


train_cols = (set(train_df.columns))
#train_cols.remove('Survived')

#######################



#test_df['Survived'] = '?' #a.t.m this gets droped in prep_data

#test_df = prep_data(test_df,cols = train_cols) #any features NOT in train_cols get dropped


#test_names = test_df['Name'].values
#test_df = test_df.drop(['Name', 'Sex', 'Ticket','PassengerId'], axis=1)


######################
train_data = train_df.values
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=10000,n_jobs = -1,
                                max_features='auto',criterion='gini')
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )
print 'training set  . . ',forest.feature_importances_

print 'Predicting...'
output = forest.predict(test_data).astype(int)

print type(output)
#output_df = pd.read_csv('.csv', header=0)
print 'Testing'
#join on name?
print test_df.columns
#reattach Names - otherwise its hard to id passengers

test_df['name'] = test_names
train_df['name'] = train_names
test_df['Survived'] = output

merged_test = pd.merge(test_df,verify_df,on='name')

print np.float(len(merged_test['Survived'])-np.sum(np.abs(merged_test['Survived'] -  merged_test['survived'])))/len(merged_test['Survived'])

print np.sum(np.abs(merged_test['Survived'] -  merged_test['survived']))


predictions_file = open("basic_RF.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
