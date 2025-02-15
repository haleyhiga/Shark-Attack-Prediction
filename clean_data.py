import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import string

df = pd.read_csv("attacks.csv", encoding='latin1')

df.drop(["Unnamed: 22", "Unnamed: 23", "Case Number", "Date", "Year", "Area", "Location", "Name", "Injury",
         "Investigator or Source", "pdf", "href formula", "href", 
         "Case Number.1", "Case Number.2", "Time", "original order", "Country"], axis=1, inplace=True)

df = df.dropna(how="any")
df.rename(columns={"Species ": "Species"}, inplace=True)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\d', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [re.sub(r'(.)\1{1,}', r'\1\1', word) for word in words]
    words = [word.strip() for word in words if len(word.strip()) > 1]
    
    text = " ".join(words)
    return text

df["Species"] = df["Species"].apply(preprocess_text)







# Clean species data
df["Species"] = df["Species"].str.replace("to", "").str.strip()

df["Species"] = df["Species"].str.replace("to", "").str.strip()
df["Species"] = df["Species"].str.replace("less than", "").str.strip()
df["Species"] = df["Species"].str.replace("possibly", "").str.strip()
df["Species"] = df["Species"].str.replace("small", "").str.strip()


df["Species"] = df["Species"].str.replace("tall", "").str.strip()
df["Species"] = df["Species"].str.replace("brown", "").str.strip()
df["Species"] = df["Species"].str.replace("cm", "").str.strip()
df["Species"] = df["Species"].str.replace("greycolored", "").str.strip()


def or_split(text):
    if "or" in text:
        return text.split("or")[0].strip()
    else:
        return text

df["Species"] = df["Species"].apply(or_split)

def len_strip(text):
    if len(text.split()) > 2:
        words = text.split()[:2]
        text = " ".join(words)
        return text
    else:
        return text


df["Species"] = df["Species"].apply(len_strip)


df["Species"] = df["Species"].replace({
 'shark': 'unknown',
 'juvenile blacktip': 'blacktip shark',
 'blacktip': 'blacktip shark',
 'tp shark': 'unknown',
 'whitetip reef': 'reef shark',
 'thought involve': 'unknown',
 'shark  blactip': 'blacktip shark',
 'p': 'unknown',
 'shark involvement': 'unknown',
 '': 'unknown',
 'yearold pound': 'unknown',
 'shark tiger': 'tiger shark',
 'spinner': 'spinner shark',
 'grey reef': 'reef shark',
 'juvenile white': 'white shark',
 'unknown but': 'unknown',
 'unidentified species': 'unknown',
 '\x93  shark\x94': 'unknown',
 '\x93\x94 shark': 'unknown',
 'bite diameter': 'unknown',
 'species unidentified': 'unknown',
 'grey col': 'unknown',
 'hammerhead': 'hammerhead shark',
 'shark bull': 'bull shark',
 'shark oth': 'unknown',
 'questionable incident': 'unknown',
 'large white': 'white shark',
 'shark  blacktip': 'blacktip shark',
 'shark was': 'unknown',
 'shark probably': 'unknown',
 'sh': 'unknown',
 'shark seen': 'unknown',
 'shark spinner': 'spinner shark',
 'young shark': 'unknown',
 'unidentified shark': 'unknown',
 'said involve': 'unknown',
 'bull': 'bull shark',
 'lb shark': 'unknown',
 'sixgill': 'unknown',
 'hammerhead sharko': 'hammerhead shark',
 'bull sandbar': 'bull shark',
 'maculpinnis': 'unknown',
 'local auth': 'unknown',
 'hand found': 'unknown',
 'h': 'unknown',
 'larger shark': 'unknown',
 'rep': 'unknown',
 'leucas': 'unknown'
})



df.dropna(subset=["Species"], inplace=True)
df.dropna(subset=["Activity"], inplace=True)

def activities(text):
    activities = ["Surfing", "Swimming", "Wading", "Spearfishing", "Standing", "Fishing", "Snorkeling", "Scuba diving", "boarding"]
    for activity in activities:
        if activity.lower() in text.lower():
            return activity
    return "Other"

def shark_species(text):
    shark_species = ["unknown", "white shark", "tiger shark", "bull shark", "blacktip shark", "bronze whaler", "nurse shark", "raggedoth shark", "wobbegong shark", "grey nurse"]
    for species in shark_species:
        if species.lower() in text.lower():
            return species
    return "Other"

df["Activity"] = df["Activity"].apply(activities)
df["Species"] = df["Species"].apply(shark_species)

df.rename(columns={'Fatal (Y/N)': 'Fatal'}, inplace=True)
df.rename(columns={'Sex ': 'Sex'}, inplace=True)

df.dropna(subset=["Type"], inplace=True)
df["Type"] = df["Type"].replace({"boat": np.nan})


# plot activity histogram
#plt.figure(figsize=(15, 7))
#sns.countplot(x="Activity", data=df, order=df["Activity"].value_counts().index)
#plt.show()
#print(df.head())
# df.Age = df.Age.fillna(0).astype(str)
# df.Age = df.Age.apply( lambda z: z[:2])
# df.Age = df.Age.astype(str)
# def fix_age(n):
#     try:
#         return int(n)
#     except:
#         return 0
# df.Age = df.Age.apply(fix_age)
# df.Age.value_counts()


df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# Drop rows with missing age values
df.dropna(subset=["Age"], inplace=True)

# Convert the 'Age' column back to integers
df["Age"] = df["Age"].astype('Int64')

""" 


def age_group(age):
    if age < 10:
        return "0-10"
    elif age < 20:
        return "10-20"
    elif age < 30:
        return "20-30"
    elif age < 40:
        return "30-40"
    elif age < 50:
        return "40-50"
    elif age < 60:
        return "50-60"
    elif age < 70:
        return "60-70"
    elif age < 80:
        return "70-80"
    else:
        return "80+"

df["Age Group"] = df["Age"].apply(age_group) 
age_group_counts = df['Age Group'].value_counts().sort_index()

 """

df["Sex"] = df["Sex"].str.strip()

# Replace "N" values with NaN
df["Sex"] = df["Sex"].replace({"N": np.nan})

# Replace "lli" with "F"
df["Sex"] = df["Sex"].replace({"lli": "F"})
df.dropna(subset=["Sex"], inplace=True)


# plt.show()
df["Fatal"] = df["Fatal"].str.strip()
df["Fatal"] = df["Fatal"].replace({"UNKNOWN": np.nan, "2017": np.nan, "y": np.nan, "M": np.nan})
df.dropna(subset=["Fatal"], inplace=True)
#print(df.head())


#df.drop(columns=["Age Group"], inplace=True)

#print(df["Species"].value_counts().head(10))
#print(df.head())

df = df[["Activity", "Sex", "Age", "Type", "Species", "Fatal"]]
#print(df.head())


'''
WRITE TO NEW CSV FILE
'''

#df.to_csv("new.csv", index=False)



#
#  GRAPH CREATION BELOW
#


# PLOT AGE GROUPSS
#age_fatal_counts = df.groupby(['Age Group', 'Fatal']).size().unstack(fill_value=0)

#age_fatal_counts.plot(kind='bar', color=["#F5A8E3", '#A931E5'], figsize=(10, 6))
#plt.title('Shark Attacks By Age Group')
#plt.xlabel('Age Groups')
#plt.ylabel('Number of Cases')
#plt.xticks(rotation=0)
#plt.legend(title='Fatal', labels=['Not Fatal', 'Fatal'])
#plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
#plt.savefig('ageGroup.png', bbox_inches='tight', pad_inches=0.05)
#plt.close()



# make table for first 10 rows
#fig, ax = plt.subplots(figsize=(10, 2))

#ax.xaxis.set_visible(False) 
#ax.yaxis.set_visible(False)
#ax.set_frame_on(False)
#ax.table(cellText=df.head(10).values, colLabels=df.columns, cellLoc='center', loc='upper left', colColours=["#F5A8E3"]*df.shape[1])
#plt.title('First 10 Rows of Data')
#plt.savefig('first10rows.png', bbox_inches='tight', pad_inches=0.05)
#plt.close()


# PLOT ACTIVITIES
#activity_fatal = pd.crosstab(df['Activity'], df['Fatal'])


#activity_fatal.plot(kind='bar', figsize=(12, 8), color=["#F5A8E3", '#A931E5'])

#plt.title('Shark Attacks by Activity')
#plt.xlabel('Activity')
#plt.ylabel('Number of Cases')
#plt.xticks(rotation=0)  # Rotate the labels to improve readability
#plt.legend(title='Fatal', labels=['Not Fatal', 'Fatal'])

#plt.tight_layout()
#plt.savefig('activityGroup.png', bbox_inches='tight', pad_inches=0.05)
#plt.show()


# PLOT SEXES
#sex_fatal = pd.crosstab(df['Sex'], df['Fatal'])


#sex_fatal.plot(kind='bar', figsize=(12, 8), color=["#F5A8E3", '#A931E5'])

#plt.title('Shark Attacks by Sex')
#plt.xlabel('Sex')
#plt.ylabel('Number of Cases')
#plt.xticks(rotation=0) 
#plt.legend(title='Fatal', labels=['Not Fatal', 'Fatal'])

#plt.tight_layout()
#plt.savefig('sexGroup.png', bbox_inches='tight', pad_inches=0.05)
#plt.show()


# PLOT TYPE
#type_fatal = pd.crosstab(df['Type'], df['Fatal'])


#type_fatal.plot(kind='bar', figsize=(12, 8), color=["#F5A8E3", '#A931E5'])

#plt.title('Shark Attacks by Type')
#plt.xlabel('Type')
#plt.ylabel('Number of Cases')
#plt.xticks(rotation=0) 
#plt.legend(title='Fatal', labels=['Not Fatal', 'Fatal'])

#plt.tight_layout()
#plt.savefig('typeGroup.png', bbox_inches='tight', pad_inches=0.05)
#plt.show()


# PLOT SPECIES
#species_fatal = pd.crosstab(df['Species'], df['Fatal'])


#species_fatal.plot(kind='bar', figsize=(12, 8), color=["#F5A8E3", '#A931E5'])

#plt.title('Shark Attacks by Species')
#plt.xlabel('Species')
#plt.ylabel('Number of Cases')
#plt.xticks(rotation=45) 
#plt.legend(title='Fatal', labels=['Not Fatal', 'Fatal'])

#plt.tight_layout()
#plt.savefig('speciesGroup.png', bbox_inches='tight', pad_inches=0.05)
#plt.show()


#PIE CHART FOR LABEL
#fatal_counts = df['Fatal'].value_counts()

#plt.figure(figsize=(8, 8))
#plt.pie(fatal_counts, labels=fatal_counts.index, autopct='%1.1f%%', startangle=140, colors=["#F5A8E3", '#A931E5'])
#plt.title('Fatal vs Non-Fatal Shark Attacks')
#plt.savefig('fatalChart.png', bbox_inches='tight', pad_inches=0.05)
#plt.show()