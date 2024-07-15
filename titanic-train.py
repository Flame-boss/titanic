import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
form = pd.read_csv("train.csv")
print(form)

# Explore data
print(form.head(3))
print(form.info())
print(form.describe())
print(form.isnull())

# data visualization seaborn
sns.countplot(x='Pclass', data= form)
plt.title('Passenger Class Distribution')
plt.show()

sns.countplot(x ='Embarked', data = form)
plt.title("Embarked distribution")
plt.show()

sns.countplot(x="Survived",data= form)
plt.title("Survived distribution")
plt.show()


sns.histplot(data=form, x='Age', bins=30, kde=True)
plt.title('Passenger Age Distribution')
plt.show()


sns.countplot(x='Sex', hue='Survived', data=form)
plt.title('Survival Count by Sex')
plt.show()

form=form[(form['Age']>0)&(form['Age']<=100)]

form.dropna(inplace=True)

sns.countplot(x="Sex",data= form)
plt.title('Distribution of gender')
plt.show()

corr_matrix = form.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

avarege_survived_by_age=form.groupby('Age')['Survived'].mean()
print(avarege_survived_by_age)


avarege_survived_by_class=form.groupby('Pclass')['Survived'].mean()
print(avarege_survived_by_class)

avarege_survived_by_sex=form.groupby('Sex')['Survived'].mean()
print(avarege_survived_by_sex)


form['Pclass'].value_counts().plot(kind = "bar") 
plt.show()

form['Pclass'].value_counts().plot(kind='pie',autopct="%0.2f")
ylabel ='Pclass'


