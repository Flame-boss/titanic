import os
os.makedirs("images", exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "train.csv")

form = pd.read_csv(csv_path)
form = pd.read_csv("train.csv")
print(form)

# Use seaborn only if available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# LOAD DATA


form = pd.read_csv("train.csv")

print("First 5 Rows")
print(form.head())

print("\nDataset Information")
print(form.info())

print("\nStatistical Summary")
print(form.describe())

print("\nMissing Values")
print(form.isnull().sum())


# DATA CLEANING

# Remove Cabin because it has many missing values
if "Cabin" in form.columns:
    form.drop(columns=["Cabin"], inplace=True)

# Fill missing Age values with median
form["Age"].fillna(form["Age"].median(), inplace=True)

# Fill missing Embarked values with mode
form["Embarked"].fillna(form["Embarked"].mode()[0], inplace=True)

# Remove any remaining missing values
form.dropna(inplace=True)

# Keep realistic ages
form = form[(form["Age"] > 0) & (form["Age"] <= 100)]


# FEATURE ENGINEERING

form["FamilySize"] = form["SibSp"] + form["Parch"] + 1

form["IsAlone"] = (form["FamilySize"] == 1).astype(int)

bins = [0,12,18,35,60,100]
labels = ["Child","Teen","Adult","Middle Age","Senior"]

form["AgeGroup"] = pd.cut(
    form["Age"],
    bins=bins,
    labels=labels
)


# VISUALIZATIONS

if HAS_SEABORN:

    sns.countplot(x="Pclass", data=form)
    plt.title("Passenger Class Distribution")
    plt.savefig("images/passenger_class_distribution.png",
                dpi=300,
                bbox_inches="tight")
    plt.show()

    sns.countplot(x="Embarked", data=form)
    plt.title("Embarked Distribution")
    plt.show()

    sns.countplot(x="Survived", data=form)
    plt.title("Survival Distribution")
    plt.show()

    sns.histplot(form["Age"], bins=30, kde=True)
    plt.title("Passenger Age Distribution")
    plt.show()

    sns.countplot(x="Sex", hue="Survived", data=form)
    plt.title("Survival by Gender")
    plt.show()

    sns.barplot(data=form,
                x="Pclass",
                y="Survived")
    plt.title("Survival Rate by Passenger Class")
    plt.show()

    sns.barplot(data=form,
                x="Embarked",
                y="Survived")
    plt.title("Survival Rate by Embarkation Port")
    plt.show()

    sns.boxplot(data=form,
                x="Survived",
                y="Age")
    plt.title("Age vs Survival")
    plt.show()

    sns.boxplot(data=form,
                x="Survived",
                y="Fare")
    plt.title("Fare vs Survival")
    plt.show()

    sns.barplot(data=form,
                x="AgeGroup",
                y="Survived")
    plt.title("Survival Rate by Age Group")
    plt.show()

    sns.barplot(data=form,
                x="FamilySize",
                y="Survived")
    plt.title("Survival by Family Size")
    plt.show()

    sns.scatterplot(data=form,
                    x="Age",
                    y="Fare",
                    hue="Survived")
    plt.title("Age vs Fare")
    plt.show()

    corr_matrix = form.select_dtypes(include="number").corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap="coolwarm",
                linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()


# ANALYSIS

print("\nAverage Survival by Passenger Class")
print(form.groupby("Pclass")["Survived"].mean())

print("\nAverage Survival by Gender")
print(form.groupby("Sex")["Survived"].mean())

print("\nAverage Survival by Age")
print(form.groupby("Age")["Survived"].mean())

print("\nAverage Fare by Passenger Class")
print(form.groupby("Pclass")["Fare"].mean())

print("\nPassengers by Gender")
print(form["Sex"].value_counts())

print("\nPassengers by Class")
print(form["Pclass"].value_counts())

print("\nCross Tab: Gender vs Survival")
print(pd.crosstab(form["Sex"], form["Survived"]))

print("\nCross Tab Percentage")
print(pd.crosstab(
    form["Sex"],
    form["Survived"],
    normalize="index"
) * 100)


# ADDITIONAL CHARTS

form["Pclass"].value_counts().plot(kind="bar")
plt.title("Passenger Class Count")
plt.show()

form["Pclass"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)
plt.ylabel("")
plt.title("Passenger Class Percentage")
plt.show()


# EXPORT CLEANED DATA


form.to_csv("cleaned_titanic.csv", index=False)


# SUMMARY

print("\n" + "="*50)
print("SUMMARY OF FINDINGS")
print("="*50)

print(f"Total Passengers: {len(form)}")
print(f"Overall Survival Rate: {form['Survived'].mean():.2%}")
print(f"Average Age: {form['Age'].mean():.1f} years")
print(f"Average Fare: ${form['Fare'].mean():.2f}")

print(f"Female Survival Rate: {form[form['Sex']=='female']['Survived'].mean():.2%}")
print(f"Male Survival Rate: {form[form['Sex']=='male']['Survived'].mean():.2%}")

print("Highest Survival Class:",
      form.groupby("Pclass")["Survived"].mean().idxmax())

plt.savefig("images/passenger_class_distribution.png",
            dpi=300,
            bbox_inches="tight")

print("\nProject Completed Successfully.")
