import pandas as pd
import altair as alt

data = pd.read_csv('merged_dataset.csv')
print(data.columns)
print(data.shape)
print(data.describe())

# EDA
"""
Relationship between age group and mooc completion rate
"""
print(data.groupby(['age_group', 'completed_mooc'])['completed_mooc'] \
      .agg(ageTot='count').reset_index())

"""
Relationship between number of prev attempt and mooc completion rate
"""
print(data.groupby(['num_of_prev_attempts', 'completed_mooc'])['completed_mooc'] \
      .agg(prevAttpTot='count').reset_index())

"""
Relationship between gender and mooc completion rate
"""
print(data.groupby(['gender', 'completed_mooc'])['completed_mooc'] \
      .agg(prevAttpTot='count').reset_index())

gender_mapping = {"M": 1, "F": 2}
data["gender"] = data["gender"].map(gender_mapping)

n_by_region = data.groupby("region").count()
print(n_by_region)

n_by_education = data.groupby("highest_education").count()
print(n_by_education)
prev_education_mapping = {"No Formal quals": 1, "Lower Than A Level": 2, "A Level or Equivalent": 3,
                          "HE Qualification ": 4, "Post Graduate Qualification": 5}
data["highest_education"] = data["highest_education"].map(prev_education_mapping)

n_by_disability = data.groupby("disability").count()
print(n_by_disability)
disability_mapping = {"N": 1, "Y": 2}
data["disability"] = data["disability"].map(disability_mapping)


data_cleaned = pd.DataFrame(data)
data_cleaned.to_csv("new_data.csv")