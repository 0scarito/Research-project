import pandas as pd

# this is the config file, which contains all the necessary variables. It lets the user understand things more easily

# define filepaths in advance. This is because it's more harmonious to keep them all in one file, rather than to directly mention them in the code files
input_path = "data/AgriRiskFin_Dataset.csv"
figure_output_path = "outputs/figures"
table_output_path = "outputs/tables"
report_output_path = "outputs/reports"

# allows us to plot three different scenarios where humanity deals with global warming, and how those scenarios affect carbon price
# carbon price therefore affects strandedness of assets
carbon_price_scenarios = {
    "Delayed Transition": 10,
    "Net Zero(NZ) 2050": 110,
    "Divergent Net Zero": 300
}

# miscellaneous variables. Those are just numbers/booleans that you put into model functions and other things.

# kfold model variables
# I made random_state to be 37. I just did it for fun.
# Because of the data's strong internal consistency, and the mechanical connection between revenue/expenses/profit variables, it does not matter
split_number = 5
shuffle_truth = True
random_state = 37