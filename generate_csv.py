import pandas as pd
def generate_result(data_list):
    submission_df = pd.DataFrame(data=data_list)
    print(submission_df)

    submission_df.to_csv(
        path_or_buf='./Result/submission.csv', 
        sep=',',
        header=True, 
        index=False
    )