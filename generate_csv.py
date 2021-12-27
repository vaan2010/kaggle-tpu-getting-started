import pandas as pd
def generate_result(data_list, TPU=False):
    submission_df = pd.DataFrame(data=data_list)
    print(submission_df)

    if TPU==False:
        path = './Result/submission.csv'
    else:
        path = 'submission.csv'

    submission_df.to_csv(
        path_or_buf=path, 
        sep=',',
        header=True, 
        index=False
    )