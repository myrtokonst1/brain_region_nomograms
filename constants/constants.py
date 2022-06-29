data_path = '/Users/myrtokonstantinidi/Desktop/UNI/projecttt/Data'

main_ukb_data_path = f'{data_path}/ukb_features_20200830.csv'
ukb_brain_data_path = f'{data_path}/ukb_distributed.csv'
ukb_preprocessed_data_path = f'{data_path}/ukb_preprocessed_table.csv'

percentiles = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
quantiles = [percentile/100 for percentile in percentiles]

gaussian_width = 20
