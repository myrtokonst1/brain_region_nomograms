project_path = '/Users/myrtokonstantinidi/Desktop/UNI/projecttt'
data_path = f'{project_path}/Data'

main_ukb_data_path = f'{data_path}/ukb_features_20200830.csv'
ukb_brain_data_path = f'{data_path}/ukb_distributed.csv'
ukb_preprocessed_data_path = f'{data_path}/ukb_preprocessed_table.csv'
ukb_participant_exclusion_data_path = f'{data_path}/ukb_participant_exclusion.csv'
pgs_data_path = f'{data_path}/PGS_AV.all_score'


def get_gpr_filename(hemisphere, sex, extra=None):
    filename = f'{project_path}/biobank_data/saved_data/GPR_{hemisphere.get_name()}_{sex.get_name()}'
    if extra is not None:
        filename+= f'_{extra}'

    filename+='.csv'

    return filename
