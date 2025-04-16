import numpy as np
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
import joblib
import pandas as pd
from datetime import datetime
import os


def domain_age_lessThanOne(report, create_date, update_date):
    if create_date != "" and create_date != "expired" and not pd.isna(create_date):
        age = datetime.strptime(report[:10], '%Y-%m-%d') - datetime.strptime(create_date[:10], '%Y-%m-%d')
        return (age.days // 365) < 1
    elif create_date == "" and update_date != "":
        age = datetime.strptime(report[:10], '%Y-%m-%d') - datetime.strptime(update_date[:10], '%Y-%m-%d')
        if age.days < 365:
            return None
        else:
            return False
    elif create_date == "expired":
        return True
    return None


def preprocess_data(data, features):
    preprocessed_data = data[features].copy()
    preprocessed_data['new_domain'] = None
    for index, item in preprocessed_data.iterrows():
        new_domain = domain_age_lessThanOne(item['report_date'], item['creation_date'], item['updated_date'])
        preprocessed_data.at[index, 'new_domain'] = new_domain

    preprocessed_data = preprocessed_data.drop(['report_date', 'creation_date', 'updated_date'], axis=1)

    # Transform binary values to numerical
    binary_features = [
        'control_over_dns',
        'domain_indexed',
        'is_archived',
        'known_hosting',
        'new_domain',
        'is_on_root',
        'is_subdomain'
    ]
    for feature in binary_features:
        preprocessed_data[feature] = preprocessed_data[feature].astype(float).replace({True: 1.0, False: 0.0})

    return preprocessed_data


def scale_numerical_features(data, scaler, num_features):
    data[num_features] = scaler.transform(data[num_features])
    return data


def perform_classification(initial_feature_set, data, model, sample_ids, output_file):
    label_mapping = {'Attacker Domain': 0, 'Compromised Domain': 1, 'Third_Party Platform': 2}
    # Reverse the label mapping
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    new_predictions = model.predict(data)
    new_predictions_labels = np.vectorize(reverse_label_mapping.get)(new_predictions)

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data.reset_index(drop=True, inplace=True)
    sample_ids = sample_ids.reset_index(drop=True)

    new_predictions_df = pd.DataFrame({
        'predicted': new_predictions_labels
        # 'sample_id': sample_ids
    })

    new_domain_column = data['new_domain'].replace({0: False, 1: True})
    result_df = pd.concat([new_predictions_df, initial_feature_set, new_domain_column], axis=1)

    result_df.to_csv(output_file, index=True)
    print(f"Predictions and data saved to {output_file}.")
    return result_df


def run_classification(df, model_path, scaler_path, output_predictions_path, path_prefix):
    ids = df['id']
    learnt_model = joblib.load(model_path)
    learnt_scaler = joblib.load(scaler_path)

    # List of selected features
    selected_features = [
        'report_date',
        'creation_date',
        'updated_date',
        'control_over_dns',
        'domain_indexed',
        'known_hosting',
        'is_archived',
        'is_on_root',
        'is_subdomain',
        'between_archives_distance',
        'phish_archives_distance'
    ]
    numerical_features = ['between_archives_distance', 'phish_archives_distance']

    # Preprocess the data
    transformed_data = preprocess_data(df, selected_features)
    # Scale numerical features
    scaled_data = scale_numerical_features(transformed_data, learnt_scaler, numerical_features)

    # Perform classification
    predictions = perform_classification(
        df,
        scaled_data,
        learnt_model,
        ids,
        os.path.join(path_prefix, output_predictions_path)
    )
    return predictions


def main():
    directory = "../PhishXtract-Class/PhishXtract-Class-Unlabeled"
    model_path = '../model/random_forest_model.pkl'
    scaler_path = '../model/scaler.pkl'
    path_prefix = "../results"

    # Iterate over all files in 'directory'
    for filename in os.listdir(directory):
        if filename.lower().endswith(".csv"):
            csv_path = os.path.join(directory, filename)

            df = pd.read_csv(csv_path)

            output_predictions_path = "prediction-results-" + filename

            # Call the classification pipeline
            run_classification(
                df=df,
                model_path=model_path,
                scaler_path=scaler_path,
                output_predictions_path=output_predictions_path,
                path_prefix=path_prefix
            )


if __name__ == "__main__":
    main()
