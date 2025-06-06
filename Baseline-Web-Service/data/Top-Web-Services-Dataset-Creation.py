import json
import pandas as pd
import matplotlib.pyplot as plt
import os


def return_json_from_txt(txt_path):
    '''return the json data for the given txt file'''
    with open(txt_path, encoding='utf-8') as file:
        raw = file.read()
        data = json.loads(raw, strict=False)
    return data


def extract_web_service_description_category(data, desc_key='Description', cat_key='PrimaryCategory'):
    '''with extract the columns from given pandas dataframe'''
    df = pd.DataFrame(data)
    df_final = pd.DataFrame(df[[desc_key, cat_key]])
    df_final.columns = ['Service Description', 'Service Classification']
    return df_final


def filter_top_n_web_service_categories(df, label_column='Service Classification', top_n=50):
    '''will filter data frame based on top_n category'''
    label_counts = df[label_column].value_counts()
    top_labels = label_counts.head(top_n).index
    filtered_df = df[df[label_column].isin(top_labels)]
    return filtered_df, label_counts[top_labels]


def save_to_csv(df, output_path):
    '''data frame to csv file saving'''
    df.to_csv(output_path, encoding='utf-8', index=False, header=True)


def plot_web_services_category_distribution(label_counts, title, output_path):
    plt.figure(figsize=(20, 10))
    label_counts.plot(kind='bar', fontsize=9)
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def process_multiple_top_n_web_services(txt_path, output_dir, top_n_list):
    data = return_json_from_txt(txt_path)
    df = extract_web_service_description_category(data)

    os.makedirs(output_dir, exist_ok=True)

    for n in top_n_list:
        filtered_df, label_counts = filter_top_n_web_service_categories(df, top_n=n)
        csv_path = os.path.join(output_dir, f"Top_{n}_Web_Services_Categories.csv")
        plot_path = os.path.join(output_dir, f"Top_{n}_Web_Services_Categories.png")
        
        save_to_csv(filtered_df, csv_path)
        plot_web_services_category_distribution(label_counts, f"Top {n} Web Service Categories", plot_path)
        print(f"Saved Top {n} Web Service Categories CSV and plot.")


if __name__ == "__main__":
    input_txt = "ProgrammWebScrapy.txt"
    output_dir = "top_web_services_categories_output"
    top_web_services_category_count_list = [10, 20, 30, 40, 50, 60, 70, 80, 90,100]
    process_multiple_top_n_web_services(input_txt, output_dir, top_web_services_category_count_list)
