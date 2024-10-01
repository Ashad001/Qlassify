import csv

def convert_statlog_to_csv(input_file, output_file):
    with open(input_file, 'r') as dat_file:
        data = [line.strip().split() for line in dat_file.readlines()]

    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        header = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 
                  'fasting_blood_sugar', 'resting_electrocardiographic', 'max_heart_rate', 
                  'exercise_induced_angina', 'oldpeak', 'slope', 'major_vessels', 'thal', 'heart_disease']
        writer.writerow(header)
        
        writer.writerows(data)

    print("[+] Data successfully converted to CSV")

convert_statlog_to_csv('./data/statlog_heart/heart.dat', './data/statlog_heart/heart_data.csv')
