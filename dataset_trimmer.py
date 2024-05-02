import csv

def filter_csv(input_file, output_file, min_count):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            header = next(reader)
            writer.writerow(header)

            id_data = {}

            curr_id = next(reader)[0]
            
            for row in reader:
                id_value = row[0]
                
                if id_value not in id_data:
                    id_data[id_value] = {
                        'count': 1,
                        'rows': [row]
                    }
                else:
                    id_data[id_value]['count'] += 1
                    id_data[id_value]['rows'].append(row)
                
                if id_data[curr_id]['count'] >= min_count and id_value != curr_id:
                    for r in id_data[curr_id]['rows']:
                        writer.writerow(r)
                    curr_id = id_value

                elif id_value != curr_id:
                    curr_id = id_value

def protocol_1_filter_CSV(input_file, output_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            header = next(reader)
            writer.writerow(header)

            for row in reader:
                if row[24] == '1':
                    writer.writerow(row)

def csv_info(csv_file):
    # this is for reading out relevant info about the csv -
    #   - number of different birds
    #   - average data points per bird
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        my_dict = {}


input_file = 'AVONET_Raw_Data.csv'

intermediate_file = 'AVONET_Protocol_1_Data.csv'

output_file = 'AVONET_Trimmed_Data.csv'

protocol_1_filter_CSV(input_file, intermediate_file)

filter_csv(intermediate_file, output_file, 10)
