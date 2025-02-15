import csv

'''
THIS FILE WAS USED TO FORMAT MY CSV FILE WITH ROW NUMBERS, ETC...
'''

def has_non_null(row):
    for cell in row:
        if cell.strip():
            return True
    return False

def format(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='latin-1') as inputfile, \
         open(output_file, 'w', newline='') as outputfile:
        r = csv.reader(inputfile)
        w = csv.writer(outputfile)
        
        # header
        w.writerow(["Row", "Activity", "Sex", "Age", "Type", "Species", "Fatal"])
        
        # row number
        row_num = -1
        
        for row in r:
            
            if r.line_num != 1:
                row_num += 1

                if has_non_null(row):
                 # select features i wanted and label
                    new_row = [row_num, row[0], row[1], row[2], row[3], row[4], row[5]]
                    w.writerow(new_row)


input_file = 'new.csv'
output_file = 'newAttacks.csv'

format(input_file, output_file)