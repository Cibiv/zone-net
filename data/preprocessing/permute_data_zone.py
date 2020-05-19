import argparse
import csv
import logging

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='input csv-file')
parser.add_argument('-o', '--output', required=True, help='output csv-file')
args = vars(parser.parse_args())

# routine for getting next row in input file 
def get_row():    
    with open(args['input'], "rt", encoding="utf8") as org:
        reader = csv.reader(org)
        next(reader)
        for row in reader:
            yield row

# open output file
file=open(args['output'],"w", newline='')

with file:
    writes=csv.writer(file,delimiter=',')
    # write header in output file
    writes.writerow(['pProb','qProb','replicate','seqLen','aaaa','aaab','aaba','aabb','aabc','abaa','abab','abac','abba','abbb','abbc','abca','abcb','abcc','abcd','label'])
    i=0
    # get unpermuted site pattern frequencies of input file
    for row in get_row():
        i+=1
        # write all possible permutations of considered site pattern frequencies to output file
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[7],row[6],row[8],row[9],row[10],row[13],row[16],row[11],row[14],row[17],row[12],row[15],row[18],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[6],row[10],row[11],row[12],row[7],row[8],row[9],row[13],row[14],row[15],row[16],row[18],row[17],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[7],row[10],row[13],row[16],row[6],row[8],row[9],row[11],row[14],row[17],row[12],row[18],row[15],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[10],row[6],row[11],row[12],row[7],row[13],row[16],row[8],row[14],row[18],row[9],row[15],row[17],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[10],row[7],row[13],row[16],row[6],row[11],row[12],row[8],row[14],row[18],row[9],row[17],row[15],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[14],row[13],row[15],row[11],row[10],row[12],row[17],row[16],row[18],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[7],row[6],row[8],row[9],row[14],row[11],row[17],row[13],row[10],row[16],row[15],row[12],row[18],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[6],row[10],row[11],row[12],row[14],row[13],row[15],row[8],row[7],row[9],row[18],row[16],row[17],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[7],row[10],row[13],row[16],row[14],row[11],row[17],row[8],row[6],row[9],row[18],row[12],row[15],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[10],row[6],row[11],row[12],row[14],row[8],row[18],row[13],row[7],row[16],row[15],row[9],row[17],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[10],row[7],row[13],row[16],row[14],row[8],row[18],row[11],row[6],row[12],row[17],row[9],row[15],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[6],row[14],row[13],row[15],row[7],row[8],row[9],row[11],row[10],row[12],row[17],row[18],row[16],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[7],row[14],row[11],row[17],row[6],row[8],row[9],row[13],row[10],row[16],row[15],row[18],row[12],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[6],row[14],row[13],row[15],row[10],row[11],row[12],row[8],row[7],row[9],row[18],row[17],row[16],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[7],row[14],row[11],row[17],row[10],row[13],row[16],row[8],row[6],row[9],row[18],row[15],row[12],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[10],row[14],row[8],row[18],row[6],row[11],row[12],row[13],row[7],row[16],row[15],row[17],row[9],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[10],row[14],row[8],row[18],row[7],row[13],row[16],row[11],row[6],row[12],row[17],row[15],row[9],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[14],row[6],row[13],row[15],row[7],row[11],row[17],row[8],row[10],row[18],row[9],row[12],row[16],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[14],row[7],row[11],row[17],row[6],row[13],row[15],row[8],row[10],row[18],row[9],row[16],row[12],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[14],row[6],row[13],row[15],row[10],row[8],row[18],row[11],row[7],row[17],row[12],row[9],row[16],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[14],row[7],row[11],row[17],row[10],row[8],row[18],row[13],row[6],row[15],row[16],row[9],row[12],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[14],row[10],row[8],row[18],row[6],row[13],row[15],row[11],row[7],row[17],row[12],row[16],row[9],row[19],row[20]])
        writes.writerow([row[0],row[1],row[3],row[4],row[5],row[14],row[10],row[8],row[18],row[7],row[11],row[17],row[13],row[6],row[15],row[16],row[12],row[9],row[19],row[20]])

logging.info("Wrote permuted data to " + args['output'] + ".")
