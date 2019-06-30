import numpy as np

N = 0
sum = np.zeros([6])

with open('l1-averages', 'a') as out:
    with open('results_sum', 'r') as res:
        for lines in res:
            try:
                if int(lines[3]) == 0:
                    N += 1
                    sum[0] += float(lines[1:9])
                    sum[1] += float(lines[11:19])
                    sum[2] += float(lines[21:29])
                    sum[3] += float(lines[31:39])
                    sum[4] += float(lines[41:49])
                    sum[5] += float(lines[51:59])
            except:
                pass
    out.write(str(sum/N))
