data_in = []
data_out = []
for i in range(0, 20):
    data_in.append([i])

for i in data_in:
    if i[0] < 10:
        data_out.append([0])
    else:
        data_out.append([1])

print(data_in)
print(data_out)