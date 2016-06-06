
for filename in os.listdir(input_path):
        input_file = open(filename,'r')
        x = input_file.read()
        if x[0:3] is '—':
           output = open(filename,'w')
           output.write(x[4:])