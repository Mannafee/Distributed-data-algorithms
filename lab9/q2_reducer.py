

#Reducer.py
import sys


(last_airport,max_depdelay)=(None,0)
(last_airport,min_depdelay)=(None,0)
airport_arrdelay = {}
airport_depdelay = {}
ave_delay_rank={}

      

for line in sys.stdin:
    data = line.strip().split('\t')
    
    if len(data) != 3:
        # Something has gone wrong. Skip this line.
        continue
    airport,arrdelay,depdelay = data
    
    if airport in airport_arrdelay:
        airport_arrdelay[airport].append(int(arrdelay))
    else:
        airport_arrdelay[airport] = []
        airport_arrdelay[airport].append(int(arrdelay))
        
    if airport in airport_depdelay:
        airport_depdelay[airport].append(int(depdelay))
    else:
        airport_depdelay[airport] = []
        airport_depdelay[airport].append(int(depdelay))
   
    if last_airport and last_airport!=airport:
        print('maximum deperature delay:','%s\t%s'% (last_airport, max_depdelay))
        print('minimum departure delay:','%s\t%s'% (last_airport, min_depdelay))
        
        (last_airport,max_depdelay)=(airport,int(depdelay))
        (last_airport,min_depdelay)=(airport,int(depdelay))
    else:
        (last_airport,max_depdelay)=(airport,max(max_depdelay,int(depdelay)))
        (last_airport,min_depdelay)=(airport,min(min_depdelay,int(depdelay)))
        
        
for airport in airport_arrdelay.keys():
    ave_delay = sum(airport_arrdelay[airport])*1.0 / len(airport_arrdelay[airport])
    ave_delay_rank[airport] = []
    ave_delay_rank[airport].append(float(ave_delay))
    
for airport in airport_depdelay.keys():
    ave_delay_dep = sum(airport_depdelay[airport])*1.0 / len(airport_depdelay[airport])
    print ('average departure delay','%s\t%s'% (airport, ave_delay_dep))
    
sorted_x = sorted(ave_delay_rank.items(), key=lambda kv: kv[1])
print("Top 10 airports by their average Arrival delay")
count=0
for keys,values in sorted_x:
    if count<10:
        print('%s\t%s'% (keys,values[0]))
        count+=1
    