# Question 1
length = int(input("how many 2-dimensional coordinates there will be, "   ))

index = 0
coordinates = []
while index < length:
    point = input("please write one coordinate by splitting the points with coma: ").split(",")
    x = float(point[0])
    y = float(point[1])
    point = (x,y)
    coordinates.append(point)
    index = index + 1
print("coordinates are: ", coordinates)

# Question 2
sumx = 0
sumy = 0
lenght = len(coordinates)
for point in coordinates:
    
    sumx += point[0]
    sumy += point[1]
    
centerofmass = (sumx/lenght, sumy/lenght)
print("center of mass is: ", centerofmass)

# Question 3
import math
lenght = len(coordinates)
distances=[]
for i in range(0, lenght):

    distX = (coordinates[i][0] - centerofmass[0])**2
    distY = (coordinates[i][1] - centerofmass[1])**2
    totalDist = math.sqrt(distX + distY)
    distances.append(totalDist)
   
print("list of distances: ", distances)

# Question 4
smallest = 0
largest = 0
together = []
for i in range(0,lenght):
    if distances[i] < distances[smallest]:
        smallest = i
    if distances[i] > distances[largest]:
        largest = i
togethers = ( coordinates[smallest], distances[smallest])
togetherl = (coordinates[largest], distances[largest])
together.append(togethers)
together.append(togetherl)
print("shortest coordinate and distance to center of mass is: ", together[0]," largest coordinate and distance to center of mass is:", together[1])


    
                  
 



    




    
    
    







    
        
   
 


    
    
                
                