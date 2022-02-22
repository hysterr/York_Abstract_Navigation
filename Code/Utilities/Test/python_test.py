class Data_Array:
	def __init__(self, lst):
		self.lst = lst

	def invert(self):
		lst = [x for x in self.lst] # creates immutable instance of array 
		for i in range(len(lst)):
			# e = lst[i] 
			e = [x for x in lst[i]] # creates immutable instance of array 
			e[0], e[1] = e[1], e[0]
			lst.append(e)

		self.lst_inv = lst


a = [
		[1, 2, 0.7, 0.999],
		[1, 4, 1.2, 0.99],
		[1, 8, 2.2, 0.999],
	]

print('-'*10)
print("a = ")
for i in a:
	print(i)

obj = Data_Array(a)
obj.invert()



print('-'*10)
print("a = ")
for i in a:
	print(i)
