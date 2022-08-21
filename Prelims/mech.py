def generate(n):
	ret=list()
	for i in range(2**(n-4)):
		str1=str(bin(i))
		str1=str1[2:]
		while (len(str1)<(n-4)):
			str1 = '0'+str1
		str1 = '0'+str1[0:2] + '0'+str1[2:4] + '0'+str1[4:6]+'1'+str1[6:]
		ret = ret + [str1,]
	return ret

def test(l):
	ret=list()
	for i in range(8):
		k = l[i:i+7:3]
		ret=ret+[k,]
	return ret
works = list()
for i in generate(14):
	if len(test(i))==len(set(test(i))):
		works = works + [i,]
		print(test(i))
