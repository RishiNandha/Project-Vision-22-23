def generate(n):
	ret=list()
	for i in range(2**n):
		str1=str(bin(i))
		str1=str1[2:]
		while len(str1)<n:
			str1='0'+str1
		ret = ret + [str1,]
	return ret

def test(l,n):
	ret=list()
	for i in range(8):
		k = l[i:i+2*n+1:n]
		ret=ret+[k,]
	return ret
works = list()
k=2
for i in generate(8+2*k):
	if len(test(i,k))==len(set(test(i,k))):
		works = works + [i,]
		print(test(i,k))


print(works)