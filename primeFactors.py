
def isPrime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True

def getFactors(n):
    if n == 1:
        return []
    if isPrime(n):
        return [n]
    i = 2
    while i * i <= n:
        if n % i == 0:
            result = getFactors(i)
            result.extend(getFactors(n // i))
            return result
        i += 1
    

if __name__ == "__main__":
    for i in range(20):
        print(i, "is prime:", isPrime(i))

    for i in range(32):
        print(i, "factors:", getFactors(i))
