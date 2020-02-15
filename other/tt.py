
lst = []
print("empty list")
print(id(list))
for i in range(10000000):
    lst.append(i)
    if i % 1000000 == 0:
        print(f"added i {i}")
        print(id(lst))