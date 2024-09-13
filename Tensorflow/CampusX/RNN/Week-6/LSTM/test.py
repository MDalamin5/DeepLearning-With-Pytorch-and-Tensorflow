num = [1, 2, 3, 4, 5, 6]
for i in range(1, len(num)):
    print(num[:i+1])
    
    """
    output:
        [1, 2]
        [1, 2, 3]
        [1, 2, 3, 4]
        [1, 2, 3, 4, 5]
        [1, 2, 3, 4, 5, 6]
    """