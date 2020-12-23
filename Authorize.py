def init(string):
    authorized = False

    file = open("allowed.txt", 'r')

    while True:
        line = file.readline().rstrip('\n')
        if line == string:
            authorized = True
            break

        if not line:
            break

    file.close()

    return authorized
