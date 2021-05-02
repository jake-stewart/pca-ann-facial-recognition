def num_input(prompt, default, min_n, max_n, type_func):
    if default:
        prompt += " (default: {}): ".format(default)
    else:
        prompt += ": "
    while True:
        user_input = input(prompt)
        if not user_input and default:
            return default
        try:
            num = type_func(user_input)

            if min_n == None:
                if max_n == None:
                    return num
                if max_n >= num:
                    return num
            if max_n == None:
                if min_n <= num:
                    return num
            if min_n <= num:
                if max_n >= num:
                    return num

            if min_n != None and max_n != None:
                print("Value must be between {} and {}.\n".format(min_n, max_n))
            elif min_n != None:
                print("Value cannot be lower than {}.\n".format(min_n))
            else:
                print("Value cannot be higher than {}.\n".format(max_n))
        except ValueError:
            print("Invalid input, try again.\n")

def input_int(prompt, default=None, min_n=None, max_n=None):
    return num_input(prompt, default, min_n, max_n, int)

def input_float(prompt, default=None, min_n=None, max_n=None):
    return num_input(prompt, default, min_n, max_n, float)

def alert(message):
    print("\n" + message + "\n")
    input("Press Enter to continue.")
