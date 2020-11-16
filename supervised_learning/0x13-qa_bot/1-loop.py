#!/usr/bin/env python3
"""
takes in input from the user with the prompt Q:
and prints A: as a response.
If the user inputs exit, quit, goodbye, or bye,
case insensitive, print A: Goodbye and exit.
"""

while 1:
    prompt = input("Q: ")

    if prompt.lower() in ['exit', 'goodbye', 'bye']:
        print("A: Goodbye")
        exit(0)
    else:
        print("A:")
